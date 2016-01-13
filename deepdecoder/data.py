# Copyright 2016 Leon Sixt
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

from beesgrid import GridGenerator, MaskGridArtist, generate_grids, \
    NUM_MIDDLE_CELLS, CONFIG_ROTS, CONFIG_RADIUS, \
    CONFIG_CENTER, TAG_SIZE

from math import pi
import numpy as np
import theano
import itertools

from deepdecoder.utils import np_binary_mask
from deepdecoder.grid_curriculum import exam, Uniform, grids_from_lecture
from dotmap import DotMap
floatX = theano.config.floatX


def normalize_angle(angle, lower_bound=0):
    two_pi = 2*pi
    angle = angle % two_pi
    angle = (angle + two_pi) % two_pi
    angle[angle > lower_bound + two_pi] -= two_pi
    return angle


def bins_for_z(z):
    z = normalize_angle(z, lower_bound=-pi/4)
    bins = np.round(z / (pi/2))
    z_diffs = z - bins*pi/2
    return bins, z_diffs


def gen_mask_grids(nb_batches, batch_size=128, scales=[1.]):
    generator = GridGenerator()
    artist = MaskGridArtist()
    gen_grids = generate_grids(batch_size, generator, artist=artist,
                               with_gird_params=True, scales=scales)
    if nb_batches == 'forever':
        counter = itertools.count()
    else:
        counter = range(nb_batches)
    for i in counter:
        masks = next(gen_grids)
        yield (masks[0].astype(floatX),) + tuple(masks[1:])


gen_diff_all_outputs = ('grid_idx', 'grid_diff', 'grid_bw')


def gen_diff_gan(batch_size=128, outputs=gen_diff_all_outputs):
    def grid_exam_generator():
        def lecture():
            lec = exam()
            lec.z = Uniform(pi/4, -pi/4)
            return lec
        while True:
            yield grids_from_lecture(lecture(), batch_size)

    def angles_to_sin_cos(grid_params):
        configs = grid_params[:, NUM_MIDDLE_CELLS:]
        angles = configs[:, CONFIG_ROTS]
        sin = np.sin(angles)
        cos = np.cos(angles)
        r = configs[:, (CONFIG_RADIUS,)] / 25. - 1
        xy = configs[:, CONFIG_CENTER] / (TAG_SIZE/2) - 1
        return np.concatenate(
            [grid_params[:, :NUM_MIDDLE_CELLS], sin, cos, xy, r], axis=1)

    for grid_params, grid_idx in grid_exam_generator():
        z_bins = np.random.choice(4, batch_size)
        batch = DotMap({
            'z_bins': z_bins[:, np.newaxis],
            'params': angles_to_sin_cos(grid_params),
        })
        if 'grid_idx' in outputs:
            batch.grid_idx = grid_idx
        if 'grid_diff' in outputs:
            batch.grid_diff = np_binary_mask(grid_idx, ignore=0., white=0.5)
        if 'grid_bw' in outputs:
            batch.grid_bw = np_binary_mask(grid_idx)
        yield batch
