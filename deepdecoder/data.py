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
import h5py

from deepdecoder.utils import np_binary_mask
from deepdecoder.grid_curriculum import exam, Uniform, grids_from_lecture, \
    DISTRIBUTION_PARAMS
from beras.data_utils import HDF5Tensor
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


def normalize_grid_params(grid_params):
    p = DISTRIBUTION_PARAMS
    bits = grid_params[:, :NUM_MIDDLE_CELLS]
    # map bits to (-1, 1)
    bits = 2*(bits + 1) - 3

    configs = grid_params[:, NUM_MIDDLE_CELLS:]
    angles = configs[:, CONFIG_ROTS]
    rot_z = angles[:, :1]
    sin_z = np.sin(rot_z)
    cos_z = np.cos(rot_z)
    rot_y = angles[:, 1:2] / p.y.std
    rot_x = angles[:, 2:] / p.x.std
    r = (configs[:, (CONFIG_RADIUS,)] - p.radius.mean) / p.radius.std
    xy = (configs[:, CONFIG_CENTER] - p.center.mean) / p.center.std
    return np.concatenate(
        [bits, sin_z, cos_z, rot_y, rot_x,  xy, r], axis=1)


def num_normalized_params():
    return len(next(grids_lecture_generator(batch_size=1)))


def normalize_generator(generator):
    for grid_params, grid_idx in generator:
        yield normalize_grid_params(grid_params), grid_idx


def grids_lecture_generator(batch_size=128, lecture=None):
    if lecture is None:
        lecture = exam()
    while True:
        params, grid_idx = grids_from_lecture(lecture, batch_size)
        yield normalize_grid_params(params), grid_idx


def load_real_hdf5_tags(fname, batch_size, batches_per_epochs):
    h5 = h5py.File(fname, 'r')
    epoch_size = batches_per_epochs*batch_size
    nb_tags = h5['tags'].shape[0]
    nb_tags = (nb_tags // epoch_size)*epoch_size
    tags = HDF5Tensor(fname, 'tags', 0, nb_tags)
    assert len(tags) % epoch_size == 0
    return tags
