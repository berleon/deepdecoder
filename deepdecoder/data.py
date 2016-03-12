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

from deepdecoder.grid_curriculum import exam, grids_from_lecture, \
    DISTRIBUTION_PARAMS, normalize
from deepdecoder.utils import np_binary_mask
from beras.data_utils import HDF5Tensor
from itertools import count

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


def normalize_grid_params(grid_params, lecture=exam()):
    return normalize(lecture, grid_params)


def nb_normalized_params():
    params, _ = next(grids_lecture_generator(batch_size=1))
    return params.shape[-1]


def normalize_generator(generator):
    for grid_params, grid_idx in generator:
        yield normalize_grid_params(grid_params), grid_idx


def grids_lecture_generator(batch_size=128, lecture=None):
    if lecture is None:
        lecture = exam()
    while True:
        params, grid_idx = grids_from_lecture(lecture, batch_size)
        yield normalize_grid_params(params), grid_idx


def mean_generator(batch_size=128, mean_distance=0.2):
    while True:
        black = np.random.uniform(0, 0.5, (batch_size, 1))
        white = np.random.uniform(0, 1., (batch_size, 1))
        white *= 1 - (black + mean_distance)
        white += black + mean_distance
        yield np.concatenate([black, white], axis=1)


def load_real_hdf5_tags(fname, batch_size):
    h5 = h5py.File(fname, 'r')
    nb_tags = h5['tags'].shape[0]
    nb_tags = (nb_tags // batch_size)*batch_size
    tags = HDF5Tensor(fname, 'tags', 0, nb_tags)
    assert len(tags) % batch_size == 0
    return tags


def real_generator(hdf5_fname, nb_real, use_mean_image=False):
    tags = load_real_hdf5_tags(hdf5_fname, nb_real)
    nb_tags = len(tags)
    print("Got {} real tags".format(nb_tags))
    mean_end = min(nb_tags, 2000)
    mean_image = (tags[0:mean_end] / 255).mean(axis=0)

    for i in count(step=nb_real):
        ti = i % nb_tags
        tag_batch = tags[ti:ti+nb_real] / 255
        if use_mean_image:
            tag_batch -= mean_image
        yield tag_batch


def z_generator(z_shape):
    while True:
        yield np.random.uniform(-1, 1, z_shape)


def zip_real_z(real_gen, z_gen):
    for real, z in zip(real_gen, z_gen):
        yield {'real': real, 'z': z}


def param_mean_grid_idx_generator(batch_size=128, lecture=None):
    if lecture is None:
        lecture = exam()

    for mean, (param, grid_idx) in zip(
            mean_generator(batch_size),
            grids_lecture_generator(batch_size, lecture)):
        yield np.concatenate([param, mean], axis=1), grid_idx


def grid_with_mean(grid_idx, mean):
    black = np_binary_mask(grid_idx, black=1, ignore=0, white=0)
    white = np_binary_mask(grid_idx, black=0, ignore=0, white=1)
    mask = np.zeros_like(black)
    mask += black * mean[:, 0].reshape(-1, 1, 1, 1)
    mask += white * mean[:, 1].reshape(-1, 1, 1, 1)
    return mask
