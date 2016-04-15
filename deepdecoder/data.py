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
    NUM_MIDDLE_CELLS

from math import pi
import numpy as np
import theano
import itertools
import h5py

from deepdecoder.grid_curriculum import exam, grids_from_lecture, normalize
from beesgrid import MASK, BlackWhiteArtist
from beras.data_utils import HDF5Tensor
from itertools import count
from skimage.transform import pyramid_reduce, \
    pyramid_expand

import scipy.ndimage.interpolation
import scipy.ndimage


floatX = theano.config.floatX


def np_binary_mask(mask, black=0., ignore=0.5,  white=1.):
    bw = ignore * np.ones_like(mask, dtype=np.float32)
    bw[mask > MASK["IGNORE"]] = white
    bw[mask < MASK["BACKGROUND_RING"]] = black
    return bw


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


def normalize_grid_params(grid_params, lecture):
    return normalize(lecture, grid_params)


def nb_normalized_params():
    params, _ = next(grids_lecture_generator(batch_size=1))
    return params.shape[-1]


def normalize_generator(generator, lecture):
    for grid_params, grid_idx in generator:
        yield normalize_grid_params(grid_params, lecture), grid_idx


def grids_lecture_generator(batch_size=128, lecture=None, scale=1, artist=None):
    if lecture is None:
        lecture = exam()
    while True:
        params, grid_idx = grids_from_lecture(lecture, batch_size,
                                              scale=scale, artist=artist)
        yield normalize_grid_params(params, lecture), grid_idx


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


def real_generator(hdf5_fname, nb_real, use_mean_image=False, range=(0, 1)):
    low, high = range
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
        yield (high - low)*tag_batch + low


def weight_pyramid(generator, weights=[1, 1, 1]):
    nb_layers = len(weights) - 1
    for batch in generator:
        batch_merged = []
        for img in batch:
            img = img[0]
            lap_pyr = []
            prev = img
            for i in range(nb_layers):
                gauss = pyramid_reduce(prev)
                lap_pyr.append(prev - pyramid_expand(gauss))
                prev = gauss

            merged = gauss*weights[0]
            for i, lap in enumerate(reversed(lap_pyr)):
                merged = pyramid_expand(merged) + weights[i+1]*lap
            batch_merged.append(merged)
        yield np.stack(batch_merged).reshape(batch.shape)


def z_generator(z_shape):
    while True:
        yield np.random.uniform(-1, 1, z_shape).astype(np.float32)


def zip_real_z(real_gen, z_gen):
    for real, z in zip(real_gen, z_gen):
        yield {'real': real, 'z': z}


def param_mean_grid_idx_generator(batch_size=128, lecture=None,
                                  mean_distance=0.2):
    if lecture is None:
        lecture = exam()

    for mean, (param, grid_idx) in zip(
            mean_generator(batch_size, mean_distance),
            grids_lecture_generator(batch_size, lecture)):
        yield self.selection_threshold.concatenate([param, mean], axis=1), grid_idx


def grid_with_mean(grid_idx, mean):
    black = np_binary_mask(grid_idx, black=1, ignore=0, white=0)
    white = np_binary_mask(grid_idx, black=0, ignore=0, white=1)
    mask = np.zeros_like(black)
    mask += black * mean[:, 0].reshape(-1, 1, 1, 1)
    mask += white * mean[:, 1].reshape(-1, 1, 1, 1)
    return mask


MASK_MEAN_PARTS = [
    ("BACKGROUND_RING", "IGNORE"),
    ("INNER_BLACK_SEMICIRCLE",),
    ("CELL_0_BLACK", "CELL_0_WHITE"),
    ("CELL_1_BLACK", "CELL_1_WHITE"),
    ("CELL_2_BLACK", "CELL_2_WHITE"),
    ("CELL_3_BLACK", "CELL_3_WHITE"),
    ("CELL_4_BLACK", "CELL_4_WHITE"),
    ("CELL_5_BLACK", "CELL_5_WHITE"),
    ("CELL_6_BLACK", "CELL_6_WHITE"),
    ("CELL_7_BLACK", "CELL_7_WHITE"),
    ("CELL_8_BLACK", "CELL_8_WHITE"),
    ("CELL_9_BLACK", "CELL_9_WHITE"),
    ("CELL_10_BLACK", "CELL_10_WHITE"),
    ("CELL_11_BLACK", "CELL_11_WHITE"),
    ("OUTER_WHITE_RING",),
    ("INNER_WHITE_SEMICIRCLE",),
]


def np_mean_mask(grid_idx, means):
    mask = np.zeros_like(grid_idx, dtype=np.float32)
    for i in range(len(grid_idx)):
        for c, mask_idxs in enumerate(MASK_MEAN_PARTS):
            idx = np.zeros_like(mask[i], dtype=np.bool)
            for mask_idx in mask_idxs:
                idx = np.logical_or(idx, np.equal(MASK[mask_idx], grid_idx[i]))

            mask[i, idx] = means[i, c]
    return mask


def resize_mask(masks, order=1, sigma=0.66, scale=0.5):
    resized = []
    for mask in masks:
        smoothed = scipy.ndimage.gaussian_filter(mask[0], sigma=sigma)
        small = scipy.ndimage.interpolation.zoom(smoothed, (scale, scale),
                                                 order=order)
        resized.append(small)
    new_size = int(masks.shape[-1] * scale)
    return np.stack(resized).reshape((len(masks), 1, new_size, new_size))


def param_mask_binary_generator(lecture, batch_size=128, scale=1,
                                antialiasing=4):
    ignore = -0.25
    background = 0
    black = 51
    white = 255
    artist = BlackWhiteArtist(black, white, background, antialiasing)
    for param, grids in grids_lecture_generator(batch_size, lecture,
                                                scale=scale, artist=artist):
        noise_scale = 0.9
        continouse_bits = np.random.uniform(-noise_scale, noise_scale,
                                            (len(param), NUM_MIDDLE_CELLS))
        bits = param[:, :NUM_MIDDLE_CELLS]
        continouse_bits = (continouse_bits + bits) / (1 + noise_scale)
        need_size = (1 - ignore)
        scale = 255 / need_size
        grids_float = (grids / scale + ignore).astype(np.float32)
        param = np.concatenate(
            [continouse_bits, param[:, NUM_MIDDLE_CELLS:]], axis=1)
        yield param, grids_float
