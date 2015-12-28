# Copyright 2015 Leon Sixt
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import numpy as np
from math import ceil

import pytest
from beesgrid.pybeesgrid import NUM_MIDDLE_CELLS, NUM_CONFIGS, MaskGridArtist
from beesgrid import draw_grids
from dotmap import DotMap
from scipy.ndimage import gaussian_filter1d


def to_radians(x):
    return x / 180. * np.pi


def config_distributions(l):
    z_low, z_high = (-l*np.pi, l*np.pi)
    y_mean, y_std = (0, l*to_radians(12))
    x_mean, x_std = (0, l*to_radians(10))
    center_mean, center_std = (0, 2)
    radius_mean, radius_std = (24.5, 1)
    id_sigma = 1*(1.-l)
    d = DotMap(locals())
    return d


def _config(d, batch_size):
    col_shape = (batch_size, 1)
    eps = 1e-7
    rot_z = np.random.uniform(d.z_low, d.z_high, col_shape)
    rot_y = np.random.normal(d.y_mean, d.y_std + eps, col_shape)
    rot_x = np.random.normal(d.x_mean, d.x_std + eps, col_shape)
    center = 25*np.ones((batch_size, 2)) + \
        np.random.normal(d.center_mean, d.center_std + eps, (batch_size, 2))
    radius = np.random.normal(d.radius_mean, d.radius_std, col_shape)
    return np.concatenate([rot_z, rot_x, rot_y, center, radius], axis=1)


def _sample_ids(sigma, batch_size):
    ids = np.random.binomial(1, 0.5, (batch_size, NUM_MIDDLE_CELLS))
    #id_idx = gaussian_filter1d(ids, sigma)
    #simple_ids = np.zeros_like(ids)
    #simple_ids[id_idx > 0.25] = 1.
    return ids


def curriculum_grids(h, batch_size=128, scale=1., artist=None):
    """
    :param h: Hardness parameter. Between 0 and 1. Higher values result in more
    complicate grids
    :return: Mask grids with  (batch_size, 1, 64*scale, 64*scale) shape
    """
    if artist is None:
        artist = MaskGridArtist()

    # identity hardness mapping
    l = h
    d = config_distributions(l)
    configs = _config(d, batch_size)
    ids = _sample_ids(d.id_sigma, batch_size)
    grids = draw_grids(ids.astype(np.float32), configs.astype(np.float32),
                       scales=[scale], artist=artist)
    assert len(grids) == 1
    return grids[0]


def curriculum_grids_generator(batch_size, learn_stepness=0.05,
                               scale=1., artist=None):
    hardness = 0
    while True:
        yield curriculum_grids(hardness, batch_size, scale, artist)
        hardness = min(1., hardness + learn_stepness)

