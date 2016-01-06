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
import math
import numpy as np
from math import ceil, pi

import pytest
from beesgrid.pybeesgrid import NUM_MIDDLE_CELLS, NUM_CONFIGS, MaskGridArtist
from beesgrid import draw_grids
from dotmap import DotMap
from keras.backend import epsilon
from scipy.ndimage import gaussian_filter1d


def to_radians(x):
    return x / 180. * np.pi

class Distribution:
    def sample(self, shape):
        raise NotImplementedError

class NormalDistribution(Distribution):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def sample(self, shape):
        eps = 1e-7
        return np.random.normal(self.mean, self.std + eps, shape)


class UniformDistribution(Distribution):
    def __init__(self, low, high):
        self.low = low
        self.high = high

    def sample(self, shape):
        return np.random.uniform(self.low, self.high, shape)


class ZDistribution(Distribution):
    def __init__(self, hardness):
        self.hardness = hardness

    def sample(self, shape):
        if self.hardness <= 0.5:
            max_axis = 16
            nb_axis = min(math.ceil(2*self.hardness*max_axis) + 1, max_axis)
            offsets = 2*pi/nb_axis * np.random.choice(nb_axis, shape)
            std = pi/16 + epsilon()
            zs = np.random.normal(0, std, shape)
            return zs + offsets
        else:
            return np.random.uniform(-pi, pi, shape)
        

def config_distributions(l):
    z = ZDistribution(l)
    y = NormalDistribution(0, l*to_radians(12))
    x = NormalDistribution(0, l*to_radians(10))
    center = NormalDistribution(0, 2)
    radius = NormalDistribution(24.5, 1)
    d = DotMap(locals())
    return d


def _config(d, batch_size):
    col_shape = (batch_size, 1)
    rot_z = d.z.sample(col_shape)
    rot_y = d.y.sample(col_shape)
    rot_x = d.x.sample(col_shape)
    center = 25*np.ones((batch_size, 2)) + d.center.sample((batch_size, 2))
    radius = d.radius.sample(col_shape)
    return np.concatenate([rot_z, rot_x, rot_y, center, radius], axis=1)


def _sample_ids(sigma, batch_size):
    return np.random.binomial(1, 0.5, (batch_size, NUM_MIDDLE_CELLS))


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
    return np.concatenate([ids, configs], axis=1), grids[0]


def curriculum_grids_generator(batch_size, learn_stepness=0.05,
                               scale=1., artist=None):
    hardness = 0
    while True:
        yield curriculum_grids(hardness, batch_size, scale, artist)
        hardness = min(1., hardness + learn_stepness)

