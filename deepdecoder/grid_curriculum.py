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
import threading

import numpy as np
from math import pi

from beesgrid import draw_grids, NUM_MIDDLE_CELLS, MaskGridArtist, \
    CONFIG_ROTS, CONFIG_CENTER, CONFIG_RADIUS, GRID_STRUCTURE_POS

from keras.backend import epsilon
from keras import callbacks
from dotmap import DotMap


def to_radians(x):
    return x / 180. * np.pi

DISTRIBUTION_PARAMS = DotMap({
    'z': {'low': -pi, 'high': pi},
    'y': {'mean': 0, 'std': to_radians(12)},
    'x': {'mean': 0, 'std': to_radians(10)},
    'center': {'mean': 0, 'std': 0.5},
    'radius': {'mean': 24.0, 'std': 0.4},
})


class Distribution:
    def sample(self, shape):
        raise NotImplementedError

    def normalize(self, array):
        return array

    def __eq__(self, other):
        return type(self) == type(other)


class Default(Distribution):
    def __init__(self, distribution):
        self.distribution = distribution

    def sample(self, shape):
        return self.distribution.sample(shape)

    def normalize(self, shape):
        return self.distribution.normalize(shape)

    def __eq__(self, other):
        return super().__eq__(other) and \
            self.distribution == other.distribution


class Zeros(Distribution):
    def sample(self, shape):
        return np.zeros(shape)


class Constant(Distribution):
    def __init__(self, value):
        self.value = value

    def sample(self, shape):
        return self.value*np.ones(shape)

    def normalize(self, array):
        return np.random.uniform(-1, 1, array.shape)


class Normal(Distribution):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def sample(self, shape):
        eps = 1e-7
        return np.random.normal(self.mean, self.std + eps, shape)

    def __eq__(self, other):
        return super().__eq__(other) and \
            self.mean == other.mean and self.std == other.std

    def normalize(self, array):
        return (array - self.mean) / (2*self.std)

    def __neq__(self, other):
        return not self.__eq__(other)


class TruncNormal(Distribution):
    def __init__(self, a, b, mean, std):
        self.a = a
        self.b = b
        self.mean = mean
        self.std = std

    def sample(self, shape):
        eps = 1e-7
        a = self.a / self.std
        b = self.b / self.std
        return scipy.stats.truncnorm.rvs(
            a, b, self.mean, self.std + eps, shape)

    def __eq__(self, other):
        return super().__eq__(other) and \
            self.mean == other.mean and self.std == other.std and \
            self.a == other.a and self.b == other.b

    def normalize(self, array):
        return 2*(array - (self.mean + self.a)) / (self.b - self.a) - 1

    def __neq__(self, other):
        return not self.__eq__(other)


class SinCosAngleNorm(Distribution):
    def __init__(self, distribution):
        self.distribution = distribution

    def sample(self, shape):
        return self.distribution.sample(shape)

    def normalize(self, array):
        s = np.sin(array)
        c = np.cos(array)
        return np.concatenate([s, c], axis=-1)


class Uniform(Distribution):
    def __init__(self, low, high):
        self.low = low
        self.high = high

    def sample(self, shape):
        return np.random.uniform(self.low, self.high, shape)

    def __eq__(self, other):
        return super().__eq__(other) and \
            self.low == other.low and self.high == other.high

    def normalize(self, array):
        return 2*(array - self.low) / (self.high - self.low) - 1


class DiscretePoints(Distribution):
    def __init__(self, low, high, nb_points, sigma):
        self.low = low
        self.high = high
        self.nb_points = nb_points
        self.sigma = sigma

    def sample(self, shape):
        around = np.linspace(self.low, self.high, num=self.nb_points)
        return np.random.choice(around, size=shape) + \
            np.random.normal(0, self.sigma)

    def __eq__(self, other):
        return super().__eq__(other) and \
            self.low == other.low and self.high == other.high and \
            self.nb_points == other.nb_points and self.sigma == other.sigma

    def normalize(self, array):
        return 2 * (array - self.low) / self.high - 1


class Bernoulli(Distribution):
    def sample(self, shape):
        return np.random.binomial(1, 0.5, shape)

    def normalize(self, array):
        return 2*array - 1


class Lecture:
    def __init__(self, pass_limit=0.01, name=''):
        self.z = Default(Zeros())
        self.x = Default(Zeros())
        self.y = Default(Zeros())
        self.ids = Default(Bernoulli())
        self.center = Default(Normal(DISTRIBUTION_PARAMS.center.mean,
                                     DISTRIBUTION_PARAMS.center.std))
        self.radius = Default(Normal(DISTRIBUTION_PARAMS.radius.mean,
                                     DISTRIBUTION_PARAMS.radius.std))

        self.inner_ring_radius = Default(Constant(0.4))
        self.middle_ring_radius = Default(Constant(0.8))
        self.outer_ring_radius = Default(Constant(1))
        self.bulge_factor = Default(Constant(0.4))
        self.focal_length = Default(Constant(2))

        self.pass_limit = pass_limit
        self.name = name

    @staticmethod
    def from_params(params, name=''):
        p = params
        lec = Lecture()
        lec.name = name
        lec.z = SinCosAngleNorm(Uniform(p['z']['low'], p['z']['high']))
        lec.y = Normal(p['y']['mean'], p['y']['std'])
        lec.x = Normal(p['x']['mean'], p['x']['std'])
        lec.center = Normal(p['center']['mean'], p['center']['std'])
        lec.radius = Normal(p['radius']['mean'], p['radius']['std'])
        return lec

    def normalize(self, ids, config, structure):
        n_ids = self.ids.normalize(ids)
        n_rot_z = self.z.normalize(config[:, CONFIG_ROTS[0], np.newaxis])
        n_rot_y = self.y.normalize(config[:, CONFIG_ROTS[1], np.newaxis])
        n_rot_x = self.x.normalize(config[:, CONFIG_ROTS[2], np.newaxis])
        n_center = self.center.normalize(config[:, CONFIG_CENTER])
        n_radius = self.radius.normalize(config[:, CONFIG_RADIUS, np.newaxis])
        n_config = np.concatenate(
            [n_rot_z, n_rot_y, n_rot_x, n_center, n_radius], axis=1)

        n_inner_ring_radius = self.inner_ring_radius.normalize(
            structure[:, GRID_STRUCTURE_POS['inner_ring_radius']])
        n_middle_ring_radius = self.middle_ring_radius.normalize(
            structure[:, GRID_STRUCTURE_POS['middle_ring_radius']])
        n_outer_ring_radius = self.outer_ring_radius.normalize(
            structure[:, GRID_STRUCTURE_POS['outer_ring_radius']])
        n_bulge_factor = self.bulge_factor.normalize(
            structure[:, GRID_STRUCTURE_POS['bulge_factor']])
        n_focal_length = self.focal_length.normalize(
            structure[:, GRID_STRUCTURE_POS['focal_length']])
        n_structure = np.stack([
            n_inner_ring_radius, n_middle_ring_radius, n_outer_ring_radius,
            n_bulge_factor, n_focal_length], axis=-1)
        return n_ids, n_config, n_structure

    def grid_params(self, batch_size):
        col_shape = (batch_size, 1)
        rot_z = self.z.sample(col_shape)
        rot_y = self.y.sample(col_shape)
        rot_x = self.x.sample(col_shape)
        center = self.center.sample((batch_size, 2))
        radius = self.radius.sample(col_shape)
        config = np.concatenate([rot_z, rot_y, rot_x, center, radius], axis=1)
        ids = self.ids.sample((batch_size, NUM_MIDDLE_CELLS))

        inner_ring_radius = self.inner_ring_radius.sample(col_shape)
        middle_ring_radius = self.middle_ring_radius.sample(col_shape)
        outer_ring_radius = self.outer_ring_radius.sample(col_shape)
        bulge_factor = self.bulge_factor.sample(col_shape)
        focal_length = self.focal_length.sample(col_shape)

        structure = np.concatenate(
            [inner_ring_radius, middle_ring_radius, outer_ring_radius,
             bulge_factor, focal_length], axis=1)

        return ids, config, structure

    def has_passed(self, mse):
        return mse <= self.pass_limit

    def __add__(self, other):
        params = ['z', 'x', 'y', 'ids', 'center', 'radius']
        new_lecture = Lecture()
        for param in params:
            self_param = getattr(self, param)
            other_param = getattr(other, param)
            if type(self_param) == Default and type(other_param) != Default:
                setattr(new_lecture, param, other_param)
            elif type(self_param) != Default and type(other_param) == Default:
                setattr(new_lecture, param, self_param)
            elif type(self_param) != Default and type(other_param) != Default:
                raise ValueError("Cannot combine two lectures with non Default"
                                 " Distribution on param {}".format(param))
        if self.name and other.name:
            new_lecture.name = self.name + " - " + other.name
        else:
            new_lecture.name = self.name + other.name
        return new_lecture

    def __eq__(self, other):
        params = ['z', 'x', 'y', 'ids', 'center', 'radius', 'pass_limit']
        for param in params:
            if getattr(self, param) != getattr(other, param):
                return False
        return True


def z_rot_lecture(hardness, lecture=None):
    if lecture is None:
        lecture = Lecture()
    lecture.name = 'z: {}'.format(hardness)
    lecture.z = Uniform(-hardness*pi, hardness*pi)
    return lecture


def z_rot_lecture_discrete(nb_points, sigma=1/360, lecture=None):
    if lecture is None:
        lecture = Lecture()
    lecture.name = 'z: {}'.format(nb_points)
    nb_points = max(1, nb_points)
    lecture.z = DiscretePoints(-pi, pi, nb_points, sigma)
    return lecture


def y_rot_lecture(hardness, lecture=None):
    if lecture is None:
        lecture = Lecture()
    lecture.name = 'y: {}'.format(hardness)
    lecture.y = Normal(0, hardness*to_radians(12))
    return lecture


def x_rot_lecture(hardness, lecture=None):
    if lecture is None:
        lecture = Lecture()
    lecture.name = 'x: {}'.format(hardness)
    lecture.x = Normal(0, hardness*to_radians(10))
    return lecture


class ReduceId(Bernoulli):
    def __init__(self, nb_ids):
        def all_ids():
            if NUM_MIDDLE_CELLS >= 16:
                a = np.arange(2**NUM_MIDDLE_CELLS, dtype=np.uint32)
            else:
                a = np.arange(2**NUM_MIDDLE_CELLS, dtype=np.uint16)

            a = a[:, np.newaxis]
            a = a.byteswap()
            return np.unpackbits(a.view(np.uint8), axis=1)[:, -12:]

        self.nb_ids = nb_ids
        idx = np.arange(2**12)
        np.random.shuffle(idx)
        self.ids = all_ids()[idx[:self.nb_ids]]

    def sample(self, shape):
        assert shape[1] == NUM_MIDDLE_CELLS
        choosen_idx = np.random.choice(self.nb_ids, shape[0])
        return self.ids[choosen_idx]


def reduced_id_lecture(hardness, lecture=None):
    if lecture is None:
        lecture = Lecture()
    nb_ids = math.ceil(hardness*2**NUM_MIDDLE_CELLS)
    lecture.name = 'ids: {}'.format(hardness)
    lecture.ids = ReduceId(nb_ids)
    return lecture


def exam():
    return Lecture.from_params(DISTRIBUTION_PARAMS, name='exam')


class CurriculumCallback(callbacks.Callback):
    def __init__(self, curriculum):
        super().__init__()
        assert len(curriculum) >= 1
        self.curriculum = curriculum
        self.lecture_id = 0
        self.losses = []

    def current_lecture(self):
        return self.curriculum[self.lecture_id]

    def on_train_begin(self, log={}):
        self.lecture_id = 0
        print("Begin with lecture #0 {}"
              .format(self.curriculum[0].name))

    def on_batch_end(self, batch, log={}):
        self.losses.append(log['loss'])

    def on_epoch_begin(self, epoch, log={}):
        self.losses = []

    def on_epoch_end(self, epoch, log={}):
        loss = np.array(self.losses).mean()
        lecture_id = self.lecture_id
        next_lecture_id = lecture_id + 1
        lecture = self.curriculum[lecture_id]
        if lecture.has_passed(loss):
            if next_lecture_id >= len(self.curriculum):
                self.model.stop_training = True
            else:
                print("Begin with lecture #{} {}"
                      .format(next_lecture_id,
                              self.curriculum[next_lecture_id].name))
                self.lecture_id += 1


def grids_from_lecture(lecture, batch_size=128, artist=None, scale=1.):
    if artist is None:
        artist = MaskGridArtist()
    ids, configs, structure = lecture.grid_params(batch_size)
    grids = draw_grids(ids.astype(np.float32), configs.astype(np.float32),
                       structure.astype(np.float32),
                       scales=[scale], artist=artist)
    assert len(grids) == 1
    return ids, configs, structure, grids[0]


def grid_generator(curriculum_cb, batch_size=128, artist=None,
                   scale=1.):
    if artist is None:
        artist = MaskGridArtist()
    while True:
        lecture = curriculum_cb.current_lecture()
        ids, configs, structure, grids = grids_from_lecture(lecture, batch_size, artist, scale)
        yield np.concatenate([ids, configs, structure], axis=1), grids


def get_generator_and_callback(curriculum, batch_size=128, artist=None, scale=1.):
    cb = CurriculumCallback(curriculum)
    gen = grid_generator(cb, batch_size, artist, scale=scale)
    return gen, cb
