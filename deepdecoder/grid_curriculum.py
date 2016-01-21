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

from beesgrid import draw_grids, NUM_MIDDLE_CELLS, MaskGridArtist, TAG_SIZE
from keras.backend import epsilon
from keras import callbacks
from dotmap import DotMap


def to_radians(x):
    return x / 180. * np.pi

DISTRIBUTION_PARAMS = DotMap({
    'z': {'low': -pi, 'high': pi},
    'y': {'mean': 0, 'std': to_radians(12)},
    'x': {'mean': 0, 'std': to_radians(10)},
    'center': {'mean': TAG_SIZE/2, 'std': 2},
    'radius': {'mean': 24.5, 'std': 1},
})


class Distribution:
    def sample(self, shape):
        raise NotImplementedError

    def __eq__(self, other):
        return type(self) == type(other)


class Default(Distribution):
    def __init__(self, distribution):
        self.distribution = distribution

    def sample(self, shape):
        return self.distribution.sample(shape)

    def __eq__(self, other):
        return super().__eq__(other) and \
            self.distribution == other.distribution


class Zeros(Distribution):
    def sample(self, shape):
        return np.zeros(shape)


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

    def __neq__(self, other):
        return not self.__eq__(other)


class Uniform(Distribution):
    def __init__(self, low, high):
        self.low = low
        self.high = high

    def sample(self, shape):
        return np.random.uniform(self.low, self.high, shape)

    def __eq__(self, other):
        return super().__eq__(other) and \
            self.low == other.low and self.high == other.high


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


class Bernoulli(Distribution):
    def sample(self, shape):
        return np.random.binomial(1, 0.5, shape)


class Lecture:
    def __init__(self, pass_limit=0.01, name=''):
        self.z = Default(Zeros())
        self.x = Default(Zeros())
        self.y = Default(Zeros())
        self.ids = Default(Bernoulli())
        self.center = Default(Normal(25, 2))
        self.radius = Default(Normal(24.5, 1))
        self.pass_limit = pass_limit
        self.name = name

    def grid_params(self, batch_size):
        col_shape = (batch_size, 1)
        rot_z = self.z.sample(col_shape)
        rot_y = self.y.sample(col_shape)
        rot_x = self.x.sample(col_shape)
        center = self.center.sample((batch_size, 2))
        radius = self.radius.sample(col_shape)
        config = np.concatenate([rot_z, rot_x, rot_y, center, radius], axis=1)
        ids = self.ids.sample((batch_size, NUM_MIDDLE_CELLS))
        return ids, config

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


class ReduceId(Distribution):
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
    d = DISTRIBUTION_PARAMS
    lec = Lecture()
    lec.name = 'exam'
    lec.z = Uniform(d.z.low, d.z.high)
    lec.y = Normal(d.y.mean, d.y.std)
    lec.x = Normal(d.x.mean, d.x.std)
    lec.center = Normal(d.center.mean, d.center.std)
    lec.radius = Normal(d.radius.mean, d.radius.std)
    return lec


class CurriculumCallback(callbacks.Callback):
    def __init__(self, curriculum):
        super().__init__()
        assert len(curriculum) >= 1
        self.curriculum = curriculum
        self.lecture_id = 0
        self.losses = []

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
    ids, configs = lecture.grid_params(batch_size)
    grids = draw_grids(ids.astype(np.float32), configs.astype(np.float32),
                       scales=[scale], artist=artist)
    assert len(grids) == 1
    return np.concatenate([ids, configs], axis=1), grids[0]


def grid_generator(curriculum, lecutre_id, batch_size=128, artist=None,
                   scale=1.):
    if artist is None:
        artist = MaskGridArtist()
    while True:
        lecture = curriculum[lecutre_id.value]
        yield grids_from_lecture(lecture, batch_size, artist, scale)


def get_generator_and_callback(curriculum, batch_size=128, scale=1.):
    lecture_id = multiprocessing.Value('i', 0)
    cb = CurriculumCallback(curriculum, lecture_id)
    gen = grid_generator(curriculum, lecture_id, batch_size, scale=scale)
    return gen, cb
