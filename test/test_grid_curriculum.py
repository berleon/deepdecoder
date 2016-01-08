#! /usr/bin/env python
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
import multiprocessing
from unittest.mock import MagicMock

import pytest
from beesgrid.pybeesgrid import NUM_MIDDLE_CELLS, NUM_CONFIGS

from deepdecoder import grid_curriculum
from beesgrid import CONFIG_ROTS, CONFIG_RADIUS
import numpy as np
from deepdecoder.grid_curriculum import Lecture, z_rot_lecture, \
    y_rot_lecture, reduced_id_lecture, ReduceId, \
    CurriculumCallback, grid_generator, grids_from_lecture


def test_lecture_add():
    def print_lecture(l):
        params = ['z', 'x', 'y', 'ids', 'center', 'radius']
        for p in params:
            print("{}: {}".format(p, getattr(l, p)))
    h = 0.1
    assert Lecture() + z_rot_lecture(h) == z_rot_lecture(h)
    assert (Lecture() + z_rot_lecture(h)).name == z_rot_lecture(h).name
    zy_lec = z_rot_lecture(h) + y_rot_lecture(h)
    assert zy_lec.z == z_rot_lecture(h).z
    assert zy_lec.y == y_rot_lecture(h).y
    assert zy_lec.name ==  z_rot_lecture(h).name + ' - ' + y_rot_lecture(h).name

def test_lecture_sample():
    n = 128
    ids, config = Lecture().grid_params(n)
    assert ids.shape == (n, NUM_MIDDLE_CELLS)
    assert config.shape == (n, NUM_CONFIGS)


def test_reduced_id_is_unique():
    def unique(x):
        dtype = np.dtype((np.void, x.dtype.itemsize * x.shape[1]))
        b = np.ascontiguousarray(x).view(dtype)
        _, idx = np.unique(b, return_index=True)
        return x[idx]

    n = 2**14
    for nb_ids in [2, 64, 256, 2**NUM_MIDDLE_CELLS]:
        ids = ReduceId(nb_ids).sample((n, NUM_MIDDLE_CELLS))
        assert type(ids) == np.ndarray
        unique_ids = np.unique(ids)
        assert len(unique_ids) <= nb_ids


def test_curriculum_config():
    batch_size = 2**16
    exam = grid_curriculum.exam()
    ids, configs = exam.grid_params(batch_size)
    assert type(ids) == np.ndarray
    assert type(configs) == np.ndarray
    assert configs[:, CONFIG_ROTS].mean() <= 0.05
    assert abs(configs[:, CONFIG_RADIUS].mean() - exam.radius.mean) <= 0.05
    assert (configs[:, CONFIG_ROTS[1]].std() - exam.y.std) <= 0.05
    assert (configs[:, CONFIG_ROTS[2]].std() - exam.x.std) <= 0.05
    assert abs(configs[:, CONFIG_RADIUS].mean() - exam.radius.mean) <= 0.05


def test_grids_from_lecture():
    lec = reduced_id_lecture(0.1)
    grids_from_lecture(lec)


def test_curriculum_generator_callback():
    curriculum = [
        reduced_id_lecture(0.1),
        reduced_id_lecture(0.1) + z_rot_lecture(0.1),
        reduced_id_lecture(0.3),
    ]
    lecture_id = multiprocessing.Value('i', 0)
    cb = CurriculumCallback(curriculum, lecture_id)
    cb.model = MagicMock()
    cb.model.stop_training = False
    gen = grid_generator(curriculum, lecture_id)
    next(gen)
    assert lecture_id.value == 0
    cb.on_train_begin()
    cb.on_batch_end(0, {'loss': 1000})
    cb.on_epoch_end(0)
    # loss to high, do not proceed into next lecture
    assert lecture_id.value == 0

    cb.on_epoch_begin(1)
    cb.on_batch_end(0, {'loss': 0})
    cb.on_batch_end(1, {'loss': 0})
    cb.on_epoch_end(1)
    # passed the lecture
    assert lecture_id.value == 1

    cb.on_epoch_begin(2)
    cb.on_batch_end(0, {'loss': 0})
    cb.on_epoch_end(2)
    assert lecture_id.value == 2

    cb.on_epoch_begin(3)
    cb.on_batch_end(0, {'loss': 0})
    cb.on_epoch_end(3, {'loss': 0})
    assert lecture_id.value == 2
    assert cb.model.stop_training


if __name__ == "__main__":
    pytest.main(__file__)
