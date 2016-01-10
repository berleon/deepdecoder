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

from conftest import plt_save_and_maybe_show
from deepdecoder.utils import rotate_by_multiple_of_90
import keras.backend as K
import numpy as np
import matplotlib.pyplot as plt
import pytest


@pytest.fixture
def batch():
    batch = np.ones((4, 1, 8, 8), dtype=np.float32)
    batch[:, :, :4, :4] = 0
    batch[:, :, 6:, :] = 0
    return batch


def test_util_rotate_by_multiple_of_90(batch):
    n = len(batch)
    th_batch = K.variable(batch)
    rots = K.variable(np.array([0, 1, 2, 3]))
    rotated = rotate_by_multiple_of_90(th_batch, rots).eval()
    for i in range(n):
        plt.subplot(131)
        plt.imshow(batch[i, 0])
        plt.subplot(132)
        plt.imshow(rotated[i, 0])
        plt.subplot(133)
        plt.imshow(np.rot90(batch[i, 0], k=i))
        plt_save_and_maybe_show("utils/rotate_{}.png".format(i))

    assert rotated.shape == batch.shape
    for i in range(n):
        assert (rotated[i, 0] == np.rot90(batch[i, 0], k=i)).all(), i


def test_util_rotate_by_multiple_of_90_missing_rots(batch):
    th_batch = K.variable(batch)
    rots = K.variable(np.array([0, 0, 2, 2]))
    rotated = rotate_by_multiple_of_90(th_batch, rots).eval()
    assert rotated.shape == batch.shape
