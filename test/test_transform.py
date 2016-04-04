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
from deepdecoder.transform import blend_pyramid, pyramid_reduce, \
    pyramid_gaussian

import keras.backend as K
import numpy as np
import matplotlib.pyplot as plt
import pytest
from scipy.misc import imread
from timeit import Timer
import theano


@pytest.fixture
def blend_images():
    def load(fname):
        return imread(fname, flatten=True)[np.newaxis, np.newaxis] / 255.
    orange = load('test/data/orange.jpg')
    apple = load('test/data/apple.jpg')
    mask = load('test/data/mask256.jpg')
    return orange, apple, mask


def test_util_pyramid_reduce(astronaut):
    reduced = pyramid_reduce(theano.shared(astronaut))
    assert reduced.eval().shape[-1] == astronaut.shape[-1] // 2


def test_util_pyramid_gaussian(astronaut):
    max_layer = np.log2(astronaut.shape[-1])
    astronauts = theano.shared(np.repeat(astronaut, 64, axis=0))
    pyr_gaus = list(pyramid_gaussian(astronauts, max_layer))
    assert len(pyr_gaus) == max_layer
    gauss_fn = theano.function([], pyr_gaus)
    for i, gaus in enumerate(gauss_fn()):
        plt.imshow(gaus[3, 0], cmap='gray')
        plt_save_and_maybe_show("utils/gaussian/gauss_{}.png".format(i))

    assert pyr_gaus[-1].shape.eval()[-1] == \
        astronaut.shape[-1] // 2**(max_layer-1)


@pytest.fixture
def blend_pyramid_fn():
    orange = K.placeholder(name='orange', ndim=4)
    apple = K.placeholder(name='apple', ndim=4)
    mask = K.placeholder(name='mask', ndim=4)
    blended = blend_pyramid(orange, apple, mask, num_layers=4)
    return K.function([orange, apple, mask], [blended])


@pytest.mark.slow
def test_util_benchmark_blending_pyramid(blend_images, blend_pyramid_fn):
    bs = 128
    orange, apple, mask = [b.repeat(bs, axis=0) for b in blend_images]

    def blend():
        blend_pyramid_fn([orange, apple, mask])

    n = 10
    t = Timer(blend)
    print("need {:.5f}s to blend. batch size: {}".format(t.timeit(n) / n, bs))


@pytest.mark.slow
def test_util_blending_pyramid(blend_images, blend_pyramid_fn):
    orange, apple, mask = blend_images
    print(orange.shape)
    print(mask.shape)
    img = blend_pyramid_fn([orange, apple, mask])[0]
    print(img.shape)
    w = orange.shape[-1]
    hard_add = np.concatenate([apple[:, :, :, :w/2],
                               orange[:, :, :, w/2:]], axis=3)
    plt.subplot(221)
    plt.imshow(orange[0, 0], cmap='gray')
    plt.subplot(222)
    plt.imshow(apple[0, 0], cmap='gray')
    plt.subplot(223)
    plt.imshow(hard_add[0, 0], cmap='gray')
    plt.subplot(224)
    plt.imshow(img[0, 0], cmap='gray')
    plt_save_and_maybe_show("utils/blending.png")
