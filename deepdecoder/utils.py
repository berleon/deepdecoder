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
import os

from beesgrid import MASK
import theano
import h5py
import numpy as np
import theano.tensor as T
from beras.util import tile, upsample, resize_interpolate, \
    smooth

floatX = theano.config.floatX


def np_binary_mask(mask, black=0., ignore=0.5,  white=1.):
    bw = ignore * np.ones_like(mask, dtype=np.float32)
    bw[mask > MASK["IGNORE"]] = white
    bw[mask < MASK["BACKGROUND_RING"]] = black
    return bw


def binary_mask(mask, black=0., ignore=0.5,  white=1.):
    bw = ignore * T.ones_like(mask, dtype=floatX)
    bw = T.set_subtensor(bw[(mask > MASK["IGNORE"]).nonzero()], white)
    bw = T.set_subtensor(bw[(mask < MASK["BACKGROUND_RING"]).nonzero()], black)
    return bw


def adaptive_mask(mask, black=0.5, ignore=0.5, white=1.):
    bw = ignore * T.ones_like(mask, dtype=floatX)
    t_black = black*T.ones_like(bw, dtype=floatX)
    t_white = white*T.ones_like(bw, dtype=floatX)
    white_idx = (mask > MASK["IGNORE"]).nonzero()
    black_idx = (mask < MASK["BACKGROUND_RING"]).nonzero()
    bw = T.set_subtensor(bw[white_idx], t_white[white_idx])
    bw = T.set_subtensor(bw[black_idx], t_black[black_idx])
    return bw


def tags_from_hdf5(fname):
    tags_list = []
    with open(fname) as pathfile:
        for hdf5_name in pathfile.readlines():
            hdf5_fname = os.path.dirname(fname) + "/" + hdf5_name.rstrip('\n')
            f = h5py.File(hdf5_fname)
            data = np.asarray(f["data"], dtype=np.float32)
            labels = np.asarray(f["labels"], dtype=np.float32)
            tags = data[labels == 1]
            tags /= 255.
            tags_list.append(tags)

    return np.concatenate(tags_list)


def loadRealData(fname):
    X = tags_from_hdf5(fname)
    X = X[:(len(X)//64)*64]
    return X


def visualise_tiles(images):
    import matplotlib.pyplot as plt
    tiled_fakes = tile(images)
    plt.imshow(tiled_fakes[0], cmap='gray')
    plt.show()


def rotate_by_multiple_of_90(img, rots):
    def idx(pos):
        return T.eq(rots, pos).nonzero()
    rots = rots.reshape((-1, ))
    return T.concatenate([
        img[idx(0)][:, :, :, :],
        img[idx(1)].swapaxes(2, 3)[:, :, ::-1, :],
        img[idx(2)][:, :, ::-1, ::-1],
        img[idx(3)].swapaxes(2, 3)[:, :, :, ::-1],
    ])


def rotate_by_multiple_of_90(img, rots):
    def idx(pos):
        return T.eq(rots, pos).nonzero()
    rots = rots.reshape((-1, ))
    img = T.set_subtensor(img[idx(0)], img[idx(0)][:, :, :, :])
    img = T.set_subtensor(img[idx(1)],
                          img[idx(1)].swapaxes(2, 3)[:, :, ::-1, :])
    img = T.set_subtensor(img[idx(2)], img[idx(2)][:, :, ::-1, ::-1])
    img = T.set_subtensor(img[idx(3)],
                          img[idx(3)].swapaxes(2, 3)[:, :, :, ::-1])
    return img


def zip_visualise_tiles(*arrs):
    import matplotlib.pyplot as plt
    assert len(arrs) >= 2
    length = len(arrs[0])
    for a in arrs:
        assert len(a) == length, "all input arrays must have the same size"
    tiles = []
    for i in range(length):
        for a in arrs:
            tiles.append(a[i])

    tiled = tile(tiles, columns_must_be_multiple_of=len(arrs))
    assert len(tiled) == 1, "currently only grayscale image are supported"
    plt.imshow(tiled[0], cmap='gray')
    plt.show()


def pyramid_expand(image, sigma=2/3):
    """
    Upsample and then smooth image.
    :image: array
    :upscale: float, optional
    :sigma:  Sigma for Gaussian filter. Default is 2 / 3 which
    corresponds to a filter mask twice the size of the scale factor that covers
    more than 99% of the Gaussian distribution.
    """
    return upsample(image, sigma)


def pyramid_reduce(image, sigma=2/3):
    return resize_interpolate(smooth(image, sigma), scale=2)


def pyramid_gaussian(image, max_layer):
    yield image
    layer = 1
    prev_image = image
    while layer != max_layer:
        layer += 1
        layer_image = pyramid_reduce(prev_image)
        yield layer_image
        prev_image = layer_image


def blend_pyramid(a, b, mask, num_layers=None, weights=None):
    """TODO: Docstring for function.

    :weights: dictionary with {scale: weight}
    :returns: TODO

    """
    def pairwise(iterable):
        prev = None
        for elem in iterable:
            if prev is not None:
                yield prev, elem
            prev = elem

    def laplace_pyramid(gauss_pyr):
        return [high - upsample(low)
                for high, low in pairwise(gauss_pyr)]
    if weights is None:
        weights = [1] * num_layers
    num_layers = len(weights)
    gauss_pyr_a = list(pyramid_gaussian(a, num_layers))
    gauss_pyr_b = list(pyramid_gaussian(b, num_layers))
    gauss_pyr_mask = list(pyramid_gaussian(mask, num_layers))
    lap_pyr_a = laplace_pyramid(gauss_pyr_a) + gauss_pyr_a[-1:]
    lap_pyr_b = laplace_pyramid(gauss_pyr_b) + gauss_pyr_b[-1:]

    blend_pyr = []
    for gauss_mask, lap_a, lap_b in zip(gauss_pyr_mask, lap_pyr_a, lap_pyr_b):
        blend = lap_a*gauss_mask + lap_b*(1-gauss_mask)
        blend_pyr.append(blend)

    img = None
    for weight, (low, high) in zip([0] + weights,
                                   pairwise(reversed(blend_pyr))):
        if img is None:
            img = low
        img = upsample(img) + weight*high
    return img
