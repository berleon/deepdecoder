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

from beras.util import upsample, resize_interpolate, \
    smooth
from more_itertools import pairwise
from keras.layers.core import Layer
from deepdecoder.utils import binary_mask, adaptive_mask
import keras.backend as K
from theano.ifelse import ifelse


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


def pyramid_gaussian(image, max_layer, sigma=2/3):
    yield image
    layer = 1
    prev_image = image
    while layer != max_layer:
        layer += 1
        layer_image = pyramid_reduce(prev_image, sigma)
        yield layer_image
        prev_image = layer_image


def pyramid_laplace(gauss_pyr):
    return [high - upsample(low)
            for high, low in pairwise(gauss_pyr)]


def blend_pyramid(a, b, mask, num_layers=None, weights=None):
    """TODO: Docstring for function.

    :weights: dictionary with {scale: weight}
    :returns: TODO

    """
    if weights is None:
        weights = [1] * num_layers
    num_layers = len(weights)
    gauss_pyr_a = list(pyramid_gaussian(a, num_layers))
    gauss_pyr_b = list(pyramid_gaussian(b, num_layers))
    gauss_pyr_mask = list(pyramid_gaussian(mask, num_layers))
    lap_pyr_a = pyramid_laplace(gauss_pyr_a) + gauss_pyr_a[-1:]
    lap_pyr_b = pyramid_laplace(gauss_pyr_b) + gauss_pyr_b[-1:]

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


class PyramidBlendingGridIdx(Layer):
    def __init__(self, tag_mean, **kwargs):
        self.tag_mean = tag_mean
        self.min_black_white_distance = 0.05
        super().__init__(**kwargs)

    @property
    def output_shape(self):
        shp = self.input_shape
        return (shp[0], 1) + shp[2:]

    def get_output(self, train=False):
        tag_mean = self.tag_mean.get_output(train)
        black_mean = tag_mean[:, 0]
        white_mean = K.abs(tag_mean[:, 1]) + black_mean + \
            self.min_black_white_distance

        nb_pyramid_layers = 3
        input = self.get_input(train)
        image = input[:, :1]
        grid_idx = input[:, 1:]
        selection_mask = binary_mask(grid_idx, ignore=0, black=0.8, white=0.8)

        pattern = (0, 'x', 'x', 'x')
        tag = adaptive_mask(grid_idx, ignore=0,
                            black=black_mean.dimshuffle(*pattern),
                            white=white_mean.dimshuffle(*pattern))

        gauss_pyr_tag = list(pyramid_gaussian(tag, nb_pyramid_layers))
        gauss_pyr_image = list(pyramid_gaussian(image, nb_pyramid_layers))
        gauss_pyr_mask = list(pyramid_gaussian(selection_mask,
                                               nb_pyramid_layers))
        pyr_masks = [0]*(len(gauss_pyr_mask) - 1) + gauss_pyr_mask[-1:]

        lap_pyr_tag = pyramid_laplace(gauss_pyr_tag) + gauss_pyr_tag[-1:]
        lap_pyr_image = pyramid_laplace(gauss_pyr_image) + gauss_pyr_image[-1:]

        blend_pyr = []
        for mask, lap_tag, lap_image in zip(pyr_masks, lap_pyr_tag,
                                            lap_pyr_image):
            blend = lap_tag*mask + lap_image*(1 - mask)
            blend_pyr.append(blend)

        img = None
        for low, high in pairwise(reversed(blend_pyr)):
            if img is None:
                img = low
            img = upsample(img) + high
        return img


class PyramidReduce(Layer):
    @property
    def output_shape(self):
        shp = self.input_shape
        return shp[:2] + (shp[2] // 2, shp[3] // 2)

    def get_output(self, train=False):
        x = self.get_input(train)
        # map [0; 1] to [-1, 1]
        return 2*pyramid_reduce(x) - 1


class PyramidBlending(Layer):
    def __init__(self, mask_layer, a_pyramid_layers=3, b_pyramid_layers=3,
                 use_blending=None,
                 **kwargs):
        self.a_pyramid_layers = a_pyramid_layers
        self.b_pyramid_layers = b_pyramid_layers
        self.mask_layer = mask_layer
        self.max_pyramid_layers = max(a_pyramid_layers, b_pyramid_layers)
        if use_blending is None:
            use_blending = K.variable(1)
        self.use_blending = use_blending
        self.weights = [K.variable(1)
                        for i in range(self.max_pyramid_layers)]
        super().__init__(**kwargs)

    def get_output(self, train=False):
        def fill_none(lst):
            return [None] * (self.max_pyramid_layers - len(lst)) + lst

        a = self.get_input(train)
        b = self.mask_layer.get_output(train)
        mask = K.cast(b < -0.1, 'float32')

        gauss_pyr_a = list(pyramid_gaussian(a, self.a_pyramid_layers))
        gauss_pyr_b = list(pyramid_gaussian(b, self.b_pyramid_layers))
        gauss_pyr_mask = list(pyramid_gaussian(mask, self.b_pyramid_layers))

        pyr_masks = [1]*(self.max_pyramid_layers - 1) + gauss_pyr_mask[-1:]

        lap_pyr_a = fill_none(pyramid_laplace(gauss_pyr_a) + gauss_pyr_a[-1:])
        lap_pyr_b = fill_none(pyramid_laplace(gauss_pyr_b) + gauss_pyr_b[-1:])

        blend_pyr = []
        for mask, lap_a, lap_b in zip(pyr_masks, lap_pyr_a, lap_pyr_b):
            if lap_a is None:
                lap_a = K.zeros_like(lap_b)
            elif lap_b is None:
                lap_b = K.zeros_like(lap_a)

            blend = lap_a*mask + lap_b*(1 - mask)
            blend_pyr.append(blend)

        img = None
        for i, (low, high) in enumerate(pairwise(reversed(blend_pyr))):
            if img is None:
                img = low*self.weights[i]
            img = upsample(img) + self.weights[i]*high
        return ifelse(self.use_blending, img, a)
