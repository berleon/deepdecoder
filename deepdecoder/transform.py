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

from beras.transform import upsample, resize_interpolate
from beras.filters import gaussian_filter_2d, gaussian_kernel_default_radius
from more_itertools import pairwise
from keras.layers.core import Layer
from deepdecoder.utils import binary_mask, adaptive_mask
import keras.backend as K


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
    return resize_interpolate(gaussian_filter_2d(image, sigma), scale=2)


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
        raise Exception("Not updated to the new keras synatx")
        self.tag_mean = tag_mean
        self.min_black_white_distance = 0.05
        super().__init__(**kwargs)

    @property
    def output_shape(self):
        shp = self.input_shape
        return (shp[0], 1) + shp[2:]

    def get_output(self, train=False):
        tag_mean = get_output(self.tag_mean, train,
                              self.layer_cache)
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
    def get_output_shape_for(self, input_shape):
        shp = input_shape
        return shp[:2] + (shp[2] // 2, shp[3] // 2)

    def call(self, x, mask=None):
        return pyramid_reduce(x)


class GaussianBlur(Layer):
    def __init__(self, sigma, window_radius=None, **kwargs):
        self.sigma = sigma
        self.window_radius = gaussian_kernel_default_radius(
            sigma, window_radius)
        super().__init__(**kwargs)

    def call(self, x, mask=None):
        return gaussian_filter_2d(x, self.sigma,
                                  window_radius=self.window_radius)


class UpsampleInterpolate(Layer):
    def __init__(self, scale, **kwargs):
        self.scale = scale
        super().__init__(**kwargs)

    def get_output_shape_for(self, input_shape):
        bs, c, h, w = input_shape
        return bs, c, int(h*self.scale), int(w*self.scale)

    def call(self, x, mask=None):
        return resize_interpolate(x, scale=1/self.scale)


class Selection(Layer):
    def __init__(self, threshold,
                 smooth_threshold,
                 sigma=1,
                 **kwargs):
        self.sigma = sigma
        self.threshold = K.variable(threshold)
        self.smooth_threshold = K.variable(smooth_threshold)
        super().__init__(**kwargs)

    def call(self, x, mask=None):
        selection = K.cast(x < self.threshold, 'float32')
        selection = gaussian_filter_2d(selection, sigma=self.sigma)
        selection = K.cast(selection > self.smooth_threshold,
                           'float32')
        selection = gaussian_filter_2d(selection, sigma=2/3)
        return selection


class AddLighting(Layer):
    def __init__(self, scale_factor=1,
                 shift_factor=1, **kwargs):
        self.scale_factor = K.variable(scale_factor)
        self.shift_factor = K.variable(shift_factor)
        super().__init__(**kwargs)

    def get_output_shape_for(self, input_shape):
        x_shape, scale_shape, shift_shape = input_shape
        assert scale_shape == x_shape, \
            "Scale shape {} does not match input shape {}" \
            .format(scale_shape, x_shape)
        assert shift_shape == x_shape, \
            "Shift shape {} does not match input shape {}" \
            .format(shift_shape, x_shape)
        return x_shape

    def call(self, inputs, mask=None):
        def norm_scale(a):
            return (a + 1) / 2
        x, scale, shift = inputs
        scale = norm_scale(scale)
        return x*(1 - scale*self.scale_factor) + self.shift_factor*shift


class PyramidBlending(Layer):
    def __init__(self, offset_pyramid_layers=3,
                 mask_pyramid_layers=3,
                 variable_offset_weights=None,
                 variable_mask_weights=None,
                 **kwargs):

        self.offset_pyramid_layers = offset_pyramid_layers
        self.mask_pyramid_layers = mask_pyramid_layers
        self.max_pyramid_layers = max(offset_pyramid_layers,
                                      mask_pyramid_layers)

        if variable_offset_weights is None:
            variable_offset_weights = [None] * self.offset_pyramid_layers
        if variable_mask_weights is None:
            variable_mask_weights = [None] * self.mask_pyramid_layers

        def collect_weights(variable_weights, defaults):
            weights = []
            for i, var_weight in enumerate(variable_weights):
                if var_weight is None:
                    weights.append(K.variable(defaults[i]))
                else:
                    weights.append('variable')
            return weights

        self.mask_weights = collect_weights(
            variable_mask_weights, [1, 1, 0])
        self.offset_weights = collect_weights(
            variable_offset_weights, [0, 0, 1])
        super().__init__(**kwargs)

    def get_output_shape_for(self, input_shapes):
        offset_shape, mask_shape, selection_shape = input_shapes[:3]
        assert mask_shape == selection_shape
        return offset_shape

    def call(self, inputs, mask=None):
        def collect_variable_weights_inputs(weights, start_idx):
            i = start_idx
            collect_weights = []
            for weight in weights:
                if weight == 'variable':
                    collect_weights.append(
                        inputs[i].dimshuffle(0, 1, 'x', 'x'))
                    i += 1
                else:
                    collect_weights.append(weight)
            return collect_weights

        def fill(lst, value=None):
            return [value] * (self.max_pyramid_layers - len(lst)) + lst

        nb_fix_inputs = 3
        offset, mask, selection = inputs[:nb_fix_inputs]
        mask = 2*mask - 1

        idx = nb_fix_inputs
        nb_variable_offsets = len([w for w in self.offset_weights
                                   if w is 'variable'])
        offset_weights = collect_variable_weights_inputs(
            self.offset_weights, idx)
        mask_weights = collect_variable_weights_inputs(
            self.mask_weights, idx + nb_variable_offsets)

        gauss_pyr_in = list(pyramid_gaussian(
            offset, self.offset_pyramid_layers))
        gauss_pyr_mask = list(pyramid_gaussian(mask, self.mask_pyramid_layers))
        gauss_pyr_select = list(pyramid_gaussian(
            selection, self.mask_pyramid_layers))

        pyr_select = fill(gauss_pyr_select, value=1)

        lap_pyr_in = fill(pyramid_laplace(gauss_pyr_in) + gauss_pyr_in[-1:])
        lap_pyr_mask = fill(pyramid_laplace(gauss_pyr_mask) +
                            gauss_pyr_mask[-1:])

        blend_pyr = []
        for selection, lap_in, lap_mask, offset_weight, mask_weight in \
                zip(pyr_select, lap_pyr_in, lap_pyr_mask,
                    offset_weights, mask_weights):
            if lap_in is None:
                lap_in = K.zeros_like(lap_mask)
            elif lap_mask is None:
                lap_mask = K.zeros_like(lap_in)
            blend = lap_in*selection*offset_weight + \
                lap_mask*(1 - selection)*mask_weight
            blend_pyr.append(blend)

        img = None
        for i, (low, high) in enumerate(pairwise(reversed(blend_pyr))):
            if img is None:
                img = low
            img = upsample(img) + high
        return img
