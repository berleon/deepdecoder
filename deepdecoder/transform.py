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

from diktya.theano.image_transform import upsample, resize_interpolate
from diktya.theano.image_filters import gaussian_filter_2d, gaussian_kernel_default_radius, \
    gaussian_filter_2d_variable_sigma
from more_itertools import pairwise
from keras.layers.core import Layer
from deepdecoder.utils import binary_mask, adaptive_mask
import keras.backend as K
import theano


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


def pyramid_reduce(image, sigma=None, scale=0.5):
    if sigma is None:
        sigma = (1/scale)*2 / 6
    return resize_interpolate(gaussian_filter_2d(image, sigma), scale=scale)


def pyramid_gaussian(image, max_layer, sigma=2/3):
    yield image
    layer = 1
    prev_image = image
    while layer != max_layer:
        layer += 1
        layer_image = pyramid_reduce(prev_image, sigma=sigma)
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


class PyramidReduce(Layer):
    def __init__(self, scale=0.5, **kwargs):
        self.scale = scale
        super().__init__(**kwargs)

    def get_output_shape_for(self, input_shape):
        shp = input_shape
        return shp[:2] + (shp[2] * self.scale, shp[3] * self.scale)

    def call(self, x, mask=None):
        return pyramid_reduce(x, scale=self.scale)


class GaussianBlur(Layer):
    def __init__(self, sigma, window_radius=None, **kwargs):
        self.sigma = sigma
        self.window_radius = gaussian_kernel_default_radius(
            sigma, window_radius)
        super().__init__(**kwargs)

    def call(self, x, mask=None):
        return gaussian_filter_2d(x, self.sigma,
                                  window_radius=self.window_radius)

    def get_config(self):
        config = {
            'sigma': self.sigma,
            'window_radius': self.window_radius,
        }
        base_config = super(GaussianBlur, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class BlendingBlur(Layer):
    def __init__(self, sigma=2, window_radius=None, **kwargs):
        self.sigma = sigma
        self.window_radius = gaussian_kernel_default_radius(
            sigma, window_radius)
        super(BlendingBlur, self).__init__(**kwargs)

    def get_output_shape_for(self, input_shape):
        input_shape, _ = input_shape
        return input_shape

    def call(self, x, mask=None):
        image, factor = x
        down = gaussian_filter_2d(image, self.sigma, self.window_radius)
        high_freq = image - down
        return down + high_freq * factor.dimshuffle(0, 1, 'x', 'x')

    def get_config(self):
        config = {
            'sigma': self.sigma,
            'window_radius': self.window_radius,
        }
        base_config = super(BlendingBlur, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class UpsampleInterpolate(Layer):
    def __init__(self, scale, **kwargs):
        self.scale = scale
        super().__init__(**kwargs)

    def get_output_shape_for(self, input_shape):
        bs, c, h, w = input_shape
        return bs, c, int(h*self.scale), int(w*self.scale)

    def call(self, x, mask=None):
        return resize_interpolate(x, scale=self.scale)

    def get_config(self):
        config = {
            'scale': self.scale,
        }
        base_config = super(UpsampleInterpolate, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class Segmentation(Layer):
    def __init__(self, threshold,
                 smooth_threshold,
                 sigma=1,
                 **kwargs):
        self.sigma = sigma
        self.threshold = K.variable(threshold)
        self.smooth_threshold = K.variable(smooth_threshold)
        super().__init__(**kwargs)

    def call(self, x, mask=None):
        segmentation = K.cast(x < self.threshold, 'float32')
        segmentation = gaussian_filter_2d(segmentation, sigma=self.sigma)
        segmentation = K.cast(segmentation > self.smooth_threshold,
                           'float32')
        segmentation = gaussian_filter_2d(segmentation, sigma=2/3)
        return segmentation

    def get_config(self):
        config = {
            'sigma': self.sigma,
            'threshold': float(K.get_value(self.threshold)),
            'smooth_threshold': float(K.get_value(self.smooth_threshold)),
        }
        base_config = super(Segmentation, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class AddLighting(Layer):
    def __init__(self, scale_factor=1,
                 shift_factor=1, **kwargs):
        self.scale_factor = K.variable(scale_factor)
        self.shift_factor = K.variable(shift_factor)
        super().__init__(**kwargs)

    def get_output_shape_for(self, input_shape):
        x_shape, scale_black_shape, scale_white_shape, shift_shape = \
            input_shape
        assert x_shape == scale_black_shape == \
            scale_white_shape == shift_shape, \
            "Input shapes are not equal. Got input shapes: {}".format(
                ", ".join([str(s) for s in input_shape]))
        return x_shape

    def call(self, inputs, mask=None):
        def norm_scale(a):
            return (a + 1) / 2
        x, scale_black, scale_white, shift = inputs
        x = 2*(x - 0.5)
        black_selection = K.cast(x < 0, K.floatx())
        white_selection = 1 - black_selection

        scale_black = scale_black * self.scale_factor + (1 - self.scale_factor)
        scale_white = scale_white * self.scale_factor + (1 - self.scale_factor)
        scaled = x*black_selection*scale_black + x*white_selection*scale_white
        return scaled + self.shift_factor*shift

    def get_config(self):
        config = {
            'scale_factor': float(K.get_value(self.scale_factor)),
            'shift_factor': float(K.get_value(self.shift_factor)),
        }
        base_config = super(AddLighting, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class HighPass(Layer):
    def __init__(self, sigma, nb_steps, **kwargs):
        super().__init__(**kwargs)
        self.sigma = sigma
        self.nb_steps = K.variable(nb_steps, dtype='int32')

    def call(self, x, mask=None):
        high, _ = theano.scan(
            lambda x: x - gaussian_filter_2d(x, self.sigma),
            outputs_info=x, n_steps=self.nb_steps)
        return high[-1]

    def get_config(self):
        config = {
            'sigma': self.sigma,
            'nb_steps': float(K.get_value(self.nb_steps)),
        }
        base_config = super(HighPass, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class Background(Layer):
    def get_output_shape_for(self, input_shapes):
        offset_shape, mask_shape, selection_shape = input_shapes[:3]
        assert mask_shape == selection_shape
        return offset_shape

    def call(self, inputs, mask=None):
        background, forground, segmentation = inputs
        return background * segmentation + forground * (1 - segmentation)


class PyramidBlending(Layer):
    def __init__(self, offset_pyramid_layers=3,
                 mask_pyramid_layers=3,
                 offset_weights=None,
                 mask_weights=None,
                 use_selection=None,
                 **kwargs):

        self.offset_pyramid_layers = offset_pyramid_layers
        self.mask_pyramid_layers = mask_pyramid_layers
        self.max_pyramid_layers = max(offset_pyramid_layers,
                                      mask_pyramid_layers)

        if offset_weights is None:
            offset_weights = [None] * self.offset_pyramid_layers
        if mask_weights is None:
            mask_weights = [None] * self.mask_pyramid_layers
        if use_selection is None:
            use_selection = [True] * self.max_pyramid_layers

        self.use_selection = use_selection

        def collect_weights(weights, defaults):
            new_weights = []
            for i, weight in enumerate(weights):
                if weight is None:
                    new_weights.append(K.variable(defaults[i]))
                elif type(weight) in (float, int):
                    new_weights.append(K.variable(weight))
                else:
                    new_weights.append(weight)

            return new_weights

        self.mask_weights = collect_weights(mask_weights, [1, 1, 1])
        self.offset_weights = collect_weights(offset_weights, [1, 1, 1])
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
        offset, mask, segmentation = inputs[:nb_fix_inputs]

        idx = nb_fix_inputs
        nb_variable_offsets = len([w for w in self.offset_weights
                                   if w is 'variable'])
        offset_weights = collect_variable_weights_inputs(
            self.offset_weights, idx)
        mask_weights = collect_variable_weights_inputs(
            self.mask_weights, idx + nb_variable_offsets)
        offset_weights = fill(offset_weights, value=0)
        mask_weights = fill(mask_weights, value=0)
        gauss_pyr_in = list(pyramid_gaussian(
            offset, self.offset_pyramid_layers))
        gauss_pyr_mask = list(pyramid_gaussian(mask, self.mask_pyramid_layers))
        gauss_pyr_select = list(pyramid_gaussian(
            segmentation, self.mask_pyramid_layers))

        pyr_select = fill(gauss_pyr_select, value=1)

        lap_pyr_in = fill(pyramid_laplace(gauss_pyr_in) + gauss_pyr_in[-1:])
        lap_pyr_mask = fill(pyramid_laplace(gauss_pyr_mask) +
                            gauss_pyr_mask[-1:])

        blend_pyr = []
        pyramids = [pyr_select, self.use_selection, lap_pyr_in, lap_pyr_mask,
                    offset_weights, mask_weights]
        assert len({len(p) for p in pyramids}) == 1, \
            "Different pyramid heights"

        for segmentation, use_selection, lap_in, lap_mask, offset_weight, \
                mask_weight in zip(*pyramids):

            if lap_in is None:
                lap_in = K.zeros_like(lap_mask)
            elif lap_mask is None:
                lap_mask = K.zeros_like(lap_in)
            if use_selection:
                blend = lap_in*segmentation*offset_weight + \
                    lap_mask*(1 - segmentation)*mask_weight
            else:
                blend = lap_in*offset_weight + lap_mask*mask_weight
            blend_pyr.append(blend)

        img = blend_pyr[-1]
        for high in reversed(blend_pyr[:-1]):
            img = upsample(img) + high

        return img

    def get_config(self):
        def to_builtin(x):
            if hasattr(x, 'get_value'):
                return float(x.get_value())
            else:
                return x

        def to_buildin_list(xs):
            return list(map(to_builtin, xs))

        config = {
            'offset_pyramid_layers': self.offset_pyramid_layers,
            'mask_pyramid_layers': self.mask_pyramid_layers,
            'offset_weights': to_buildin_list(self.offset_weights),
            'mask_weights': to_buildin_list(self.mask_weights),
            'use_selection': self.use_selection,
        }
        base_config = super(PyramidBlending, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
