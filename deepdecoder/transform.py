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
