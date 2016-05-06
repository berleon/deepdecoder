# /usr/bin/env python
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

from conftest import on_gpu
import theano
from keras.models import Sequential
from keras.layers.convolutional import Convolution2D

from keras.layers.normalization import BatchNormalization
from keras.layers.core import Dense, Flatten, Reshape, Activation, \
    Layer

from beras.layers.core import Split, ZeroGradient, LinearInBounds

from deepdecoder.deconv import Deconvolution2D, Deconv2DVariableWeights
import numpy as np
import theano.tensor as T
import pytest


def deconv(X, w, subsample=(1, 1), border_mode=(0, 0), conv_mode='conv'):
    """
    sets up dummy convolutional forward pass and uses its grad as deconv
    currently only tested/working with same padding
    """
    from theano.sandbox.cuda.basic_ops import gpu_contiguous, gpu_alloc_empty
    from theano.sandbox.cuda.dnn import GpuDnnConvDesc, GpuDnnConvGradI
    img = gpu_contiguous(X)
    kerns = gpu_contiguous(w)
    empty = gpu_alloc_empty(img.shape[0], kerns.shape[1],
                            img.shape[2]*subsample[0],
                            img.shape[3]*subsample[1]).shape
    desc = GpuDnnConvDesc(border_mode=border_mode, subsample=subsample,
                          conv_mode=conv_mode)(empty, kerns.shape)
    out = gpu_alloc_empty(img.shape[0], kerns.shape[1],
                          img.shape[2]*subsample[0],
                          img.shape[3]*subsample[1])
    d_img = GpuDnnConvGradI()(kerns, img, out, desc)
    return d_img


@pytest.mark.skipif(not on_gpu(), reason="only works with cuda")
def test_deconv_dcgan():
    X = T.tensor4()
    w = theano.shared(np.random.sample((1, 10, 3, 3)).astype(np.float32))
    out = deconv(X, w, subsample=(2, 2))
    fn = theano.function([X], out)
    img = np.random.sample((64, 1, 16, 16)).astype(np.float32)
    deconv_out = fn(img)
    assert deconv_out.shape == (64, 10, 32, 32)


@pytest.mark.skipif(not on_gpu(), reason="only works with cuda")
def test_deconvolution2d():
    x = Input(shape=(1, 16, 16))
    out = Deconvolution2D(10, 3, 3, subsample=(2, 2))(x)
    fn = theano.function([x], out)
    img = np.random.sample((64, 1, 16, 16)).astype(np.float32)
    deconv_out = fn(img)
    assert deconv_out.shape == (64, 10, 32, 32)


@pytest.mark.skipif(not on_gpu(), reason="only works with cuda")
def test_deconvolution2d_with_conv2d_gpu_contiguous():
    input_shape = (64, 1, 8, 8)
    model = Sequential()
    model.add(Layer(batch_input_shape=input_shape))
    model.add(Deconvolution2D(8, 3, 3, subsample=(1, 1),
                              border_mode=(1, 1)))
    model.add(BatchNormalization(axis=1))
    model.add(Activation('relu'))
    model.add(Convolution2D(10, 3, 3, border_mode='same'))
    model.add(BatchNormalization(axis=1))
    model.add(Activation('relu'))
    model.add(Deconvolution2D(1, 3, 3, subsample=(2, 2),
                              border_mode=(1, 1)))
    model.compile('sgd', 'mse')

    img = np.random.sample((64, 1, 16, 16)).astype(np.float32)
    model.predict(img)


@pytest.mark.skipif(not on_gpu(), reason="only works with cuda")
def test_deconvolution2d_variable():
    input_dim = 20
    z_shape = (64, input_dim)
    model = Sequential()
    z = Layer(input_shape=(input_dim,))
    model = Sequential()
    model.add(z)
    model.add(Dense(8*4*4))
    model.add(Reshape((8, 4, 4,)))
    model.add(Activation('relu'))
    model.add(Deconv2DVariableWeights(z, 8, 3, 3, subsample=(1, 1),
                                      border_mode=(1, 1)))
    model.add(BatchNormalization(axis=1))
    model.add(Activation('relu'))
    model.compile('sgd', 'mse')

    img = np.random.uniform(-1, 1, z_shape).astype(np.float32)
    model.predict(img)
