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
import theano
from keras.models import Sequential
from theano.sandbox.cuda.basic_ops import gpu_contiguous, gpu_alloc_empty
from theano.sandbox.cuda.dnn import GpuDnnConvDesc, GpuDnnConvGradI

from deepdecoder.dcgan_train import Deconvolution2D
import numpy as np
import theano.tensor as T


def deconv(X, w, subsample=(1, 1), border_mode=(0, 0), conv_mode='conv'):
    """
    sets up dummy convolutional forward pass and uses its grad as deconv
    currently only tested/working with same padding
    """
    img = gpu_contiguous(X)
    kerns = gpu_contiguous(w)
    desc = GpuDnnConvDesc(border_mode=border_mode, subsample=subsample,
                          conv_mode=conv_mode)(gpu_alloc_empty(img.shape[0], kerns.shape[1], img.shape[2]*subsample[0], img.shape[3]*subsample[1]).shape, kerns.shape)
    out = gpu_alloc_empty(img.shape[0], kerns.shape[1], img.shape[2]*subsample[0], img.shape[3]*subsample[1])
    d_img = GpuDnnConvGradI()(kerns, img, out, desc)
    return d_img


def test_deconv_dcgan():
    X = T.tensor4()
    w = theano.shared(np.random.sample((1, 10, 3, 3)).astype(np.float32))
    out = deconv(X, w, subsample=(2, 2))
    fn = theano.function([X], out)
    img = np.random.sample((64, 1, 16, 16)).astype(np.float32)
    deconv_out = fn(img)
    assert deconv_out.shape == (64, 10, 32, 32)


def test_deconvolution2d():
    model = Sequential()
    model.add(Deconvolution2D(10, 3, 3, subsample=(2, 2), input_shape=(1, 16, 16)))
    out = model.get_output()
    fn = theano.function([model.input], out)
    img = np.random.sample((64, 1, 16, 16)).astype(np.float32)
    deconv_out = fn(img)
    assert deconv_out.shape == (64, 10, 32, 32)


