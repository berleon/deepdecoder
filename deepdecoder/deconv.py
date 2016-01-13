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

from keras import initializations
from theano.sandbox.cuda.dnn import GpuDnnConvDesc, GpuDnnConvGradI, \
    gpu_alloc_empty
from keras.layers.core import Layer


class Deconvolution2D(Layer):
    def __init__(self, nb_filter, nb_row, nb_col,
                 subsample=(1, 1), border_mode=(0, 0),
                 init='glorot_uniform', **kwargs):
        self.nb_filter = nb_filter
        self.nb_row = nb_row
        self.nb_col = nb_col
        self.init = initializations.get(init)
        self.border_mode = border_mode
        self.subsample = subsample
        super(Deconvolution2D, self).__init__(**kwargs)

    def build(self):
        stack_size = self.input_shape[1]
        self.W_shape = (stack_size, self.nb_filter, self.nb_row, self.nb_col)
        self.W = self.init(self.W_shape)
        self.params = [self.W]

    @property
    def output_shape(self):
        in_rows, in_cols = self.input_shape[2:]
        sr, sc = self.subsample
        return self.input_shape[0], self.nb_filter, sr*in_rows, sr*in_cols

    def get_output(self, train=False):
        """
        sets up dummy convolutional forward pass and uses its grad as deconv
        currently only tested/working with same padding
        """
        img = self.get_input(train)
        kerns = self.W
        sr, sc = self.subsample
        out = gpu_alloc_empty(img.shape[0], kerns.shape[1],
                              img.shape[2]*sr, img.shape[3]*sc)
        desc = GpuDnnConvDesc(
            border_mode=self.border_mode, subsample=self.subsample,
            conv_mode='conv')(out.shape, kerns.shape)
        d_img = GpuDnnConvGradI()(kerns, img, out, desc)
        return d_img