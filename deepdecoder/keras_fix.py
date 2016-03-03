# Copyright 2016 Leon Sixt
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
from keras import activations
from keras import initializations
from keras.layers.core import Layer
from keras import constraints
from keras import regularizers
import theano.tensor as T
import numpy as np


def conv_output_length(input_length, filter_size, border_mode, stride):
    if input_length is None:
        return None
    assert border_mode in {'same', 'full', 'valid'}
    if border_mode == 'same':
        output_length = input_length
    elif border_mode == 'full':
        output_length = input_length + filter_size - 1
    elif border_mode == 'valid':
        output_length = input_length - filter_size + 1
    return (output_length + stride - 1) // stride


def sharedX(X, dtype=theano.config.floatX, name=None):
    return theano.shared(np.asarray(X, dtype=dtype), name=name)


def shared_zeros(shape, dtype=theano.config.floatX, name=None):
    return sharedX(np.zeros(shape), dtype=dtype, name=name)


def on_gpu():
    return theano.config.device[:3] == 'gpu'

if on_gpu():
    from theano.sandbox.cuda import dnn


class Convolution2D(Layer):
    input_ndim = 4

    def __init__(self, nb_filter, nb_row, nb_col,
                 init='glorot_uniform', activation='linear', weights=None,
                 border_mode='valid', subsample=(1, 1),
                 W_regularizer=None, b_regularizer=None, activity_regularizer=None,
                 W_constraint=None, b_constraint=None, **kwargs):

        if border_mode not in {'valid', 'full', 'same'}:
            raise Exception('Invalid border mode for Convolution2D:', border_mode)
        self.nb_filter = nb_filter
        self.nb_row = nb_row
        self.nb_col = nb_col
        self.init = initializations.get(init)
        self.activation = activations.get(activation)
        self.border_mode = border_mode
        self.subsample = tuple(subsample)

        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)
        self.constraints = [self.W_constraint, self.b_constraint]

        self.initial_weights = weights
        super(Convolution2D, self).__init__(**kwargs)

    def build(self):
        stack_size = self.input_shape[1]
        self.input = T.tensor4()
        self.W_shape = (self.nb_filter, stack_size, self.nb_row, self.nb_col)
        self.W = self.init(self.W_shape)
        self.b = shared_zeros((self.nb_filter,))
        self.trainable_weights = [self.W, self.b]
        self.regularizers = []

        if self.W_regularizer:
            self.W_regularizer.set_param(self.W)
            self.regularizers.append(self.W_regularizer)

        if self.b_regularizer:
            self.b_regularizer.set_param(self.b)
            self.regularizers.append(self.b_regularizer)

        if self.activity_regularizer:
            self.activity_regularizer.set_layer(self)
            self.regularizers.append(self.activity_regularizer)

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights

    @property
    def output_shape(self):
        input_shape = self.input_shape
        rows = input_shape[2]
        cols = input_shape[3]
        rows = conv_output_length(rows, self.nb_row, self.border_mode, self.subsample[0])
        cols = conv_output_length(cols, self.nb_col, self.border_mode, self.subsample[1])
        return (input_shape[0], self.nb_filter, rows, cols)

    def get_output(self, train=False):
        X = self.get_input(train)
        border_mode = self.border_mode
        if on_gpu() and dnn.dnn_available():
            if border_mode == 'same':
                assert(self.subsample == (1, 1))
                pad_x = (self.nb_row - self.subsample[0]) // 2
                pad_y = (self.nb_col - self.subsample[1]) // 2
                conv_out = dnn.dnn_conv(img=X,
                                        kerns=self.W,
                                        border_mode=(pad_x, pad_y))
            else:
                conv_out = dnn.dnn_conv(img=X,
                                        kerns=self.W,
                                        border_mode=border_mode,
                                        subsample=self.subsample)
        else:
            if border_mode == 'same':
                border_mode = 'full'
                assert(self.subsample == (1, 1))

            conv_out = T.nnet.conv.conv2d(X, self.W,
                                          border_mode=border_mode,
                                          subsample=self.subsample,
                                          image_shape=self.input_shape,
                                          filter_shape=self.W_shape)
            if self.border_mode == 'same':
                shift_x = (self.nb_row - 1) // 2
                shift_y = (self.nb_col - 1) // 2
                conv_out = conv_out[:, :, shift_x:X.shape[2] + shift_x, shift_y:X.shape[3] + shift_y]

        return self.activation(conv_out + self.b.dimshuffle('x', 0, 'x', 'x'))

    def get_config(self):
        config = {"name": self.__class__.__name__,
                  "nb_filter": self.nb_filter,
                  "nb_row": self.nb_row,
                  "nb_col": self.nb_col,
                  "init": self.init.__name__,
                  "activation": self.activation.__name__,
                  "border_mode": self.border_mode,
                  "subsample": self.subsample,
                  "W_regularizer": self.W_regularizer.get_config() if self.W_regularizer else None,
                  "b_regularizer": self.b_regularizer.get_config() if self.b_regularizer else None,
                  "activity_regularizer": self.activity_regularizer.get_config() if self.activity_regularizer else None,
                  "W_constraint": self.W_constraint.get_config() if self.W_constraint else None,
                  "b_constraint": self.b_constraint.get_config() if self.b_constraint else None}
        base_config = super(Convolution2D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
