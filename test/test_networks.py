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

import numpy as np
from keras.models import Model
from keras.layers.core import Dense, Flatten, Reshape
from keras.layers.convolutional import Convolution2D, UpSampling2D
from keras.engine.topology import Input

from deepdecoder.networks import *


def test_get_offset_merge_mask():
    batch_shape = (64, 1, 16, 16)
    input = Input(shape=batch_shape[1:])
    output = get_offset_merge_mask([input], 2, nb_conv_layers=1)
    model = Model(input, output)
    model.compile('adam', 'mse')
    x = np.random.sample(batch_shape)
    model.train_on_batch(x, np.concatenate([x, x], axis=1))


def test_get_mask_driver():
    batch_shape = (64, 22)
    n = 50
    input = Input(shape=batch_shape[1:])
    output = get_mask_driver(input, n)
    model = Model(input, output)
    model.compile('adam', 'mse')
    x = np.random.sample(batch_shape)
    y = np.random.sample((64, n))
    model.train_on_batch(x, y)


def test_get_offset_front():
    a_shape = (22, )
    b_shape = (10, )
    n = 5
    a_input = Input(shape=a_shape)
    b_input = Input(shape=b_shape)
    output = get_offset_front([a_input, b_input], n)
    model = Model([a_input, b_input], output)
    model.compile('adam', 'mse')
    bs = (64, )
    a = np.random.sample(bs + a_shape)
    b = np.random.sample(bs + b_shape)
    y = np.random.sample(bs + (2*n, 16, 16))
    model.train_on_batch([a, b], y)


def test_get_offset_middle():
    a_shape = (5, 16, 16)
    b_shape = (10, 16, 16)
    n = 5
    a_input = Input(shape=a_shape)
    b_input = Input(shape=b_shape)
    output = get_offset_middle([a_input, b_input], n)
    model = Model([a_input, b_input], output)
    model.compile('adam', 'mse')
    bs = (64, )
    a = np.random.sample(bs + a_shape)
    b = np.random.sample(bs + b_shape)
    y = np.random.sample(bs + (2*n, 32, 32))
    model.train_on_batch([a, b], y)


def test_get_offset_back():
    a_shape = (5, 32, 32)
    b_shape = (10, 32, 32)
    n = 5
    a_input = Input(shape=a_shape)
    b_input = Input(shape=b_shape)
    output = get_offset_back([a_input, b_input], n)
    model = Model([a_input, b_input], output)
    model.compile('adam', 'mse')
    bs = (64, )
    a = np.random.sample(bs + a_shape)
    b = np.random.sample(bs + b_shape)
    y = np.random.sample(bs + (1, 64, 64))
    model.train_on_batch([a, b], y)


def test_get_mask_weight_blending():
    a_shape = (5, 32, 32)
    b_shape = (10, 32, 32)
    a_input = Input(shape=a_shape)
    b_input = Input(shape=b_shape)
    output = get_mask_weight_blending([a_input, b_input])
    model = Model([a_input, b_input], output)
    model.compile('adam', 'mse')
    bs = (64, )
    a = np.random.sample(bs + a_shape)
    b = np.random.sample(bs + b_shape)
    y = np.random.sample(bs + (1,))
    model.train_on_batch([a, b], y)


def test_get_lighting_generator():
    a_shape = (5, 16, 16)
    b_shape = (1, 16, 16)
    c_shape = (1, 16, 16)
    n = 5
    a_input = Input(shape=a_shape)
    b_input = Input(shape=b_shape)
    c_input = Input(shape=c_shape)

    scale16, shift16, scale32, shift32 = \
        get_lighting_generator([a_input, b_input, c_input], n)

    model = Model([a_input, b_input, c_input], [scale16, shift16, scale32, shift32])
    model.compile('adam', 'mse')
    bs = (64, )
    a = np.random.sample(bs + a_shape)
    b = np.random.sample(bs + b_shape)
    c = np.random.sample(bs + c_shape)

    y_scale16 = np.random.sample(bs + (1, 16, 16))
    y_shift16 = np.random.sample(bs + (1, 16, 16))
    y_scale32 = np.random.sample(bs + (1, 32, 32))
    y_shift32 = np.random.sample(bs + (1, 32, 32))
    model.train_on_batch([a, b, c],
                         [y_scale16, y_shift16, y_scale32, y_shift32])


def test_mask_generator():
    shape = (15, )
    input = Input(shape=shape)
    output = mask_generator([input])
    model = Model(input, output)
    model.compile('adam', 'mse')
    bs = (64, )
    x = np.random.sample(bs + shape)
    y = np.random.sample(bs + (1, 32, 32))
    model.train_on_batch(x, y)


def test_mask_blending_discriminator():
    fake_shape = (5, 1, 64, 64)
    real_shape = (10, 1, 64, 64)
    fake = Input(batch_shape=fake_shape)
    real = Input(batch_shape=real_shape)
    output = mask_blending_discriminator([fake, real])
    model = Model([fake, real], output)
    model.compile('adam', 'mse')
    f = np.random.sample(fake_shape)
    r = np.random.sample(real_shape)
    y = np.random.sample((fake_shape[0] + real_shape[1], 1,))
    model.train_on_batch([f, r], y)


def test_mask_blending_generator():
    def driver(z):
        return sequential([
            Dense(16),
        ])(z)

    def mask_generator(x):
        return sequential([
            Dense(16),
            Reshape((1, 4, 4)),
            UpSampling2D((8, 8))
        ])(x)

    def merge_mask(x):
        return sequential([
            Convolution2D(1, 3, 3, border_mode='same')
        ])(x)

    def light_generator(ins):
        seq = sequential([
            Convolution2D(1, 3, 3, border_mode='same')
        ])(concat(ins))
        return seq, seq, UpSampling2D()(seq), UpSampling2D()(seq),

    def offset_front(x):
        return sequential([
            Dense(16),
            Reshape((1, 4, 4)),
            UpSampling2D((4, 4))
        ])(x)

    def offset_middle(x):
        return UpSampling2D()(concat(x))

    def offset_back(x):
        return sequential([
            UpSampling2D(),
            Convolution2D(1, 3, 3, border_mode='same')
        ])(concat(x))

    def mask_weight_blending(x):
        return sequential([
            Flatten(),
            Dense(1),
        ])(x)

    gen = mask_blending_generator(
        mask_driver=driver,
        mask_generator=mask_generator,
        light_merge_mask16=merge_mask,
        offset_merge_mask16=merge_mask,
        offset_merge_mask32=merge_mask,
        lighting_generator=light_generator,
        offset_front=offset_front,
        offset_middle=offset_middle,
        offset_back=offset_back,
        mask_weight_blending=mask_weight_blending)

    z = Input(shape=(20,), name='z')
    fake = gen([z])
    model = Model(z, fake)
    model.compile('adam', 'mse')
    z_in = np.random.sample((4, 20))
    model.predict_on_batch(z_in)
