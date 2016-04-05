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
import pytest
import numpy as np
from keras.models import Sequential, Model
from keras.layers.core import Dense, Flatten, Reshape
from keras.optimizers import Adam
from keras.engine.topology import Input
import keras.backend as K

from beesgrid import TAG_SIZE
from deepdecoder.networks import gan_with_z_rot90_grid_idx, dcgan_generator, \
    mogan_learn_bw_grid, dcgan_discriminator, \
    dcgan_small_generator, get_offset_merge_mask, get_mask_driver, \
    get_offset_front, get_offset_middle, get_offset_back, \
    get_mask_weight_blending, get_lighting_generator


def test_networks_gan_with_z_rot90_grid_idx():
    n = TAG_SIZE*TAG_SIZE
    nb_z = 20
    generator = Sequential()
    generator.add(Dense(n, input_dim=nb_z+NB_GAN_GRID_PARAMS))
    generator.add(Reshape((1, TAG_SIZE, TAG_SIZE)))

    discriminator = Sequential()
    discriminator.add(Flatten(input_shape=(1, TAG_SIZE, TAG_SIZE)))
    discriminator.add(Dense(1, input_dim=n))

    gan = gan_with_z_rot90_grid_idx(generator, discriminator, nb_z=nb_z,
                                    reconstruct_fn=add_diff_rot90)
    gan.compile(Adam(), Adam())
    batch = next(gen_diff_gan(gan.batch_size * 10))
    gan.fit(batch.grid_bw, {'grid_idx': batch.grid_idx,
                            'z_rot90': batch.z_bins,
                            'grid_params': batch.params},
            nb_epoch=1, verbose=1)


@pytest.mark.skipif(not on_gpu(), reason="only works with cuda")
def test_mogan_learn_bw_grid():
    gan, _ = mogan_learn_bw_grid(dcgan_generator, dcgan_discriminator)
    gan.compile()


def test_autoencoder_mask_generator_loading(tmpdir):
    nb_input = 21
    g = dcgan_small_generator(input_dim=nb_input, nb_output_channels=2)
    fname = str(tmpdir) + "/g.hdf5"
    g.save_weights(fname)

    g_load = dcgan_small_generator(input_dim=nb_input, nb_output_channels=2)
    g_load.load_weights(fname)
    for a, b in zip(g.trainable_weights, g_load.trainable_weights):
        assert (K.get_value(a) == K.get_value(b)).all()


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
