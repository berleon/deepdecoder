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
from keras.layers.convolutional import Convolution2D, UpSampling2D, \
    MaxPooling2D
from keras.engine.topology import Input
from keras.optimizers import Adam
from beras.gan import GAN, gan_binary_crossentropy, gan_outputs
import pytest
from beesgrid import get_gt_files_in_dir, gt_grids, NUM_MIDDLE_CELLS, \
    CONFIG_CENTER, NUM_CONFIGS

from beras.util import sequential, concat
from deepdecoder.networks import mask_blending_generator, get_mask_driver, \
    get_lighting_generator, get_offset_merge_mask, get_mask_weight_blending, \
    get_offset_back, get_offset_front, get_offset_middle, mask_generator, \
    mask_blending_discriminator, get_mask_postprocess, get_decoder_model, \
    NormSinCosAngle

from deepdecoder.evaluate import GTEvaluator
from deepdecoder.blending_gan import lecture


def test_decoder_model():
    image_size = 64
    x = Input(shape=(1, image_size, image_size))
    out = get_decoder_model(x, nb_units=8,
                            nb_output=NUM_MIDDLE_CELLS + NUM_CONFIGS + 1)
    model = Model(x, out)
    gt_dir = '/home/leon/repos/deeplocalizer_data/images/season_2015'
    gt_files = get_gt_files_in_dir(gt_dir)
    gt, ids, config = next(gt_grids(gt_files, all=True, center='zeros'))
    lec = lecture()
    ids_norm = lec.normalize_ids(ids)
    config[:, CONFIG_CENTER] = 0
    config_norm = lec.normalize_config(config)
    config_norm[:, CONFIG_CENTER] = 0

    print("config: {}, {}".format(config_norm.max(), config_norm.min()))
    model.compile('adam', 'mse')
    model.fit(gt, np.concatenate([ids_norm, config_norm], axis=1), nb_epoch=50)
    evaluator = GTEvaluator(gt_files)

    def predict(x):
        outs = model.predict(x)
        ids = outs[:, :NUM_MIDDLE_CELLS]
        config = outs[:, NUM_MIDDLE_CELLS:]
        config_denorm = lec.denormalize_config(config)
        config_denorm[:, CONFIG_CENTER] = 0
        return lec.denormalize_ids(ids), config_denorm
    results = evaluator.evaluate(predict)

    for name, value in sorted(results.items()):
        if "z_rotation" in name:
            continue
        elif "radius" in name:
            assert value < 0.3, name
        elif name.startswith("mse"):
            assert value < 0.1, name
        elif name.startswith("accuracy"):
            assert abs(value - 1) < 0.01, name

        print("{}: {}".format(name, value))


def test_norm_sin_cos_angle():
    batch_shape = (64, 4)
    input = Input(shape=batch_shape[1:])
    output = NormSinCosAngle(idx=0)(input)
    model = Model(input=input, output=output)
    model.compile('adam', 'mse')
    x = (5*np.random.sample(batch_shape)).astype(np.float32)
    y = model.predict(x)
    print(y.shape)
    assert y.shape == x.shape
    assert (y != x).any()

    angle = np.random.uniform(-np.pi, np.pi, batch_shape)
    s = np.sin(angle)
    c = np.cos(angle)
    np.testing.assert_allclose(s**2 + c**2, np.ones_like(s))

    s = y[:, 0]
    c = y[:, 1]
    np.testing.assert_allclose(s**2 + c**2, np.ones_like(s, dtype=np.float32),
                               rtol=1e-5, atol=1e-5)


def test_get_offset_merge_mask():
    batch_shape = (64, 1, 16, 16)
    input = Input(shape=batch_shape[1:])
    output = get_offset_merge_mask([input], nb_units=2, nb_conv_layers=1)
    model = Model(input, output)
    model.compile('adam', 'mse')
    x = np.random.sample(batch_shape)
    model.train_on_batch(x, np.concatenate([x, x], axis=1))


def test_get_mask_driver():
    batch_shape = (64, 22)
    n = 50
    input = Input(shape=batch_shape[1:])
    output = get_mask_driver(input, nb_units=n, nb_output_units=n)
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
    output = get_offset_back([a_input, b_input], n)[1]
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

    scale_black, scale_white, shift64 = get_lighting_generator(
        [a_input, b_input, c_input], n)

    model = Model([a_input, b_input, c_input],
                  [scale_black, scale_white, shift64])
    model.compile('adam', 'mse')
    bs = (64, )
    a = np.random.sample(bs + a_shape)
    b = np.random.sample(bs + b_shape)
    c = np.random.sample(bs + c_shape)

    y_scale_black = np.random.sample(bs + (1, 64, 64))
    y_scale_white = np.random.sample(bs + (1, 64, 64))
    y_shift64 = np.random.sample(bs + (1, 64, 64))
    model.train_on_batch([a, b, c], [y_scale_black, y_scale_white, y_shift64])


def test_mask_generator():
    shape = (15, )
    input = Input(shape=shape)
    output = mask_generator([input])
    model = Model(input, output)
    model.compile('adam', 'mse')
    bs = (64, )
    x = np.random.sample(bs + shape)
    y_depth_map = np.random.sample(bs + (1, 16, 16))
    y_mask = np.random.sample(bs + (1, 64, 64))
    model.train_on_batch(x, [y_mask, y_depth_map])


def test_mask_blending_discriminator():
    fake_shape = (10, 1, 64, 64)
    real_shape = (10, 1, 64, 64)
    fake = Input(batch_shape=fake_shape)
    real = Input(batch_shape=real_shape)
    output = mask_blending_discriminator([fake, real])
    model = Model([fake, real], output)
    model.compile('adam', 'mse')
    f = np.random.sample(fake_shape)
    r = np.random.sample(real_shape)
    y = np.random.sample((fake_shape[0], 1,))
    # TODO: fix different batch sizes for input and output
    # y = np.random.sample((fake_shape[0] + real_shape[0], 1,))
    with pytest.raises(ValueError):
        model.train_on_batch([f, r], y)


def test_mask_blending_generator():
    nb_driver = 20

    def driver(z):
        return Dense(nb_driver)(z)

    def mask_generator(x):
        mask = sequential([
            Dense(16),
            Reshape((1, 4, 4)),
            UpSampling2D((16, 16))
        ])(x)

        depth_map = sequential([
            Dense(16),
            Reshape((1, 4, 4)),
            UpSampling2D((4, 4))
        ])(x)
        return mask, depth_map

    def merge_mask(subsample):
        def call(x):
            if subsample:
                x = MaxPooling2D(subsample)(x)
            return Convolution2D(1, 3, 3, border_mode='same')(x)
        return call

    def light_generator(ins):
        seq = sequential([
            Convolution2D(1, 3, 3, border_mode='same')
        ])(concat(ins))
        return UpSampling2D((4, 4))(seq), UpSampling2D((4, 4))(seq), \
            UpSampling2D((4, 4))(seq),

    def offset_front(x):
        return sequential([
            Dense(16),
            Reshape((1, 4, 4)),
            UpSampling2D((4, 4))
        ])(concat(x))

    def offset_middle(x):
        return UpSampling2D()(concat(x))

    def offset_back(x):
        feature_map = sequential([
            UpSampling2D(),
        ])(concat(x))
        return feature_map, Convolution2D(1, 3, 3,
                                          border_mode='same')(feature_map)

    def mask_post(x):
        return sequential([
            Convolution2D(1, 3, 3, border_mode='same')
        ])(concat(x))

    def mask_weight_blending(x):
        return sequential([
            Flatten(),
            Dense(1),
        ])(x)

    def discriminator(x):
        return gan_outputs(sequential([
            Flatten(),
            Dense(1),
        ])(concat(x)), fake_for_gen=(0, 10), fake_for_dis=(0, 10),
                           real=(10, 20))

    gen = mask_blending_generator(
        mask_driver=driver,
        mask_generator=mask_generator,
        light_merge_mask16=merge_mask(None),
        offset_merge_light16=merge_mask((4, 4)),
        offset_merge_mask16=merge_mask(None),
        offset_merge_mask32=merge_mask(None),
        lighting_generator=light_generator,
        offset_front=offset_front,
        offset_middle=offset_middle,
        offset_back=offset_back,
        mask_weight_blending32=mask_weight_blending,
        mask_weight_blending64=mask_weight_blending,
        mask_postprocess=mask_post,
        z_for_driver=(0, 10),
        z_for_offset=(10, 20),
        z_for_bits=(20, 32),
    )
    z_shape = (32, )
    real_shape = (1, 64, 64)
    gan = GAN(gen, discriminator, z_shape, real_shape)
    gan.build(Adam(), Adam(), gan_binary_crossentropy)
    for l in gan.gen_layers:
        print("{}: {}, {}".format(
            l.name, l.output_shape, getattr(l, 'regularizers', [])))
    bs = 10
    z_in = np.random.sample((bs,) + z_shape)
    gan.compile_generate()
    gan.generate({'z': z_in})
