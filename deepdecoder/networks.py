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
import numpy as np
import copy

from keras.models import Sequential, Graph
from keras.objectives import mse, binary_crossentropy
import keras.initializations
from keras.layers.core import Dense, Flatten, Reshape, Activation, \
    Layer, Merge, Dropout, Lambda
from keras.layers.convolutional import Convolution2D, MaxPooling2D, UpSampling2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.normalization import BatchNormalization
from keras.layers.noise import GaussianNoise
from keras.engine.topology import merge
import keras.backend as K

from beras.layers.attention import RotationTransformer, GraphSpatialTransformer
from beras.layers.transform import iDCT
from beras.gan import GAN, add_gan_outputs
from beras.util import sequential, concat
from beras.regularizers import ActivityInBoundsRegularizer, SumBelow
from beras.layers.core import Split, ZeroGradient, LinearInBounds

from beesgrid import NUM_MIDDLE_CELLS, TAG_SIZE

from deepdecoder.deconv import Deconvolution2D, Deconv2DVariableWeights
from deepdecoder.keras_fix import Convolution2D as TheanoConvolution2D
from deepdecoder.utils import binary_mask, rotate_by_multiple_of_90
from deepdecoder.data import nb_normalized_params
from deepdecoder.mask_loss import pyramid_loss
from deepdecoder.transform import PyramidBlending, PyramidReduce, \
    AddLighting, Selection, GaussianBlur, UpsampleInterpolate, \
    DifferenceOfGaussians


def get_decoder_model(nb_output=NUM_MIDDLE_CELLS, inputs=['input'],
                      add_output=True, batch_size=None):

    def conv_bn_relu(model, nb_filter, kernel_size=(3, 3), **kwargs):
        model.add(Convolution2D(nb_filter, *kernel_size, **kwargs))
        model.add(BatchNormalization(axis=1))
        model.add(Activation('relu'))

    size = 64
    rot_model = Sequential()
    conv_bn_relu(rot_model, 16, batch_input_shape=(batch_size, 1, size, size))
    rot_model.add(MaxPooling2D((3, 3)))
    conv_bn_relu(rot_model, 16)
    rot_model.add(MaxPooling2D((3, 3)))
    conv_bn_relu(rot_model, 16)
    rot_model.add(MaxPooling2D((3, 3)))
    rot_model.add(Flatten())
    rot_model.add(Dense(64))
    rot_model.add(BatchNormalization(axis=1))
    rot_model.add(Activation('relu'))

    rot_model.add(Dense(1, activation='linear'))

    model = Sequential()
    conv_bn_relu(model, 32, batch_input_shape=(batch_size, 1, size, size))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    conv_bn_relu(model, 64)
    model.add(MaxPooling2D(pool_size=(2, 2)))
    conv_bn_relu(model, 128)
    model.add(MaxPooling2D(pool_size=(2, 2)))
    conv_bn_relu(model, 256)
    model.add(Flatten())

    g = Graph()
    for input in inputs:
        g.add_input(input, batch_input_shape=(batch_size, 1, size, size))

    g.add_node(rot_model, "rot", inputs=inputs,
               concat_axis=0)
    g.add_node(RotationTransformer(rot_model), "rot_input", inputs=inputs,
               concat_axis=0)
    g.add_node(model, "conv_encoder", input="rot_input")
    g.add_node(Dense(1024, activation='relu'), "dense1",
               inputs=["conv_encoder", "rot"])
    g.add_node(Dense(nb_output, activation='linear'), "encoder",
               input="dense1")
    if add_output:
        g.add_output("output", input="encoder")
    return g


def autoencoder_mask_decoder(nb_output=21, batch_size=128,
                             inputs=['mask', 'real']):

    def conv_bn_relu(model, nb_filter, kernel_size=(3, 3), **kwargs):
        model.add(Convolution2D(nb_filter, *kernel_size, **kwargs))
        model.add(BatchNormalization(axis=1))
        model.add(Activation('relu'))
    size = 64
    shape = (len(inputs)*batch_size, 1, 64, 64)
    pre_st = Sequential()
    pre_st.add(Convolution2D(48, 3, 3, activation='relu',
               batch_input_shape=shape))
    conv_bn_relu(pre_st, 48, border_mode='same')
    pre_st.add(MaxPooling2D())
    pre_st.add(Convolution2D(96, 3, 3, activation='relu'))
    conv_bn_relu(pre_st, 96, border_mode='same')

    print(pre_st.output_shape)
    st_model = Sequential()
    st_model.add(MaxPooling2D((3, 3),
                 batch_input_shape=pre_st.output_shape))
    conv_bn_relu(st_model, 48)
    st_model.add(Flatten())
    st_model.add(Dense(128))
    st_model.add(BatchNormalization(axis=1))
    st_model.add(Activation('relu'))
    st_model.add(Dense(6, activation='linear'))

    post_st = Sequential()
    post_st.add(MaxPooling2D(batch_input_shape=pre_st.output_shape))
    post_st.add(Convolution2D(96, 3, 3, activation='relu'))
    conv_bn_relu(post_st, 96, border_mode='same')
    post_st.add(MaxPooling2D())
    post_st.add(Convolution2D(128, 3, 3, activation='relu'))
    conv_bn_relu(post_st, 128, border_mode='same')
    post_st.add(Flatten())

    g = Graph()
    for input in inputs:
        g.add_input(input, batch_input_shape=(batch_size, 1, size, size))

    g.add_node(pre_st, "pre_st", inputs=inputs,
               concat_axis=0)
    g.add_node(st_model, "st", input="pre_st",
               concat_axis=0)
    g.add_node(GraphSpatialTransformer(st_model), "st_input", input="pre_st",
               concat_axis=0)

    g.add_node(post_st, "post_st", input="st_input")
    g.add_node(Dense(512, activation='relu'), "dense1",
               inputs=["post_st", "st"])
    g.add_node(Dense(512, activation='relu'), "dense2",
               input="dense1")
    g.add_node(Dense(nb_output, activation='linear'), "encoder",
               input="dense2")
    return g


def batch_norm():
    mode = 0
    bn = BatchNormalization(mode, axis=1)
    return bn


def normal(scale=0.02):
    def normal_wrapper(shape, name=None):
        return keras.initializations.normal(shape, scale, name)
    return normal_wrapper


def deconv_bn(model, nb_filter, h=3, w=3, init=normal(0.02),
              activation='relu', upsample=True, **kwargs):
    if upsample:
        subsample = (2, 2)
    else:
        subsample = (1, 1)

    model.add(Deconvolution2D(nb_filter, h, w, subsample=subsample,
                              border_mode=(h//2, w//2), init=init, **kwargs))
    model.add(BatchNormalization(axis=1))
    if activation is not None:
        model.add(Activation(activation))


def mask_generator_with_conv(nb_units=64, input_dim=20, init=normal(0.02),
                             nb_output_channels=1, filter_size=3):
    n = nb_units
    f = filter_size
    def deconv(nb_filter, h, w):
        return Deconvolution2D(nb_filter, h, w, subsample=(2, 2),
                               border_mode=(h//2, w//2), init=init)

    model = Sequential()
    model.add(Dense(5*input_dim, input_dim=input_dim, init=init))
    model.add(batch_norm())
    model.add(Activation('relu'))

    model.add(Dense(8*n*4*4, input_dim=input_dim, init=init))
    model.add(batch_norm())
    model.add(Activation('relu'))
    model.add(Reshape((8*n, 4, 4,)))

    model.add(deconv(4*n, f, f))
    model.add(batch_norm())
    model.add(Activation('relu'))

    model.add(Convolution2D(4*n, f, f, border_mode='same'))
    model.add(batch_norm())
    model.add(Activation('relu'))

    model.add(deconv(2*n, f, f))
    model.add(batch_norm())
    model.add(Activation('relu'))

    model.add(Convolution2D(2*n, f, f, border_mode='same'))
    model.add(batch_norm())
    model.add(Activation('relu'))

    model.add(deconv(n, f, f))
    model.add(batch_norm())
    model.add(Activation('relu'))

    model.add(Convolution2D(n, f, f, border_mode='same'))
    model.add(batch_norm())
    model.add(Activation('relu'))

    model.add(Deconvolution2D(nb_output_channels,
                              f, f, border_mode=(1, 1), init=init))
    model.add(Activation('linear'))
    return model


def mask_generator(nb_units=64, input_dim=20, init=normal(0.02),
                   dense_factor=3,
                   nb_dense_layers=2):
    n = nb_units

    def conv(n):
        model.add(Convolution2D(n, 3, 3, border_mode='same',
                                init='he_normal'))
        model.add(Activation('relu'))

    def up():
        model.add(UpSampling2D())

    model = Sequential()
    model.add(Layer(input_shape=(input_dim,)))
    for _ in range(nb_dense_layers):
        model.add(Dense(dense_factor*nb_units, input_dim=input_dim,
                        activation='relu'))

    model.add(Dense(8*n*4*4, input_dim=input_dim))
    model.add(Activation('relu'))
    model.add(Reshape((8*n, 4, 4,)))
    up()  # 8
    conv(4*n)
    conv(4*n)
    up()  # 16
    conv(2*n)
    conv(2*n)
    up()  # 32
    conv(n)
    model.add(Deconvolution2D(1, 3, 3, border_mode=(1, 1), init=init))

    model.add(Activation('linear'))
    return model


def dcgan_small_generator(nb_units=64, input_dim=20, init=normal(0.02),
                          dense_factor=2,
                          nb_dense_layers=2,
                          nb_output_channels=1, filter_size=3,
                          deconv_layers=False,
                          output_size=32):
    n = nb_units
    f = filter_size

    def deconv_up(nb_filter, h, w):
        return Deconvolution2D(nb_filter, h, w, subsample=(2, 2),
                               border_mode=(h//2, w//2), init=init)

    def deconv(nb_filter, h, w):
        return Deconvolution2D(nb_filter, h, w, subsample=(1, 1),
                               border_mode=(h//2, w//2), init=init)
    model = Sequential()
    model.add(Layer(input_shape=(input_dim,)))
    for _ in range(nb_dense_layers):
        model.add(Dense(dense_factor*nb_units, input_dim=input_dim,
                        activation='relu'))

    model.add(Dense(8*n*4*4, input_dim=input_dim))
    model.add(Activation('relu'))
    model.add(Reshape((8*n, 4, 4,)))

    if deconv_layers:
        model.add(deconv(8*n, f, f))
        model.add(Activation('relu'))

    model.add(deconv_up(4*n, f, f))
    model.add(Activation('relu'))

    if deconv_layers:
        model.add(deconv(4*n, f, f))
        model.add(Activation('relu'))

    if output_size >= 16:
        model.add(deconv_up(2*n, f, f))
        model.add(Activation('relu'))

        if deconv_layers:
            model.add(deconv(2*n, f, f))
            model.add(Activation('relu'))

    if output_size == 32:
        model.add(deconv_up(n, f, f))
        model.add(Activation('relu'))

    if deconv_layers:
        model.add(deconv(n, f, f))
        model.add(Activation('relu'))

    model.add(Deconvolution2D(nb_output_channels,
                              f, f, border_mode=(1, 1), init=init))

    model.add(Activation('linear'))
    return model


def dcgan_pyramid_mse_generator(generator, input_dim=40):
    g = Graph()
    g.add_input("input", input_shape=(input_dim, ))
    g.add_node(generator, "generator", input="input")
    g.add_node(Split(-2, None, axis=1), "mean", input="input")
    g.add_output("output", input="generator")
    return g


def autoencoder(encoder, decoder):
    encoder.add_node(decoder, "decoder", input='encoder')
    encoder.add_output("output", input="decoder")
    return encoder


def autoencoder_with_mask_loss(encoder, decoder, batch_size=128):
    half = batch_size // 2
    encoder.add_node(Split(0, half), 'encoder_real', input='encoder')
    encoder.add_node(Split(half, batch_size), 'encoder_mask', input='encoder',
                     create_output=True)
    encoder.add_node(decoder, "decoder", input='encoder_real',
                     create_output=True)
    return encoder


def get_mask_driver(x):
    driver = sequential([
        Dense(64),
        BatchNormalization(),
        Dropout(0.25),
        Activation('relu'),
        Dense(64),
        BatchNormalization(),
        Dropout(0.25),
        Activation('relu'),
    ])
    return driver(x)


def mask_blending_generator(mask_driver, mask_generator,
                            offset_inputs=['gen_driver'],
                            nb_units=64,
                            nb_merge_conv_layers=0,
                            ):
    assert len(mask_generator.input_shape) == 2

    def get_offset_front(inputs, nb_units):
        n = nb_units
        input = concat(inputs)

        return sequential([
            Dense(8*n*4*4),
            BatchNormalization(),
            Activation('relu'),
            Reshape((8*n, 4, 4)),
            UpSampling2D(),  # 8x8
            conv(4*n, 3, 3),
            UpSampling2D(),  # 16x16
            conv(2*n, 3, 3),
        ])(input)

    def get_offset_middle(inputs, nb_units):
        n = nb_units
        input = concat(inputs)
        return sequential([
            UpSampling2D(),  # 32x32
            conv(2*n, 3, 3),
            conv(2*n, 3, 3),
        ])(input)

    def get_offset_back(inputs, nb_units):
        n = nb_units
        input = concat(inputs)
        return sequential([
            UpSampling2D(),  # 64x64
            conv(n, 3, 3),
            Deconvolution2D(1, 3, 3, border_mode=(1, 1)),
            LinearInBounds(-1, 1, clip=True),
        ])(input)

    def get_mask_weight_blending(inputs):
        input = concat(inputs)
        return sequential([
            Convolution2D(1, 3, 3),
            Flatten(),
            Dense(1),
            LinearInBounds(0.1, 4, clip=True),
        ])(input)

    def get_lighting_generator(inputs, nb_units):
        # front_back_ins = ['gen_light_merge16_mask']
        n = nb_units
        light_conv = sequential([
            conv(n, 3, 3),
            Convolution2D(2, 1, 1, border_mode='same'),
            LinearInBounds(-1, 1, clip=True),
        ])

        shift_split = Split(0, 1, axis=1)(light_conv)
        light_shift16 = GaussianBlur(sigma=2)(shift_split)
        light_shift32 = sequential([
            UpSampling2D(),
            GaussianBlur(sigma=8),
        ])(light_shift16)

        scale_split = Split(1, 2, axis=1)(light_conv)
        light_scale16 = GaussianBlur(sigma=2)(scale_split)

        light_scale32 = sequential([
            UpSampling2D(),
            GaussianBlur(sigma=6),
        ])(light_scale16)

        # TODO:
        # dog = DifferenceOfGaussians(2, 6, window_radius2=10)
        # sum_below = SumBelow(1)
        # dog.regularizers = [sum_below]
        # sum_below.set_layer(dog)
        return light_scale16, light_shift16, light_scale32, light_shift32

    def get_offset_merge_mask(inputs, nb_units, nb_conv_layers):
        assert len(inputs) == 1
        input = inputs[0]

        def conv_layers():
            return [
                Convolution2D(nb_units, 3, 3, border_mode='same'),
                BatchNormalization(axis=1),
                Activation('relu'),
            ]
        layers = [Layer()]
        for i in range(nb_conv_layers):
            layers.extends(conv_layers())
        return sequential(layers)(input)

    n = nb_units

    def generator(inputs):
        z,  = inputs
        driver = mask_driver(z)
        mask = mask_generator(driver)
        mask_down = PyramidReduce()(mask)

        offset_mask16 = get_offset_merge_mask(mask_down, n // 2,
                                              nb_merge_conv_layers)

        light_mask16 = get_offset_merge_mask(mask_down, n // 2,
                                             nb_merge_conv_layers)

        offset_front = get_offset_front([z], n)
        light_scale16, light_shift16, light_scale32, light_shift32 = \
            get_lighting_generator([offset_front, light_mask16])

        offset_middle = get_offset_middle(
            [offset_front, offset_mask16, light_scale16, light_shift16], n)

        mask_weight32 = get_mask_weight_blending(offset_middle)

        mask_selection = Selection(threshold=-0.03, smooth_threshold=0.08,
                                   sigma=1.5)(mask_generator)

        mask_with_lighting = AddLighting(scale_factor=0.75, shift_factor=1)(
                [light_scale32, light_shift32, mask])

        offset_mask_light32 = get_offset_merge_mask(
            mask_down, n // 2, nb_merge_conv_layers)

        offset_back = get_offset_back([offset_middle, offset_mask_light32])

        blending = PyramidBlending(input_pyramid_layers=3,
                                   mask_pyramid_layers=2)(
                [mask_with_lighting, mask_selection, offset_back,
                 mask_weight32])
        return LinearInBounds(-1, 1)(blending)

    return generator


def conv(nb_filter, h, w, activation='relu'):
    if issubclass(type(activation), Layer):
        activation_layer = activation
    else:
        activation_layer = Activation(activation)

    def call(x):
        return sequential([
            Convolution2D(nb_filter, h, w, border_mode='same'),
            BatchNormalization(axis=1),
            activation_layer
        ])(x)
    return call


def deconv(model, nb_filter, h, w, activation='relu'):
    def deconv_init(shape, name=None):
        w = np.random.normal(0, 0.02, shape).astype(np.float32)
        nb_filter = shape[1]
        mask = np.random.binomial(1, 10/nb_filter, (1, nb_filter, 1, 1))
        w *= mask
        return K.variable(w, name=name)

    model.add(Deconvolution2D(nb_filter, h, w, border_mode=(1, 1),
                              init=deconv_init))
    model.add(BatchNormalization(axis=1))
    if issubclass(type(activation), Layer):
        model.add(activation)
    else:
        model.add(Activation(activation))


def up(size=(2, 2)):
    def call(x):
        return UpSampling2D(size)(x)
    return cal


def get_offset_generator(n=32, batch_input_shape=50, nb_output_channels=1,
                         init=normal(0.02), merge_layer=None):
    def middle_input_shape():
        input_shape = list(front.output_shape)
        if merge_layer is not None:
            input_shape[1] += merge_layer.output_shape[1]
        return tuple(input_shape)

    front = Sequential()
    front.add(Dense(8*n*4*4, batch_input_shape=batch_input_shape, init=init))
    front.add(BatchNormalization())
    front.add(Activation('relu'))
    front.add(Reshape((8*n, 4, 4,)))

    up(front)  # 8x8
    conv(front, 4*n, 3, 3)

    up(front)  # 16x16
    conv(front, 2*n, 3, 3)

    middle = Sequential()
    middle.add(Layer(batch_input_shape=middle_input_shape()))
    up(middle)
    conv(middle, n, 3, 3)
    deconv(front, n, 3, 3)
    conv(middle, n, 3, 3)
    return front, middle


def dcgan_generator(n=32, input_dim=50, nb_output_channels=1, use_dct=False,
                    init=normal(0.02)):

    def deconv(nb_filter, h, w):
        return Deconvolution2D(nb_filter, h, w, subsample=(2, 2),
                               border_mode=(2, 2), init=init)
    model = Sequential()
    model.add(Dense(8*n*4*4, input_dim=input_dim, init=init))
    model.add(batch_norm())
    model.add(Reshape((8*n, 4, 4,)))
    if use_dct:
        model.add(iDCT())
    model.add(Activation('relu'))

    model.add(deconv(4*n, 5, 5))
    model.add(batch_norm())
    model.add(Activation('relu'))

    model.add(deconv(2*n, 5, 5))
    model.add(batch_norm())
    model.add(Activation('relu'))

    model.add(deconv(n, 5, 5))
    model.add(batch_norm())
    model.add(Activation('relu'))

    model.add(Deconvolution2D(nb_output_channels, 5, 5, subsample=(2, 2),
                              border_mode=(2, 2), init=init))
    model.add(LinearInBounds(-1, 1))
    return model


def dcgan_generator_conv(n=32, input_dim=50, nb_output_channels=1,
                         init=normal(0.02)):
    def conv(nb_filter, h, w):
        model.add(Convolution2D(nb_filter, h, w, border_mode='same',
                                init=init))
        model.add(BatchNormalization(axis=1))
        model.add(Activation('relu'))

    def deconv(nb_filter, h, w):
        deconv_layer = Deconvolution2D(nb_filter, h, w, border_mode=(1, 1),
                                       init=init)
        model.add(deconv_layer)

        w = np.random.normal(0, 0.02, deconv_layer.W_shape).astype(np.float32)
        w *= np.random.uniform(0, 1, (1, w.shape[1], 1, 1))
        deconv_layer.W.set_value(w)
        model.add(BatchNormalization(axis=1))
        model.add(Activation('relu'))

    def up():
        model.add(UpSampling2D())

    z = Layer(input_shape=(input_dim,))
    model = Sequential()
    model.add(z)
    model.add(Dense(8*n*4*4, init=init))
    model.add(batch_norm())
    model.add(Reshape((8*n, 4, 4,)))
    model.add(Activation('relu'))

    up()  # 8
    conv(4*n, 3, 3)
    up()  # 16
    conv(2*n, 3, 3)
    conv(2*n, 3, 3)
    up()  # 32
    conv(n, 3, 3)
    conv(n, 3, 3)
    up()  # 64
    conv(n, 3, 3)

    model.add(Deconvolution2D(nb_output_channels, 3, 3, border_mode=(1, 1),
                              init=init))
    model.add(LinearInBounds(-1, 1))
    return model


def mask_blending_gan(offset_generator, mask_generator, discriminator,
                      nb_fake=64, nb_real=32):
    assert len(mask_generator.input_shape) == 2
    assert len(offset_generator.input_shape) == 2

    g = Graph()
    mask_input_dim = mask_generator.input_shape[1]
    z_shape = (nb_fake, offset_generator.input_shape[1] - mask_input_dim)
    g.add_input(GAN.z_name, batch_input_shape=z_shape)

    g.add_node(Dense(mask_input_dim, activation='relu'), 'dense_mask',
               input=GAN.z_name)
    g.add_node(mask_generator, 'mask_generator', input='dense_mask')
    g.add_node(offset_generator, 'gen_offset',
               inputs=[GAN.z_name, 'dense_mask'])
    g.add_node(PyramidBlending(mask_generator, input_pyramid_layers=3,
                               mask_pyramid_layers=2),
               GAN.generator_name,
               input='gen_offset')

    real_shape = (nb_real,) + g.nodes[GAN.generator_name].output_shape[1:]
    g.add_input(GAN.real_name, batch_input_shape=real_shape)
    g.add_node(discriminator, "discriminator",
               inputs=[GAN.generator_name, "real"], concat_axis=0)
    add_gan_outputs(g, fake_for_gen=(0, nb_fake),
                    fake_for_dis=(nb_fake//2, nb_fake),
                    real=(nb_fake, nb_fake+nb_real))
    return g


def mask_blending_gan_new(offset_generator, mask_generator, discriminator,
                          nb_fake=64, nb_real=32):
    assert len(mask_generator.input_shape) == 2
    assert len(offset_generator.input_shape) == 2

    g = Graph()
    mask_input_dim = mask_generator.input_shape[1]
    z_shape = (nb_fake, offset_generator.input_shape[1] - mask_input_dim)

    g.add_input(GAN.z_name, batch_input_shape=z_shape)

    g.add_node(Dense(32), 'gen_driver_dense_1', input=GAN.z_name)
    g.add_node(BatchNormalization(), 'gen_driver_bn_1',
               input='gen_driver_dense_1')
    g.add_node(Activation('relu'), 'gen_driver_act_1',
               input='gen_driver_bn_1')

    g.add_node(Dense(mask_input_dim), 'gen_driver_dense_2',
               input='gen_driver_act_1')
    g.add_node(BatchNormalization(), 'gen_driver_bn_2',
               input='gen_driver_dense_2')
    g.add_node(Layer(), 'driver', input='gen_driver_bn_2')
    # g.add_node(ZeroGradient(), 'gen_driver_zero_grad', input='driver')

    g.add_node(mask_generator, 'mask_generator', input='driver')
    g.add_node(offset_generator, 'gen_offset',
               input=GAN.z_name)
    g.add_node(PyramidBlending(mask_generator, input_pyramid_layers=3,
                               mask_pyramid_layers=2),
               'blending', input='gen_offset')
    reg_layer = Layer()
    act = ActivityInBoundsRegularizer(-1, 1)
    act.set_layer(reg_layer)
    reg_layer.regularizers = [act]
    g.add_node(reg_layer, GAN.generator_name, input='blending')

    real_shape = (nb_real,) + g.nodes[GAN.generator_name].output_shape[1:]
    g.add_input(GAN.real_name, batch_input_shape=real_shape)
    g.add_node(discriminator, "discriminator",
               inputs=[GAN.generator_name, "real"], concat_axis=0)
    add_gan_outputs(g, fake_for_gen=(0, nb_fake - nb_real),
                    fake_for_dis=(nb_fake - nb_real, nb_fake),
                    real=(nb_fake, nb_fake+nb_real))
    return g


def dcgan_discriminator(n=32, image_views=1, extra_dense_layer=False,
                        out_activation='sigmoid'):
    model = Sequential()
    model.add(TheanoConvolution2D(n, 5, 5, subsample=(2, 2),
                                  border_mode='full',
                                  input_shape=(1, 64, 64*image_views)))
    model.add(LeakyReLU(0.2))

    model.add(TheanoConvolution2D(2*n, 5, 5, subsample=(2, 2),
                                  border_mode='full'))
    model.add(batch_norm())
    model.add(LeakyReLU(0.2))

    model.add(TheanoConvolution2D(4*n, 5, 5, subsample=(2, 2),
                                  border_mode='full'))
    model.add(batch_norm())
    model.add(LeakyReLU(0.2))

    model.add(TheanoConvolution2D(8*n, 5, 5, subsample=(2, 2),
                                  border_mode='full'))
    model.add(batch_norm())
    model.add(LeakyReLU(0.2))

    model.add(Flatten())
    if extra_dense_layer:
        model.add(Dense(4*n))
        model.add(LeakyReLU(0.2))

    model.add(Dense(1, activation=out_activation))
    return model


def dcgan_generator_add_preprocess_network(g, input_dim=50):
    p = Sequential()
    p.add(Dense(g.layers[0].input_shape[1],
                activation='relu', input_dim=input_dim))
    g.trainable = False
    p.add(g)
    return p


def dcgan_seperated_generator(input_dim=40, model_generator=None,
                              variation_generator=None):
    if model_generator is None:
        model_generator = dcgan_generator(input_dim)
    if variation_generator is None:
        variation_generator = dcgan_generator(input_dim, include_last_layer=False)

    g = Graph()
    g.add_input('input', input_shape=(input_dim,))
    g.add_node(model_generator, 'model', input='input')
    g.add_node(variation_generator, 'variation', input='input')
    g.add_node(Deconvolution2D(1, 5, 5, subsample=(2, 2), border_mode=(2, 2)),
              'variation_deconv', input='variation')
    g.add_node(Activation('sigmoid'), 'variation_out', input='variation_deconv')
    g.add_output('output', inputs=['model', 'variation_out'], concat_axis=1)
    return g


def dcgan_variational_add_generator(nb_z=40, n=16):
    g = Graph()

    class Disconnected(Layer):
        def get_output(self, train=True):
            input = self.get_input(train)
            return theano.gradient.disconnected_grad(input)

    class Builder:
        def __init__(self, g):
            self.g = g
            self.outputs = ['input']
            self.next_input = ''
            self.join_cnt = 0
            self.deconv_filter = n*4
            self.vi = 0
            self.mi = 0

        def v(self):
            name = 'var_{}'.format(self.vi)
            self.vi += 1
            return name

        def m(self):
            name = 'model_{}'.format(self.mi)
            self.mi += 1
            return name

        def add_output(self, name):
            self.outputs.append(name)
            return name

        def get_input(self):
            if self.next_input:
                tmp = copy.copy(self.next_input)
                self.next_input = ''
                return tmp
            else:
                return self.outputs[-1]

        def set_input(self, name):
            self.next_input = name

        def project(self, name):
            dense_name = name + '_dense'
            self.g.add_node(Dense(8*n*4*4), dense_name,
                            input=self.get_input())
            reshape_name = name + '_reshape'
            self.g.add_node(Reshape((8*n, 4, 4,)), reshape_name,
                            input=dense_name)
            return self.add_output(reshape_name)

        def deconv(self, name, batch_norm=True, activation='relu'):
            deconv = name + "_deconv"
            batch = name + "_batch_norm"
            act = name + "_" + activation
            self.g.add_node(Deconvolution2D(
                self.deconv_filter, 5, 5, subsample=(2, 2),
                border_mode=(2, 2)),
                            deconv, input=self.get_input())
            if batch_norm:
                self.g.add_node(BatchNormalization(), batch, input=deconv)
                act_in = batch
            else:
                act_in = deconv
            self.g.add_node(Activation(activation), act, input=act_in)
            return self.add_output(act)

        def join(self, mo, vo):
            name = 'join_{}'.format(self.join_cnt)
            self.join_cnt += 1
            disconn = mo + '_disconn'
            self.g.add_node(Disconnected(), disconn, input=mo)
            self.g.add_node(Layer(), name, inputs=[disconn, vo],
                            concat_axis=1)
            return self.add_output(name)

    g.add_input('input', input_shape=(nb_z,))
    b = Builder(g)
    vo = b.project(b.v())
    b.set_input('input')
    mo = b.project(b.m())
    b.join(mo, vo)

    vo = b.deconv(b.v())
    b.set_input(mo)
    mo = b.deconv(b.m())

    b.deconv_filter //= 2

    b.join(mo, vo)
    vo = b.deconv(b.v())
    b.set_input(mo)
    mo = b.deconv(b.m())

    b.deconv_filter //= 2

    b.join(mo, vo)
    vo = b.deconv(b.v())
    b.set_input(mo)
    mo = b.deconv(b.m())

    b.deconv_filter = 1

    b.join(mo, vo)
    v_out = b.deconv(b.v(), batch_norm=False, activation='sigmoid')
    b.set_input(mo)
    m_out = b.deconv(b.m(), batch_norm=False, activation='sigmoid')

    g.add_output('output', inputs=[m_out, v_out], concat_axis=1)
    return g



def gan_with_z_rot90_grid_idx(generator, discriminator,
                              batch_size=128, nb_z=20, reconstruct_fn=None):
    nb_grid_params = nb_normalized_params()
    z_shape = (batch_size, nb_z)
    grid_shape = (1, TAG_SIZE, TAG_SIZE)
    grid_params_shape = (nb_grid_params, )

    g_graph = Graph()
    g_graph.add_input('z', input_shape=z_shape[1:])
    g_graph.add_input('grid_params', input_shape=grid_params_shape)
    g_graph.add_input('grid_idx', input_shape=grid_shape)
    g_graph.add_input('z_rot90', input_shape=(1, ))
    g_graph.add_node(generator, 'generator', inputs=['grid_params', 'z'])
    g_graph.add_output('output', input='generator')
    g_graph.add_output('z_rot90', input='z_rot90')
    g_graph.add_output('grid_idx', input='grid_idx')
    d_graph = asgraph(discriminator, input_name=GAN.d_input)
    return GAN(g_graph, d_graph, z_shape, reconstruct_fn=reconstruct_fn)


def add_diff_rot90(g_outmap):
    g_out = g_outmap["output"]
    grid_idx = g_outmap['grid_idx']
    z_rot90 = g_outmap['z_rot90']
    alphas = binary_mask(grid_idx, black=0.5, ignore=1.0,  white=0.5)
    bw_mask = binary_mask(grid_idx, black=0., ignore=0,  white=0.5)
    combined = (alphas * g_out) + bw_mask
    return rotate_by_multiple_of_90(combined, z_rot90)


def gan_add_bw_grid(generator, discriminator, batch_size=128, nb_z=20):
    return gan_with_z_rot90_grid_idx(
        generator, discriminator, batch_size=batch_size,
        nb_z=nb_z, reconstruct_fn=add_diff_rot90)


def dcgan_pyramid_generator(nb_dcgan_units=64, z_dim=50, nb_fake=64):
    nb_grid_params = nb_normalized_params()
    g_input_dim = nb_grid_params + z_dim
    g = Graph()
    g.add_input(GAN.z_name, batch_input_shape=(nb_fake, z_dim))
    g.add_input("gen_grid_params",
                batch_input_shape=(nb_fake, nb_grid_params))
    g.add_input("gen_grid_idx", batch_input_shape=(nb_fake, 1, 64, 64))
    tag_mean = Dense(2, activation='relu', init=normal(0.005))
    g.add_node(tag_mean, "tag_mean", inputs=[GAN.z_name, "gen_grid_params"])
    g.add_node(dcgan_generator(nb_dcgan_units, input_dim=g_input_dim),
               "dcgan_generator",
               inputs=[GAN.z_name, "gen_grid_params"])
    g.add_node(PyramidBlending(tag_mean), GAN.generator_name,
               inputs=["dcgan_generator", "gen_grid_idx"], concat_axis=1)
    return g


def mogan_learn_bw_grid(generator, discriminator, optimizer_fn,
                        batch_size=128, nb_z=20):
    variation_weight = 0.3

    def reconstruct(g_outmap):
        g_out = g_outmap["output"]
        grid_idx = g_outmap["grid_idx"]
        z_rot90 = g_outmap['z_rot90']
        alphas = binary_mask(grid_idx, black=variation_weight,
                             ignore=1.0, white=variation_weight)
        m = theano.gradient.disconnected_grad(g_out[:, :1])
        v = g_out[:, 1:]
        combined = v  # T.clip(m + alphas*v, 0., 1.)
        return rotate_by_multiple_of_90(combined, z_rot90)

    grid_loss_weight = theano.shared(np.cast[np.float32](1))

    def grid_loss(g_outmap):
        g_out = g_outmap['output']
        grid_idx = g_outmap["grid_idx"]
        m = g_out[:, :1]
        b = binary_mask(grid_idx, ignore=0, black=0,
                        white=1 - variation_weight)
        return grid_loss_weight*mse(b, m)

    gan = gan_with_z_rot90_grid_idx(generator, discriminator,
                                    batch_size=batch_size, nb_z=nb_z,
                                    reconstruct_fn=reconstruct)
    # FIXME
    mogan = MOGAN(gan, grid_loss, optimizer_fn,
                  gan_regulizer=GAN.L2Regularizer())
    return mogan, grid_loss_weight


def dummy_dcgan_generator(n=32, input_dim=50, nb_output_channels=1,
                          include_last_layer=True):
    model = Sequential()
    model.add(Dense(64*64, input_dim=input_dim, activation='relu'))
    model.add(Reshape((1, 64, 64)))
    return model


def gan_grid_idx(generator, discriminator,
                 batch_size=128, nb_z=20, reconstruct_fn=None):
    nb_grid_params = nb_normalized_params()
    z_shape = (batch_size, nb_z)
    grid_params_shape = (nb_grid_params, )
    g_graph = Graph()
    g_graph.add_input('z', input_shape=z_shape[1:])
    g_graph.add_input('grid_params', input_shape=grid_params_shape)
    g_graph.add_node(generator, 'generator', inputs=['grid_params', 'z'])
    g_graph.add_output('output', input='generator')
    d_graph = asgraph(discriminator, input_name=GAN.d_input)
    return GAN(g_graph, d_graph, z_shape, reconstruct_fn=reconstruct_fn)


def mogan_pyramid(generator, discriminator, optimizer_fn,
                  batch_size=128, nb_z=20,
                  gan_objective=binary_crossentropy,
                  d_loss_grad_weight=0):
    def tag_loss(cond_true, g_out_dict):
        g_out = g_out_dict['output']
        grid_idx = cond_true
        return pyramid_loss(grid_idx, g_out).loss

    gan = gan_grid_idx(generator, discriminator, batch_size, nb_z)
    # FIXME
    mogan = MOGAN(gan, tag_loss, optimizer_fn,
                  gan_regulizer=GAN.L2Regularizer(),
                  gan_objective=gan_objective,
                  d_loss_grad_weight=d_loss_grad_weight)
    return mogan
