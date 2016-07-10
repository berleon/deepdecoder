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

from keras.models import Sequential, Graph
import keras.initializations
from keras.layers.core import Dense, Flatten, Reshape, Activation, \
    Layer, Merge, Dropout, Lambda
from keras.layers.convolutional import Convolution2D, MaxPooling2D, \
    UpSampling2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.normalization import BatchNormalization
from keras.regularizers import Regularizer
from keras.engine.topology import merge
import keras.backend as K

from beras.layers.attention import RotationTransformer
from beras.util import sequential, concat, collect_layers, \
    namespace, name_tensor
from beras.regularizers import with_regularizer
from beras.layers.core import Split, ZeroGradient, LinearInBounds

from beesgrid import NUM_MIDDLE_CELLS, NUM_CONFIGS

from deepdecoder.deconv import Deconvolution2D
from deepdecoder.keras_fix import Convolution2D as TheanoConvolution2D
from deepdecoder.transform import PyramidBlending, PyramidReduce, \
    AddLighting, Selection, GaussianBlur, HighPass


def get_decoder_model(
        input,
        nb_units,
        nb_output=NUM_MIDDLE_CELLS + NUM_CONFIGS,
        depth=1,
        dense=[]):
    def dense_bn(n):
        return [
            Dense(n),
            BatchNormalization(),
            Activation('relu')
        ]
    n = nb_units
    d = depth
    return sequential([
        conv(n, depth=d),
        MaxPooling2D(),  # 32x32
        conv(2*n, depth=d),
        MaxPooling2D(),  # 16x16
        conv(4*n, depth=d),
        MaxPooling2D(),  # 8x8
        conv(8*n, depth=d),
        MaxPooling2D(),  # 4x4
        conv(16*n, depth=d),
        Flatten(),
        [dense_bn(d) for d in dense],
        Dense(nb_output)
    ])(input)


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


class MinCoveredRegularizer(Regularizer):
    def __init__(self, mask_size=32, min_covered=1/8, max_loss=4):
        self.mask_size = mask_size
        self.min_covered = min_covered
        self.max_loss = max_loss
        self.uses_learning_phase = True

    def set_layer(self, layer):
        self.layer = layer

    def __call__(self, loss):
        if not hasattr(self, 'layer'):
            raise Exception('Need to call `set_layer` on '
                            'MaskRegularizer instance '
                            'before calling the instance.')
        min_tag_size = self.mask_size**2 * self.min_covered
        factor = min_tag_size / self.max_loss
        out = self.layer.output
        out_sum = out.sum(axis=(1, 2, 3))
        reg_loss = K.switch(out_sum <= min_tag_size,
                            factor*(out_sum - min_tag_size)**2, 0)
        return K.in_train_phase(loss + reg_loss.mean(), loss)


def tag_3d_model_network_dense(input, nb_units=64, dense_factor=3, nb_dense_layers=2,
                               depth=2, nb_output_channels=1, trainable=True):
    n = nb_units

    def conv(n, repeats=1):
        return [
            [
                Convolution2D(n, 3, 3, border_mode='same', init='he_normal'),
                Activation('relu')
            ] for _ in range(repeats)
        ]

    dense_layers = [
        Dense(dense_factor*nb_units, activation='relu')
        for _ in range(nb_dense_layers)]
    base = sequential(dense_layers + [
        Dense(8*n*4*4),
        Activation('relu'),
        Reshape((8*n, 4, 4,)),
        conv(8*n),
        UpSampling2D(),  # 8x8
        conv(4*n, depth),
        UpSampling2D(),  # 16x16
        conv(2*n),
    ], ns='mask_gen.base', trainable=trainable)(input)

    mask = sequential([
        conv(2*n, depth),
        UpSampling2D(),  # 32x32
        conv(n, 2),
        UpSampling2D(),  # 64x64
        conv(n, 1),
        Convolution2D(1, 3, 3, border_mode='same', init='he_normal'),
    ], ns='mask_gen.mask', trainable=trainable)(base)

    depth_map = sequential([
        conv(n // 2, depth - 1),
        Convolution2D(1, 3, 3, border_mode='same', init='he_normal'),
    ], ns='mask_gen.depth_map', trainable=trainable)(base)

    return mask, depth_map


def tag_3d_network(input, nb_inputs, nb_units=64, depth=2, filter_size=3):
    n = nb_units

    def conv(n, repeats=1, f=None):
        if f is None:
            f = filter_size
        return [
            [
                Convolution2D(n, f, f, border_mode='same', init='he_normal'),
                Activation('relu')
            ] for _ in range(repeats)
        ]

    base = sequential([
        Reshape((nb_inputs, 1, 1,)),
        conv(8*n, depth, f=1),
        UpSampling2D(),  # 2x2
        conv(8*n, depth, f=2),
        UpSampling2D(),  # 4x4
        conv(8*n, depth),
        UpSampling2D(),  # 8x8
        conv(4*n, depth),
        UpSampling2D(),  # 16x16
        conv(2*n),
    ], ns='mask_gen.base')(input)

    mask = sequential([
        conv(2*n, depth),
        UpSampling2D(),  # 32x32
        conv(n, depth),
        UpSampling2D(),  # 64x64
        conv(n, depth - 1),
        Convolution2D(1, 3, 3, border_mode='same', init='he_normal'),
    ], ns='mask_gen.mask')(base)

    depth_map = sequential([
        conv(n // 2, depth - 1),
        Convolution2D(1, 3, 3, border_mode='same', init='he_normal'),
    ], ns='mask_gen.depth_map')(base)

    return mask, depth_map


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


def constant_init(value):
    def wrapper(shape, name=None):
        return K.variable(value*np.ones(shape), name=name)
    return wrapper


def get_mask_driver(x, nb_units, nb_output_units):
    n = nb_units
    driver = sequential([
        Dense(n),
        BatchNormalization(),
        Dropout(0.25),
        Activation('relu'),
        Dense(n),
        BatchNormalization(),
        Dropout(0.25),
        Activation('relu'),
        Dense(nb_output_units),
        BatchNormalization(gamma_init=constant_init(0.25)),
        LinearInBounds(-1, 1, clip=True),
    ], ns='driver')
    return driver(x)


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
        conv(4*n, 3, 3),
        UpSampling2D(),  # 16x16
        conv(2*n, 3, 3),
        conv(2*n, 3, 3),
    ], ns='offset.front')(input)


def get_offset_middle(inputs, nb_units):
    n = nb_units
    input = concat(inputs)
    return sequential([
        UpSampling2D(),  # 32x32
        conv(2*n, 3, 3),
        conv(2*n, 3, 3),
        conv(2*n, 3, 3),
    ], ns='offset.middle')(input)


def get_offset_back(inputs, nb_units):
    n = nb_units
    input = concat(inputs)
    back_feature_map = sequential([
        UpSampling2D(),  # 64x64
        conv(n, 3, 3),
        conv(n, 3, 3),
    ], ns='offset.back')(input)

    return back_feature_map, sequential([
        Convolution2D(1, 3, 3, border_mode='same'),
        LinearInBounds(-1, 1, clip=True),
    ], ns='offset.back_out')(back_feature_map)


def get_mask_weight_blending(inputs, min=0, max=2):
    input = concat(inputs)
    return sequential([
        Convolution2D(1, 3, 3),
        Flatten(),
        Dense(1),
        LinearInBounds(min, max, clip=True),
    ], ns='mask_weight_blending')(input)


def get_lighting_generator(inputs, nb_units):
    n = nb_units
    input = concat(inputs)
    light_conv = sequential([
        conv(n, 5, 5),   # 16x16
        MaxPooling2D(),  # 8x8
        conv(n, 3, 3),
        conv(n, 3, 3),
        UpSampling2D(),  # 16x16
        conv(n, 5, 5),
        UpSampling2D(),  # 32x32
        conv(n, 5, 5),
        Convolution2D(3, 1, 1, border_mode='same'),
        UpSampling2D(),  # 64x64
        LinearInBounds(-1, 1, clip=True),
        GaussianBlur(sigma=3.5),
    ], ns='lighting')(input)

    shift = Split(0, 1, axis=1)(light_conv)
    scale_black = Split(1, 2, axis=1)(light_conv)
    scale_white = Split(2, 3, axis=1)(light_conv)

    return shift, scale_black, scale_white


def get_mask_postprocess(inputs, nb_units):
    n = nb_units
    return sequential([
        conv(n, 3, 3),
        conv(n, 3, 3),
        Convolution2D(1, 5, 5, border_mode='same', init='normal'),
    ], ns='mask_post')(concat(inputs))


def get_offset_merge_mask(input, nb_units, nb_conv_layers, poolings=None, ns=None):
    def conv_layers(units, pooling):
        layers = [Convolution2D(units, 3, 3, border_mode='same')]
        if pooling:
            layers.append(MaxPooling2D())
        layers.extend([
            BatchNormalization(axis=1),
            Activation('relu'),
        ])
        return layers

    if poolings is None:
        poolings = [False] * nb_conv_layers
    if type(nb_units) == int:
        nb_units = [nb_units] * nb_conv_layers
    layers = []
    for i, (units, pooling) in enumerate(zip(nb_units, poolings)):
        layers.extend(conv_layers(units, pooling))
    return sequential(layers, ns=ns)(input)


def get_bits(z):
    return Lambda(lambda x: 2*K.cast(x > 0, K.floatx()) - 1)(z)


class NormSinCosAngle(Layer):
    def __init__(self, idx, **kwargs):
        self.sin_idx = idx
        self.cos_idx = idx + 1
        super().__init__(**kwargs)

    def call(self, x, mask=None):
        sin = x[:, self.sin_idx:self.sin_idx+1]
        cos = x[:, self.cos_idx:self.cos_idx+1]
        eps = 1e-7
        scale = K.sqrt(1./(eps + sin**2 + cos**2))
        sin_scaled = K.clip(scale*sin, -1, 1)
        cos_scaled = K.clip(scale*cos, -1, 1)
        return K.concatenate([x[:, :self.sin_idx], sin_scaled, cos_scaled,
                              x[:, self.cos_idx+1:]], axis=1)

    def get_config(self):
        config = {'idx': self.sin_idx}
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))


def mask_blending_generator(
        mask_driver,
        tag_3d_model_network,
        light_merge_mask16,
        offset_merge_mask16,
        offset_merge_mask32,
        offset_merge_light16,
        lighting_generator,
        offset_front,
        offset_middle,
        offset_back,
        mask_weight_blending32,
        mask_weight_blending64,
        mask_postprocess,
        z_for_bits,
        z_for_driver,
        z_for_offset
        ):

    def generator(inputs):
        z, = inputs
        z_driver = Split(*z_for_driver, axis=1)(z)
        z_offset = Split(*z_for_offset, axis=1)(z)
        z_bits = Split(*z_for_bits, axis=1)(z)
        bits = get_bits(z_bits)
        driver = mask_driver(z_driver)
        driver_norm = NormSinCosAngle(0, name='driver_norm')(driver)
        mask_input = concat([bits, driver_norm], name='mask_gen_input')

        mask, mask_depth_map = tag_3d_model_network(mask_input)

        mask = name_tensor(mask, 'mask')
        mask_depth_map = name_tensor(mask_depth_map, 'mask_depth_map')

        selection = with_regularizer(Selection(threshold=-0.08,
                                               smooth_threshold=0.2,
                                               sigma=1.5, name='selection'),
                                     MinCoveredRegularizer())

        mask_down = PyramidReduce()(mask)
        mask_selection = selection(mask)

        out_offset_front = offset_front([z_offset,
                                         ZeroGradient()(driver_norm)])

        light_outs = list(lighting_generator(
            [out_offset_front, light_merge_mask16(mask_depth_map)]))

        mask_with_lighting = AddLighting(
            scale_factor=0.6, shift_factor=0.75,
            name='mask_with_lighting')([mask] + light_outs)

        out_offset_middle = offset_middle(
            [out_offset_front, offset_merge_mask16(mask_depth_map),
             offset_merge_light16(concat(light_outs))])

        offset_back_feature_map, out_offset_back = offset_back(
            [out_offset_middle, offset_merge_mask32(mask_down)])

        mask_weight64 = mask_weight_blending64(out_offset_middle)
        blending = PyramidBlending(offset_pyramid_layers=2,
                                   mask_pyramid_layers=2,
                                   mask_weights=['variable', 1],
                                   offset_weights=[1, 1],
                                   use_selection=[True, True],
                                   name='blending')(
                [out_offset_back, mask_with_lighting, mask_selection,
                 mask_weight64
                 ])

        mask_post = mask_postprocess(
            [blending, mask_selection, mask, out_offset_back,
             offset_back_feature_map] + light_outs)
        mask_post = name_tensor(mask_post, 'mask_post')
        mask_post_high = HighPass(4, nb_steps=4,
                                         name='mask_post_high')(mask_post)
        blending_post = merge([mask_post_high, blending], mode='sum',
                              name='blending_post')
        return LinearInBounds(-1.2, 1.2, name='generator')(blending_post)

    return generator


def conv(nb_filter, h=3, w=3, depth=1, activation='relu'):
    if issubclass(type(activation), Layer):
        activation_layer = activation
    else:
        activation_layer = Activation(activation)

    return [
        [
            Convolution2D(nb_filter, h, w, border_mode='same'),
            BatchNormalization(axis=1),
            activation_layer
        ] for _ in range(depth)
    ]


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


def mask_blending_discriminator(x, n=32, conv_repeat=1,
                                dense=[],
                                out_activation='sigmoid'):
    def conv(n):
        layers = [
            Convolution2D(n, 3, 3, subsample=(2, 2), border_mode='same'),
            BatchNormalization(axis=1),
            LeakyReLU(0.2),
        ]

        return layers + [[
            Convolution2D(n, 3, 3, border_mode='same'),
            BatchNormalization(axis=1),
            LeakyReLU(0.2),
        ] for _ in range(conv_repeat-1)]

    def get_dense(nb):
        return [
            Dense(nb),
            BatchNormalization(axis=1),
            LeakyReLU(0.2),
        ]

    return sequential([
        Convolution2D(n, 5, 5, subsample=(2, 2), border_mode='same'),
        LeakyReLU(0.2),
        conv(2*n),
        conv(4*n),
        conv(8*n),
        Flatten(),
        [get_dense(nb) for nb in dense],
        Dense(1, activation=out_activation)
    ], ns='dis')(concat(x, axis=0, name='concat_fake_real'))


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
