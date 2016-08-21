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

from keras.models import Sequential
import keras.initializations
from keras.layers.core import Dense, Flatten, Reshape, Activation, \
    Layer, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D, \
    UpSampling2D, AveragePooling2D
from keras.layers.advanced_activations import LeakyReLU, ELU
from keras.layers.normalization import BatchNormalization
from keras.engine.topology import merge, Input
from keras.engine.training import Model
from keras.optimizers import Adam
from keras.regularizers import Regularizer
import keras.backend as K

from diktya.func_api_helpers import sequential, concat
from diktya.blocks import resnet, conv2d_block
from diktya.layers.core import Subtensor, InBounds
from beesgrid import NUM_MIDDLE_CELLS, NUM_CONFIGS
from deepdecoder.deconv import Deconvolution2D
from deepdecoder.transform import GaussianBlur
from collections import OrderedDict


def get_decoder_model(
        input,
        nb_units,
        nb_output=NUM_MIDDLE_CELLS + NUM_CONFIGS,
        depth=1,
        dense=[]):
    def dense_bn(n):
        return [
            Dense(n),
            batch_norm(mode=1),
            Activation('relu')
        ]

    def conv(n):
        return [
            [Convolution2D(n, 3, 3),
             batch_norm(mode=1),
             Activation('relu')
             ]
            for _ in range(depth)
        ]
    n = nb_units
    return sequential([
        conv(n),
        MaxPooling2D(),  # 32x32
        conv(2*n),
        MaxPooling2D(),  # 16x16
        conv(4*n),
        MaxPooling2D(),  # 8x8
        conv(8*n),
        MaxPooling2D(),  # 4x4
        conv(16*n),
        Flatten(),
        [dense_bn(d) for d in dense],
        Dense(nb_output)
    ])(input)


def batch_norm(mode=2, **kwargs):
    return BatchNormalization(mode=mode, axis=1, **kwargs)


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
    model.add(batch_norm())
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


def tag3d_network_dense(input, nb_units=64, nb_dense_units=[512, 512],
                        depth=2, nb_output_channels=1, trainable=True):
    n = nb_units

    def conv(n, repeats=None):
        if repeats is None:
            repeats = depth
        return [
            [
                Convolution2D(n, 3, 3, border_mode='same', init='he_normal'),
                Activation('relu')
            ] for _ in range(repeats)
        ]

    base = sequential([
        [
            Dense(nb_dense, activation='relu')
            for nb_dense in nb_dense_units
        ],
        Dense(8*n*4*4),
        Activation('relu'),
        Reshape((8*n, 4, 4,)),
        conv(8*n),
        UpSampling2D(),  # 8x8
        conv(4*n),
        UpSampling2D(),  # 16x16
        conv(2*n),
    ], ns='tag3d_gen.base', trainable=trainable)(input)

    tag3d = sequential([
        conv(2*n),
        UpSampling2D(),  # 32x32
        conv(n),
        UpSampling2D(),  # 64x64
        conv(n, 1),
        Convolution2D(1, 3, 3, border_mode='same', init='he_normal'),
    ], ns='tag3d_gen.tag3d', trainable=trainable)(base)

    depth_map = sequential([
        conv(n // 2, depth - 1),
        Convolution2D(1, 3, 3, border_mode='same', init='he_normal'),
    ], ns='tag3d_gen.depth_map', trainable=trainable)(base)

    return tag3d, depth_map


def tag_3d_network_conv(input, nb_inputs, nb_units=64, depth=2, filter_size=3):
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


def constant_init(value):
    def wrapper(shape, name=None):
        return K.variable(value*np.ones(shape), name=name)
    return wrapper


def get_label_generator(x, nb_units, nb_output_units):
    n = nb_units
    driver = sequential([
        Dense(n),
        batch_norm(),
        Dropout(0.25),
        Activation('relu'),
        Dense(n),
        batch_norm(),
        Dropout(0.25),
        Activation('relu'),
        Dense(nb_output_units),
        batch_norm(gamma_init=constant_init(0.25)),
        InBounds(-1, 1),
    ], ns='driver')
    return driver(x)


def get_offset_front(inputs, nb_units):
    n = nb_units
    input = concat(inputs)

    return sequential([
        Dense(8*n*4*4),
        batch_norm(),
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
        InBounds(-1, 1),
    ], ns='offset.back')(input)

    return back_feature_map, sequential([
        Convolution2D(1, 3, 3, border_mode='same'),
        InBounds(-1, 1),
    ], ns='offset.back_out')(back_feature_map)


def get_blur_factor(inputs, min=0, max=2):
    input = concat(inputs)
    return sequential([
        Convolution2D(1, 3, 3),
        Flatten(),
        Dense(1),
        InBounds(min, max),
    ], ns='mask_weight_blending')(input)


def get_lighting_generator(inputs, nb_units):
    n = nb_units
    input = concat(inputs)
    light_conv = sequential([
        conv(n, 3, 3),   # 16x16
        MaxPooling2D(),  # 8x8
        conv(n, 3, 3),
        conv(n, 3, 3),
        UpSampling2D(),  # 16x16
        conv(n, 3, 3),
        UpSampling2D(),  # 32x32
        conv(n, 3, 3),
        Convolution2D(3, 1, 1, border_mode='same'),
        UpSampling2D(),  # 64x64
        GaussianBlur(sigma=3.5),
    ], ns='lighting')(input)

    shift = Subtensor(0, 1, axis=1)(light_conv)
    shift = InBounds(-1, 1)(shift)
    in_bounds = InBounds(0, 2)

    scale_black = in_bounds(Subtensor(1, 2, axis=1)(light_conv))
    scale_white = in_bounds(Subtensor(2, 3, axis=1)(light_conv))

    return [scale_black, scale_white, shift]


def get_details(inputs, nb_units):
    n = nb_units
    return sequential([
        conv(n, 3, 3),
        conv(n, 3, 3),
        conv(n, 3, 3),
        Convolution2D(1, 3, 3, border_mode='same', init='normal'),
        InBounds(-0.75, 0.75)
    ], ns='details')(concat(inputs))


def conv_block(nb_units, *args):
    assert not ("down" in args and "up" in args)
    layers = [
        Convolution2D(nb_units, 3, 3, border_mode='same'),
    ]
    if 'down' in args:
        layers.append(MaxPooling2D())
    if 'up' in args:
        layers.append(UpSampling2D())
    layers.append(batch_norm())
    layers.append(Activation('relu'))
    return layers


def get_preprocess(input, nb_units, nb_conv_layers=None, resize=None, ns=None):
    assert not (nb_conv_layers is None and resize is None)

    if nb_conv_layers is None:
        nb_conv_layers = len(resize)

    if resize is None:
        resize = [None] * nb_conv_layers

    if type(nb_units) == int:
        nb_units = [nb_units] * nb_conv_layers

    layers = []
    for i, (units, up_or_down) in enumerate(zip(nb_units, resize)):
        layers.extend(conv_block(units, up_or_down))
    return sequential(layers, ns=ns)(input)


def conv(nb_filter, h=3, w=3, depth=1, activation='relu'):
    if issubclass(type(activation), Layer):
        activation_layer = activation
    else:
        activation_layer = Activation(activation)

    return [
        [
            Convolution2D(nb_filter, h, w, border_mode='same'),
            batch_norm(),
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
    model.add(batch_norm())
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
    model.add(InBounds(-1, 1))
    return model


def dcgan_generator_conv(n=32, input_dim=50, nb_output_channels=1,
                         init=normal(0.02)):
    def conv(nb_filter, h, w):
        model.add(Convolution2D(nb_filter, h, w, border_mode='same',
                                init=init))
        model.add(batch_norm())
        model.add(Activation('relu'))

    def deconv(nb_filter, h, w):
        deconv_layer = Deconvolution2D(nb_filter, h, w, border_mode=(1, 1),
                                       init=init)
        model.add(deconv_layer)

        w = np.random.normal(0, 0.02, deconv_layer.W_shape).astype(np.float32)
        w *= np.random.uniform(0, 1, (1, w.shape[1], 1, 1))
        deconv_layer.W.set_value(w)
        model.add(batch_norm())
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
    model.add(InBounds(-1, 1))
    return model


def render_gan_discriminator(x, n=32, conv_repeat=1, dense=[],
                             out_activation='sigmoid'):
    def conv(n):
        layers = [
            Convolution2D(n, 3, 3, subsample=(2, 2), border_mode='same'),
            batch_norm(),
            LeakyReLU(0.2),
        ]

        return layers + [[
            Convolution2D(n, 3, 3, border_mode='same'),
            batch_norm(),
            LeakyReLU(0.2),
        ] for _ in range(conv_repeat-1)]

    def get_dense(nb):
        return [
            Dense(nb),
            batch_norm(),
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


def render_gan_discriminator_resnet(x, n=32, conv_repeat=1, dense=[], out_activation='sigmoid'):
    def get_dense(nb):
        return [
            Dense(nb),
            batch_norm(),
            LeakyReLU(0.2),
        ]

    return sequential([
        Convolution2D(n, 3, 3, border_mode='same'),
        resnet(n, activation=LeakyReLU(0.3)),
        Convolution2D(n, 3, 3, subsample=(2, 2), border_mode='same'),
        LeakyReLU(0.2),
        batch_norm(),
        resnet(2*n, activation=LeakyReLU(0.3)),
        resnet(2*n, activation=LeakyReLU(0.3)),
        resnet(2*n, activation=LeakyReLU(0.3)),
        resnet(2*n, activation=LeakyReLU(0.3)),
        resnet(2*n, activation=LeakyReLU(0.3)),
        Convolution2D(4*n, 3, 3, subsample=(2, 2), border_mode='same'),
        resnet(4*n, activation=LeakyReLU(0.3)),
        resnet(4*n, activation=LeakyReLU(0.3)),
        resnet(4*n, activation=LeakyReLU(0.3)),
        resnet(4*n, activation=LeakyReLU(0.3)),
        resnet(4*n, activation=LeakyReLU(0.3)),
        Convolution2D(4*n, 3, 3, subsample=(2, 2), border_mode='same'),
        resnet(4*n, activation=LeakyReLU(0.3)),
        resnet(4*n, activation=LeakyReLU(0.3)),
        resnet(4*n, activation=LeakyReLU(0.3)),
        resnet(4*n, activation=LeakyReLU(0.3)),
        resnet(4*n, activation=LeakyReLU(0.3)),
        Flatten(),
        [get_dense(nb) for nb in dense],
        Dense(1, activation=out_activation)
    ], ns='dis')(concat(x, axis=0, name='concat_fake_real'))


def decoder_resnet(label_sizes, nb_filter=16, data_shape=(1, 64, 64), nb_bits=12,
                   resnet_depth=(3, 4, 6, 3),
                   optimizer='adam'):
    def _bn_relu_conv(nb_filter, nb_row=3, nb_col=3, subsample=1):
        return sequential([
            BatchNormalization(mode=0, axis=1),
            ELU(),
            Convolution2D(nb_filter=nb_filter, nb_row=nb_row, nb_col=nb_col,
                          subsample=(subsample, subsample), init="he_normal", border_mode="same")
        ])

    def f(nb_filter, subsample=1):
        return sequential([
            _bn_relu_conv(nb_filter, subsample=subsample),
            _bn_relu_conv(nb_filter),
        ])

    input = Input(shape=data_shape)
    fitlers_by_depth = [nb_filter * 2**i for i in range(len(resnet_depth))]
    print("fitlers_by_depth", fitlers_by_depth)
    x = _bn_relu_conv(nb_filter, 3, 3, subsample=2)(input)
    for i, (n, d) in enumerate(zip(fitlers_by_depth, resnet_depth)):
        for di in range(d):
            if di == 0 and i != 0:
                shortcut = _bn_relu_conv(n, 1, 1, subsample=2)
                subsample = 2
            else:
                shortcut = lambda x: x
                subsample = 1
            x = merge([shortcut(x), f(n, subsample)(x)], mode='sum')

    x = Flatten()(x)
    ids = sequential([
        Dense(256),
        ELU(),
        BatchNormalization(mode=0, axis=1),
        Dropout(0.5),
    ])(x)

    outputs = OrderedDict()
    losses = OrderedDict()
    for i in range(nb_bits):
        name = 'bit_{}'.format(i)
        outputs[name] = Dense(1, activation='sigmoid', name=name)(ids)
        losses[name] = 'binary_crossentropy'

    params = sequential([
        Dense(256),
        ELU(),
        BatchNormalization(mode=0, axis=1),
        Dropout(0.5),
    ])(x)

    for name, output_size in label_sizes:
        outputs[name] = Dense(output_size, activation='tanh', name=name)(params)
        losses[name] = 'mse'

    def get_loss_weight(name):
        if name.startswith('bit'):
            return 1.
        else:
            return 0.1

    model = Model(input, list(outputs.values()))
    model.compile(optimizer, loss=list(losses.values()),
                  loss_weights={k: get_loss_weight(k) for k in losses.keys()})
    return model


def simple_gan_generator(nb_units, z, labels, depth_map,
                         tag3d, depth=2):
    n = nb_units
    depth_map_features = sequential([
        conv2d_block(n),
        conv2d_block(2*n),
    ])(depth_map)

    tag3d_features = sequential([
        conv2d_block(n, subsample=2),
        conv2d_block(2*n, subsample=2),
    ])(tag3d)

    x = sequential([
        Dense(5*n),
        BatchNormalization(mode=2),
        Activation('relu'),
        Dense(5*n),
        BatchNormalization(mode=2),
        Activation('relu'),
    ])(concat([z, labels]))

    blur = InBounds(0, 1, clip=True)(Dense(1)(x))

    x = sequential([
        Dense(8*4*4*n),
        Activation('relu'),
        BatchNormalization(mode=2),
        Reshape((8*n, 4, 4)),
    ])(x)

    x = sequential([
        conv2d_block(8*n, filters=1, depth=1, up=True),  # 4x4 -> 8x8
        conv2d_block(8*n, depth=depth, up=True),  # 8x8 -> 16x16
    ])(x)

    off_depth_map = sequential([
        conv2d_block(2*n, depth=depth),
    ])(concat([x, depth_map_features]))

    light = sequential([
        conv2d_block(2*n, depth=depth, up=True),  # 16x16 -> 32x32
        conv2d_block(n, depth=depth, up=True),  # 32x32 -> 64x64
    ])(off_depth_map)

    def get_light(x):
        return sequential([
            conv2d_block(1, filters=1, batchnorm=False),
            GaussianBlur(sigma=4),
            InBounds(0, 1, clip=True),
        ])(x)

    light_sb = get_light(light)
    light_sw = get_light(light)
    light_t = get_light(light)

    background = sequential([
        conv2d_block(2*n, depth=depth, up=True),  # 16x16 -> 32x32
        conv2d_block(n, depth=depth, up=True),  # 32x32 ->  64x64
        conv2d_block(1, batchnorm=False),
        InBounds(-1, 1, clip=True),
    ])(off_depth_map)

    details = sequential([
        conv2d_block(2*n, depth=depth, up=True),  # 16x16 -> 32x32
        conv2d_block(n, depth=depth, up=True),  # 32x32 ->  64x64
        conv2d_block(1, depth=1, batchnorm=False),
        InBounds(-1, 1, clip=True)
    ])(concat(tag3d_features, off_depth_map))
    return blur, [light_sb, light_sw, light_t], background, details
