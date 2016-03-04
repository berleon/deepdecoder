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


from keras.models import Sequential, Graph
from keras.objectives import mse, binary_crossentropy
from keras.objectives import mse
from keras.layers.core import Dense, Dropout, Flatten, Reshape, Activation, \
    Layer
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.normalization import BatchNormalization
from beesgrid import NUM_MIDDLE_CELLS, TAG_SIZE
from beras.layers.attention import RotationTransformer
from beras.gan import GAN
from beras.models import asgraph
from deepdecoder.deconv import Deconvolution2D
from deepdecoder.keras_fix import Convolution2D as TheanoConvolution2D
from deepdecoder.utils import binary_mask, rotate_by_multiple_of_90
from deepdecoder.mogan import MOGAN
from deepdecoder.data import num_normalized_params
import theano
import numpy as np
import copy


def get_decoder_model():
    size = 64
    rot_model = Sequential()
    rot_model.add()
    rot_model.add(Convolution2D(8, 5, 5, activation='relu',
                                input_shape=(1, size, size)))
    rot_model.add(MaxPooling2D((3, 3)))
    rot_model.add(Dropout(0.5))
    rot_model.add(Convolution2D(8, 5, 5, activation='relu'))
    rot_model.add(MaxPooling2D((3, 3)))
    rot_model.add(Dropout(0.5))

    rot_model.add(Flatten())
    rot_model.add(Dense(256, activation='relu'))
    rot_model.add(Dropout(0.5))
    rot_model.add(Dense(1, activation='relu'))

    model = Sequential()
    model.add(RotationTransformer(rot_model, input_shape=(1, size, size)))
    model.add(Convolution2D(16, 3, 3, border_mode='valid',
                            activation='relu', input_shape=(1, size, size)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))
    model.add(Convolution2D(32, 2, 2, border_mode='valid',
                            activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))
    model.add(Convolution2D(48, 2, 2, border_mode='valid',
                            activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))
    model.add(Convolution2D(48, 2, 2, border_mode='valid',
                            activation='relu'))
    model.add(Dropout(0.5))

    model.add(Convolution2D(48, 2, 2, border_mode='valid',
                            activation='relu'))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(1000, activation='relu'))
    model.add(Dense(1000, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(512, activation='relu'))
    model.add(Dense(NUM_MIDDLE_CELLS, activation='sigmoid'))
    return model


def batch_norm():
    mode = 0
    bn = BatchNormalization(mode, axis=1)
    bn.init = keras.initializations.one
    return bn


def normal(scale=0.02):
    def normal_wrapper(shape, name=None):
        return keras.initializations.normal(shape, scale, name)
    return normal_wrapper


def dcgan_generator(n=32, input_dim=50, nb_output_channels=1,
                    include_last_layer=True, init=normal(0.02)):

    def deconv(nb_filter, h, w):
        return Deconvolution2D(nb_filter, h, w, subsample=(2, 2),
                               border_mode=(2, 2), init=init)
    model = Sequential()
    model.add(Dense(8*n*4*4, input_dim=input_dim, init=init))
    model.add(batch_norm())
    model.add(Activation('relu'))
    model.add(Reshape((8*n, 4, 4,)))

    model.add(deconv(4*n, 5, 5))
    model.add(batch_norm())
    model.add(Activation('relu'))

    model.add(deconv(2*n, 5, 5))
    model.add(batch_norm())
    model.add(Activation('relu'))

    model.add(deconv(n, 5, 5))
    model.add(batch_norm())
    model.add(Activation('relu'))

    if include_last_layer:
        model.add(Deconvolution2D(nb_output_channels, 5, 5, subsample=(2, 2),
                                  border_mode=(2, 2), init=init))
        # model.add(BatchNormalization())
        model.add(Activation('linear'))
    return model


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
    nb_grid_params = num_normalized_params()
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
    mogan = MOGAN(gan, grid_loss, optimizer_fn,
                  gan_regulizer=GAN.L2Regularizer())
    return mogan, grid_loss_weight


def dummy_dcgan_generator(n=32, input_dim=50, nb_output_channels=1,
                          include_last_layer=True):
    model = Sequential()
    model.add(Dense(64*64, input_dim=input_dim, activation='relu'))
    model.add(Reshape((1, 64, 64)))
    return model
