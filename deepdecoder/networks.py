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
from keras.layers.core import Dense, Dropout, Flatten, Reshape, Activation
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.normalization import BatchNormalization
from beesgrid import NUM_MIDDLE_CELLS, TAG_SIZE, NUM_CONFIGS, CONFIG_ROTS
from beras.layers.attention import RotationTransformer
from beras.gan import GAN
from beras.models import asgraph
from deepdecoder.deconv import Deconvolution2D
from deepdecoder.keras_fix import Convolution2D as TheanoConvolution2D
from deepdecoder.utils import binary_mask, rotate_by_multiple_of_90


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


def dcgan_generator(input_dim=50, nb_output_channels=1):
    n = 64
    model = Sequential()
    model.add(Dense(8*n*4*4, input_dim=input_dim))
    model.add(Reshape((8*n, 4, 4,)))

    model.add(Deconvolution2D(4*n, 5, 5, subsample=(2, 2), border_mode=(2, 2)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Deconvolution2D(2*n, 5, 5, subsample=(2, 2), border_mode=(2, 2)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Deconvolution2D(n, 5, 5, subsample=(2, 2), border_mode=(2, 2)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Deconvolution2D(nb_output_channels, 5, 5, subsample=(2, 2),
                              border_mode=(2, 2)))
    model.add(Activation('sigmoid'))
    return model


def dcgan_discriminator():
    n = 64
    model = Sequential()
    model.add(TheanoConvolution2D(n, 5, 5, subsample=(2, 2),
                                  border_mode='full', input_shape=(1, 64, 64)))
    model.add(LeakyReLU(0.2))

    model.add(TheanoConvolution2D(2*n, 5, 5, subsample=(2, 2),
                                  border_mode='full'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(0.2))

    model.add(TheanoConvolution2D(4*n, 5, 5, subsample=(2, 2),
                                  border_mode='full'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(0.2))

    model.add(TheanoConvolution2D(8*n, 5, 5, subsample=(2, 2),
                                  border_mode='full'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(0.2))

    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))
    return model


NB_GAN_GRID_PARAMS = NUM_CONFIGS + NUM_MIDDLE_CELLS + len(CONFIG_ROTS)


def diff_gan(generator, discriminator, batch_size=128, nb_z=20):
    nb_grid_params = NB_GAN_GRID_PARAMS
    z_shape = (batch_size, nb_z)
    grid_shape = (1, TAG_SIZE, TAG_SIZE)
    grid_params_shape = (nb_grid_params, )

    g_graph = Graph()
    g_graph.add_input('z', input_shape=z_shape[1:])
    g_graph.add_input('grid_params', input_shape=grid_params_shape)
    g_graph.add_input('grid_idx', input_shape=grid_shape)
    g_graph.add_input('z_rot90', input_shape=(1, ))
    g_graph.add_node(generator, 'generator', inputs=['z', 'grid_params'])
    g_graph.add_output('output', input='generator')
    g_graph.add_output('z_rot90', input='z_rot90')
    g_graph.add_output('grid_idx', input='grid_idx')
    d_graph = asgraph(discriminator, input_name=GAN.d_input)

    def add_diff(g_outmap):
        g_out = g_outmap["output"]
        grid_idx = g_outmap['grid_idx']
        z_rot90 = g_outmap['z_rot90']
        g_rotate = rotate_by_multiple_of_90(g_out, z_rot90)
        alphas = binary_mask(grid_idx, black=0.5, ignore=1.0,  white=0.5)
        bw_mask = binary_mask(grid_idx, black=0., ignore=0,  white=0.5)
        return (alphas * g_rotate) + bw_mask

    return GAN(g_graph, d_graph, z_shape, add_diff)
