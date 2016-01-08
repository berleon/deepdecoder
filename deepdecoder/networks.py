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


from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Flatten, Reshape, Activation
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.normalization import BatchNormalization
from beesgrid import NUM_MIDDLE_CELLS
from beras.layers.attention import RotationTransformer
from deepdecoder.deconv import Deconvolution2D
from deepdecoder.keras_fix import Convolution2D as TheanoConvolution2D


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
