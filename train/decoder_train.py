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
import os
import sys
from beras.layers.attention import RotationTransformer
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Dropout, Flatten, Dense, Activation
from keras.models import Sequential

import numpy as np

from mask_loss import mask_loss
#
import GeneratedGridTrainer
from GeneratedGridTrainer import NetworkArgparser

def get_decoder_model():
    size = 64
    rot_model = Sequential()
    rot_model.add(Convolution2D(1, 5, 5, activation='relu',
                                input_shape=(1, size, size)))
    rot_model.add(MaxPooling2D((5, 5)))
    rot_model.add(Dropout(0.5))
    rot_model.add(Flatten(input_shape=(1, 64, 64)))
    rot_model.add(Dense(256, activation='relu'))
    rot_model.add(Dropout(0.5))
    rot_model.add(Dense(1, activation='relu'))

    model = Sequential()
    model.add(RotationTransformer(rot_model, input_shape=(1, size, size)))
    model.add(Convolution2D(16, 3, 3, border_mode='valid',
                            activation='relu', input_shape=(1, size, size)))
    model.add(MaxPooling2D(pool_size=(3, 3)))
    model.add(Dropout(0.5))
    model.add(Convolution2D(32, 2, 2, border_mode='valid',
                            activation='relu'))
    model.add(MaxPooling2D(pool_size=(3, 3)))
    model.add(Dropout(0.5))
    model.add(Convolution2D(48, 2, 2, border_mode='valid',
                            activation='relu'))
    model.add(Flatten())
    model.add(Dense(500))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(12, activation='sigmoid'))
    return model, rot_model


decoder, rot_model = get_decoder_model()
print("Compiling...")
decoder.compile("adam", "mse")
rot_model.compile("adam", "mse")
print("Done")
n = 128 * 20
X = np.random.sample((n, 1, 64, 64)).astype(np.float32)
y = np.random.sample((n, 12)).astype(np.float32)
y1 = np.random.sample((n, 1)).astype(np.float32)
rot_model.fit(X, y1, nb_epoch=2)
decoder.fit(X, y, nb_epoch=2)
print("Predict worked")
