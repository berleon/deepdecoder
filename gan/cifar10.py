#! /usr/bin/env python3
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
from beras.gan import GenerativeAdverserial
from beras.util import Sample, LossPrinter
from keras.callbacks import ModelCheckpoint

from keras.models import Sequential, Graph
from keras.datasets import cifar10

from keras.layers.core import Dense, Activation, Reshape, Dropout, Flatten, \
    MaxoutDense
from keras.optimizers import Adam, SGD
from keras.preprocessing.image import ImageDataGenerator
import scipy
from scipy.misc import imsave
import numpy as np


noiseDim = 100
data_dir = "cifar10_data"

def test_sample(mnist_gan):
    mnist_gan.load_weights(data_dir + '/{}_test.hdf5')
    nb_samples = 64
    mnist_sample = mnist_gan.generate(np.random.uniform(-1, 1, (nb_samples, nb_z)))
    out_dir = data_dir + "/epoche_{}/".format(1)
    os.makedirs(out_dir, exist_ok=True)
    for i in range(nb_samples):
        imsave(os.path.join(out_dir, str(i) + ".png"),
               (mnist_sample[i, :].reshape(28, 28)*255).astype(np.uint8))


def build_cifar10_gan():
    geometry = (1, 3, 16, 16)
    input_sz = geometry[1] * geometry[2] * geometry[3]
    numhid = 8000

    generator = Sequential()
    generator.add(Dense(noiseDim, numhid, activation='relu'))
    generator.add(Dense(numhid, numhid, activation='relu'))
    generator.add(Dense(numhid, input_sz, activation='sigmoid'))

    numhid = 1600
    discriminator = Sequential()
    discriminator.add(Dense(input_sz, numhid, activation='relu'))
    discriminator.add(Dropout(.5))
    discriminator.add(Dense(numhid, numhid, activation='relu'))
    discriminator.add(Dropout(.5))
    discriminator.add(Dense(numhid, 1, activation='sigmoid'))
    return GenerativeAdverserial(generator, discriminator)


def main():
    mnist_gan = build_cifar10_gan()
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()
    X_train = X_train.reshape(-1, 3, 32, 32) / 255.
    X_train = X_train[:, :, ::2, ::2]
    print(X_train.shape)
    X_train = X_train.reshape(len(X_train), 3*16*16)
    sgd = SGD(lr=0.02, decay=1e-6)
    mnist_gan.compile(sgd, sgd)
    mnist_gan.fit(X_train, z_shape=(X_train.shape[0], noiseDim), nb_epoch=150,
                  batch_size=100, verbose=0,
                  callbacks=[Sample(data_dir + "/samples", (30, noiseDim), every_nth_epoch=1),
                             LossPrinter(), ModelCheckpoint("cifar10_models_{}.hdf5")])


if __name__ == "__main__":
    main()
