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
from keras.datasets import mnist

from keras.layers.core import Dense, Activation, Reshape, Dropout, Flatten, \
    MaxoutDense
from keras.optimizers import Adam, SGD
from scipy.misc import imsave
import numpy as np


nb_z = 1200


def test_sample(mnist_gan):
    mnist_gan.load_weights('test_data/{}_test.hdf5')
    nb_samples = 64
    mnist_sample = mnist_gan.generate(np.random.uniform(-1, 1, (nb_samples, nb_z)))
    out_dir = "test_data/epoche_{}/".format(1)
    os.makedirs(out_dir, exist_ok=True)
    for i in range(nb_samples):
        imsave(os.path.join(out_dir, str(i) + ".png"),
               (mnist_sample[i, :].reshape(28, 28)*255).astype(np.uint8))


def build_mnist_gan():
    generator = Sequential()
    generator.add(Dense(nb_z, nb_z, activation='relu', init='normal', name='d_dense1'))
    generator.add(Dropout(0.5))
    generator.add(Dense(nb_z, nb_z, activation='relu', init='normal', name='d_dense2'))
    generator.add(Dropout(0.5))
    generator.add(Dense(nb_z, 784, activation='sigmoid', init='normal', name='d_dense3'))

    discriminator = Sequential()
    discriminator.add(MaxoutDense(784, 240, nb_feature=5, init='normal'))
    discriminator.add(Dropout(0.5))
    discriminator.add(MaxoutDense(240, 240, nb_feature=5, init='normal'))
    discriminator.add(Dropout(0.5))
    discriminator.add(Dense(240, 1, activation='sigmoid', init='normal', name='d_dense3'))
    return GenerativeAdverserial(generator, discriminator)


def main():
    mnist_gan = build_mnist_gan()
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    imsave("mnist0.png", X_train[0])
    X_train = X_train.reshape(-1, 784) / 255.
    sgd = SGD(lr=0.005, decay=1e-6, momentum=0.8, nesterov=True)
    mnist_gan.compile(sgd)
    # mnist_gan.print_svg()
    mnist_gan.fit(X_train, z_shape=(X_train.shape[0], nb_z), nb_epoch=150,
                  batch_size=100, verbose=0,
                  callbacks=[Sample("mnist_samples", (60, nb_z)),
                             LossPrinter(), ModelCheckpoint("models_{}.hdf5")])


if __name__ == "__main__":
    main()
