#! /usr/bin/env python
# Copyright 2016 Leon Sixt
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

import matplotlib
matplotlib.use('Agg')  # noqa

import argparse
import os
import time
import sys

import numpy as np
import json
from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers.core import Dense

from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.normalization import BatchNormalization
from deepdecoder.networks import dcgan_generator, dcgan_discriminator, \
    dcgan_generator_add_preprocess_network
from deepdecoder.train import train_dcgan
from deepdecoder.data import real_generator, z_generator, zip_real_z, \
    nb_normalized_params, grids_lecture_generator
from deepdecoder.mask_loss import to_keras_loss, pyramid_loss
from beras.gan import sequential_to_gan
from beras.callbacks import VisualiseGAN, SaveModels


def train_g64_d64():
    nb_units = 64
    generator_input_dim = 100
    nb_real = 32
    nb_fake = 96
    lr = 0.0002
    beta_1 = 0.5
    nb_batches_per_epoch = 100
    nb_epoch = 1000
    output_dir = "models/dcgan_g64_d64_retry"
    hdf5_fname = "/home/leon/data/tags_plain_t6.hdf5"

    g = dcgan_generator(nb_units, generator_input_dim)
    d = dcgan_discriminator(nb_units // 2)

    gan = sequential_to_gan(g, d, nb_real, nb_fake)

    save = SaveModels({"generator.hdf5": g, "discriminator.hdf5": d},
                      output_dir=output_dir)
    visual = VisualiseGAN(nb_samples=13**2, output_dir=output_dir,
                          preprocess=lambda x: np.clip(x, 0, 1))

    real_z_gen = zip_real_z(real_generator(hdf5_fname, nb_real),
                            z_generator((nb_fake, generator_input_dim)))

    history = train_dcgan(gan, Adam(lr, beta_1), Adam(lr, beta_1), real_z_gen,
                          nb_batches_per_epoch=nb_batches_per_epoch,
                          nb_epoch=nb_epoch, callbacks=[save, visual])
    with open(os.path.join(output_dir, "history.json"), 'w+') as f:
        json.dump(history.history, f)


def train_g64_d64_dct():
    nb_units = 64
    generator_input_dim = 25
    nb_real = 64
    nb_fake = 96
    lr = 0.0002
    beta_1 = 0.5
    nb_batches_per_epoch = 100
    nb_epoch = 1000
    output_dir = "models/dcgan_g64_d64_dct"
    hdf5_fname = "/home/leon/data/tags_plain_t6.hdf5"

    g = dcgan_generator(nb_units, generator_input_dim)
    d = dcgan_discriminator(nb_units)

    gan = sequential_to_gan(g, d, nb_real, nb_fake)

    save = SaveModels({"generator.hdf5": g, "discriminator.hdf5": d},
                      output_dir=output_dir)
    visual = VisualiseGAN(nb_samples=13**2, output_dir=output_dir,
                          preprocess=lambda x: np.clip(x, -1, 1))

    real_z_gen = zip_real_z(real_generator(hdf5_fname, nb_real),
                            z_generator((nb_fake, generator_input_dim)))

    history = train_dcgan(gan, Adam(lr, beta_1), Adam(lr, beta_1), real_z_gen,
                          nb_batches_per_epoch=nb_batches_per_epoch,
                          nb_epoch=nb_epoch, callbacks=[save, visual])
    with open(os.path.join(output_dir, "history.json"), 'w+') as f:
        json.dump(history.history, f)


def train_g64_d64_fine_tune():
    nb_units = 64
    generator_input_dim = 100
    nb_real = 64
    nb_fake = 128 + nb_real
    lr = 0.00002
    beta_1 = 0.5
    nb_batches_per_epoch = 100
    nb_epoch = 60
    output_dir = "models/dcgan_g64_d64_fine_tune"
    hdf5_fname = "/home/leon/data/tags_plain_t6.hdf5"

    g = dcgan_generator(nb_units, generator_input_dim)
    d = dcgan_discriminator(nb_units)

    g.load_weights("models/dcgan_g64_d64/generator.hdf5")
    d.load_weights("models/dcgan_g64_d64/fix_discriminator.hdf5")

    gan = sequential_to_gan(g, d, nb_real, nb_fake,
                            nb_fake_for_gen=128,
                            nb_fake_for_dis=nb_real)

    save = SaveModels({"generator_{epoch:04d}.hdf5": g,
                       "discriminator_{epoch:04d}.hdf5": d},
                      every_epoch=20,
                      output_dir=output_dir)
    visual = VisualiseGAN(nb_samples=13**2, output_dir=output_dir,
                          preprocess=lambda x: np.clip(x, -1, 1))

    real_z_gen = zip_real_z(real_generator(hdf5_fname, nb_real, range=(-1, 1)),
                            z_generator((nb_fake, generator_input_dim)))

    history = train_dcgan(gan, Adam(lr, beta_1), Adam(lr, beta_1), real_z_gen,
                          nb_batches_per_epoch=nb_batches_per_epoch,
                          nb_epoch=nb_epoch, callbacks=[save, visual])

    with open(os.path.join(output_dir, "history.json"), 'w+') as f:
        json.dump(history.history, f)


def train_g64_d64_pyramid():
    nb_units = 64
    generator_input_dim = 100
    preprocess_input_dim = 100
    batch_size = 128
    nb_batches_per_epoch = 100
    nb_epoch = 200
    output_dir = "models/dcgan_g64_d64_pyramid"

    os.makedirs(output_dir, exist_ok=True)
    g = dcgan_generator(nb_units, generator_input_dim)
    g.load_weights("models/dcgan_g64_d64_fine_tune/generator_0060.hdf5")

    p = dcgan_generator_add_preprocess_network(
        g, input_dim=preprocess_input_dim)
    save = SaveModels({"pyramid_{epoch:04d}.hdf5": p},
                      every_epoch=20,
                      output_dir=output_dir)

    nb_z_param = preprocess_input_dim - nb_normalized_params()

    def generator():
        for z, (param, grid_idx) in zip(
                z_generator((batch_size, nb_z_param)),
                grids_lecture_generator(batch_size)):
            yield np.concatenate([param, z], axis=1), grid_idx

    print(next(generator())[0].shape)

    print("Compiling...")
    start = time.time()
    p.compile('adam', to_keras_loss(pyramid_loss))
    print("Done Compiling in {0:.2f}s".format(time.time() - start))

    history = p.fit_generator(generator(), nb_batches_per_epoch*batch_size,
                              nb_epoch,
                              verbose=1, callbacks=[save])

    with open(os.path.join(output_dir, "history.json"), 'w+') as f:
        json.dump(history.history, f)

    with open(os.path.join(output_dir, "network_config.json"), 'w+') as f:
        f.write(p.to_json())


def train_g64_d64_pyramid_preprocess_2layer():
    nb_units = 64
    generator_input_dim = 100
    preprocess_input_dim = 50
    preprocess_nb_hidden = 256
    batch_size = 128
    nb_batches_per_epoch = 100
    nb_epoch = 200
    output_dir = "models/train_g64_d64_pyramid_preprocess_2layer"

    os.makedirs(output_dir, exist_ok=True)
    g = dcgan_generator(nb_units, generator_input_dim)
    g.load_weights("models/dcgan_g64_d64_fine_tune/generator_0060.hdf5")

    p = Sequential()
    p.add(Dense(preprocess_nb_hidden,
                activation='relu', input_dim=preprocess_input_dim))
    p.add(Dense(g.layers[0].input_shape[1],
                activation='relu', input_dim=preprocess_input_dim))
    g.trainable = False
    p.add(g)

    save = SaveModels({"pyramid_{epoch:04d}.hdf5": p},
                      every_epoch=20,
                      output_dir=output_dir)

    nb_z_param = preprocess_input_dim - nb_normalized_params()

    def generator():
        for z, (param, grid_idx) in zip(
                z_generator((batch_size, nb_z_param)),
                grids_lecture_generator(batch_size)):
            yield np.concatenate([param, z], axis=1), grid_idx

    print(next(generator())[0].shape)

    print("Compiling...")
    start = time.time()
    p.compile('adam', to_keras_loss(pyramid_loss))
    print("Done Compiling in {0:.2f}s".format(time.time() - start))

    history = p.fit_generator(generator(), nb_batches_per_epoch*batch_size,
                              nb_epoch,
                              verbose=1, callbacks=[save])

    with open(os.path.join(output_dir, "history.json"), 'w+') as f:
        json.dump(history.history, f)

    with open(os.path.join(output_dir, "network_config.json"), 'w+') as f:
        f.write(p.to_json())


def main():
    functions = [train_g64_d64, train_g64_d64_fine_tune,
                 train_g64_d64_dct,
                 train_g64_d64_pyramid,
                 train_g64_d64_pyramid_preprocess_2layer]

    parser = argparse.ArgumentParser(
        description='Train a dcgan. Available functions: ' +
        ", ".join([f.__name__ for f in functions]))

    parser.add_argument('name', type=str, help='train function to run.')
    args = parser.parse_args()

    name = args.name
    for func in functions:
        if name == func.__name__:
            return func()

    print("No so function {}".format(name))
    sys.exit(1)

if __name__ == "__main__":
    main()
