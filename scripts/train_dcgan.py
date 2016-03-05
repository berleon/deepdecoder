#! /usr/bin/env python
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

import matplotlib
matplotlib.use('Agg')  # noqa

import argparse
import os
import numpy as np
import json
from keras.optimizers import Adam

from deepdecoder.networks import dcgan_generator, dcgan_discriminator
from deepdecoder.train import train_dcgan
from deepdecoder.data import real_generator, z_generator, zip_real_z
from beras.gan import sequential_to_gan
from beras.callbacks import VisualiseGAN, SaveModels


def train_g64_d64():
    nb_units = 64
    generator_input_dim = 100
    nb_real = 64
    nb_fake = 96
    lr = 0.0002
    beta_1 = 0.5
    nb_batches_per_epoch = 100
    nb_epoch = 1000
    output_dir = "models/dcgan_g64_d64"
    hdf5_fname = "/home/leon/data/tags_plain_t6.hdf5"

    g = dcgan_generator(nb_units, generator_input_dim)
    d = dcgan_discriminator(nb_units)

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


def main():
    functions = [train_g64_d64]

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
    os.exit(1)

if __name__ == "__main__":
    main()
