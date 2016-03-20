#! /usr/bin/env python
# Copyright 2016 Leon Sixt
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import matplotlib
matplotlib.use('Agg')  # noqa

from hyperopt import hp, Trials, STATUS_OK, fmin, tpe, STATUS_FAIL
from deepdecoder.networks import dcgan_small_generator, dcgan_discriminator

from beras.layers.core import LinearInBounds
from beras.gan import GAN, gan_binary_crossentropy
from beras.callbacks import VisualiseGAN
from deepdecoder.deconv import Deconvolution2D
from deepdecoder.data import real_generator
from deepdecoder.networks import mask_blending_gan_hyperopt, get_mask_driver
from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers.core import Activation, Layer, Dense, Reshape
from keras.layers.normalization import BatchNormalization
import keras.initializations
import time
import traceback
import numpy as np
import hashlib
import os
import pickle
import sys


def network_space():
    space = {}

    def add(name, fn, *args):
        space[name] = fn(name, *args)

    add('z_dim', hp.quniform, 10, 50, 10)
    add('discriminator_nb_units', hp.quniform, 16, 64, 16)
    add('gan_regularizer', hp.choice, ('none', 'stop'))
    add('offset_inputs', hp.choice,
        (('gen_driver',),  ('gen_driver_zero_grad',),
         ('gen_driver', GAN.z_name),  ('gen_driver_zero_grad', GAN.z_name),
         (GAN.z_name,)))
    add('nb_merge_conv_layers', hp.quniform, 0, 2, 1)

    space['merge'] = 'merge_16'
    space['offset_nb_units'] = 64
    space['lr'] = 0.0002
    return space


def construct_gan(sample, nb_fake, nb_real):
    print('construct_gan')
    g_mask = dcgan_small_generator(nb_units=128, input_dim=22)
    g_mask.trainable = False
    d = dcgan_discriminator(out_activation='sigmoid')

    mask_driver = get_mask_driver(input_dim=sample['z_dim'],
                                  output_dim=g_mask.input_shape[1])
    print('construct_gan')
    graph = mask_blending_gan_hyperopt(
        mask_driver, g_mask, d,
        offset_inputs=sample['offset_inputs'],
        offset_nb_units=sample['offset_nb_units'],
        merge=sample['merge'],
        nb_merge_conv_layers=sample['nb_merge_conv_layers'],
        z_dim=sample['z_dim'],
        nb_fake=nb_fake,
        nb_real=nb_real,
    )
    g_mask.load_weights("models/train_autoencoder_mask_generator_adam_n128_gray/mask_generator.hdf5")
    return GAN(graph)


def normal(scale=0.02):
    def normal_wrapper(shape, name=None):
        return keras.initializations.normal(shape, scale, name)
    return normal_wrapper


def get_offset_generator(n=32, batch_input_shape=50, nb_output_channels=1,
                         init=normal(0.02), merge_mode=None, merge_layer=None):
    def deconv(model, nb_filter, h, w):
        model.add(Deconvolution2D(nb_filter, h, w, subsample=(2, 2),
                                  border_mode=(2, 2), init=init))
        model.add(BatchNormalization(axis=1))
        model.add(Activation('relu'))

    front = Sequential()
    front.add(Dense(8*n*4*4, batch_input_shape=batch_input_shape, init=init))
    front.add(BatchNormalization())
    front.add(Activation('relu'))
    front.add(Reshape((8*n, 4, 4,)))

    deconv(front, 4*n, 5, 5)
    deconv(front, 2*n, 5, 5)

    if merge_mode != 'merge_16':
        deconv(front, n, 5, 5)

    input_shape = list(front.output_shape)
    if merge_layer is not None:
        input_shape[1] += merge_layer.output_shape[1]

    input_shape = tuple(input_shape)
    back = Sequential()
    back.add(Layer(batch_input_shape=input_shape))

    if merge_mode == 'merge_16':
        deconv(back, n, 5, 5)

    back.add(Deconvolution2D(nb_output_channels, 5, 5, subsample=(2, 2),
                             border_mode=(2, 2), init=init))
    back.add(LinearInBounds())
    return front, back


def compile(gan, sample):
    lr = sample['lr']
    beta_1 = 0.5
    g_optimizer = Adam(lr, beta_1)
    d_optimizer = Adam(lr, beta_1)
    start = time.time()
    gan.compile(g_optimizer, d_optimizer, gan_binary_crossentropy)
    compile_time = time.time() - start
    return compile_time


def generator(nb_fake, nb_real, z_dim):
    for real in real_generator("/home/leon/data/tags_plain_t6.hdf5", nb_real):
        z = np.random.uniform(-1, 1, (nb_fake, z_dim)).astype(np.float32)
        yield {
            'real': 2*real - 1,
            GAN.z_name: z
        }


def objective(sample):
    def md5(x):
        m = hashlib.md5()
        m.update(str(sample).encode())
        return m.hexdigest()

    intify = ['z_dim', 'offset_nb_units', 'discriminator_nb_units',
              'nb_merge_conv_layers']
    for k in intify:
        sample[k] = int(sample[k])

    nb_fake = 96
    nb_real = 32
    nb_epoch = 25
    batches_per_epoch = 128

    output_dir = 'models/hyperopt/{}/'.format(md5(sample))
    os.makedirs(output_dir)
    print(sample)
    info = {
        'status': STATUS_OK,
        'output_dir': output_dir,
    }
    try:
        info['loss'] = 0.
        gan = construct_gan(sample, nb_fake, nb_real)
        if sample['gan_regularizer'] == 'stop':
            gan.add_gan_regularizer(GAN.StopRegularizer())
        print(info)
        compile_time = compile(gan, sample)
        print("Done Compiling in {0:.2f}s".format(compile_time))
        info['compile_time'] = compile_time
        vis = VisualiseGAN(10**2, output_dir,
                           preprocess=lambda x: np.clip(x, -1, 1))

        hist = gan.fit_generator(generator(nb_fake, nb_real, sample['z_dim']),
                                 batches_per_epoch, nb_epoch,
                                 verbose=1, callbacks=[vis])
        g_loss = np.mean(np.array(hist.history['g_loss']))
        d_loss = np.mean(np.array(hist.history['d_loss']))
        info['loss'] = g_loss - d_loss
        info['hist'] = hist.history
        gan.graph.save_weights(output_dir + 'graph.hdf5')
        with open(output_dir + 'gan.json', 'w') as f:
            f.write(gan.graph.to_json())
    except KeyboardInterrupt:
        sys.exit(1)
    except:
        err, v, tb = sys.exc_info()
        print(type(err))
        print(err)
        print(v)
        print(tb)
        traceback.print_tb(tb, limit=50)
        info['status'] = STATUS_FAIL
    finally:
        return info

if os.path.exists('trails.pickle'):
    with open('trails.pickle', 'rb') as f:
        trials = pickle.load(f)
else:
    trials = Trials()

trials = Trials()
best = fmin(objective,
            space=network_space(),
            algo=tpe.suggest,
            max_evals=1,  # len(trials) + 1,
            trials=trials)
sys.exit(0)
with open('trails.pickle', 'wb') as f:
    pickle.dump(trials, f)

print(best)
