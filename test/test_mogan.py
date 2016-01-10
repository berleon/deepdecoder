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


from conftest import TEST_OUTPUT_DIR
import os

import keras
from beras.gan import GAN
from beras.models import asgraph
from keras.layers.core import Dense
from keras.models import Sequential
from keras.optimizers import Adam
from keras.objectives import mse

from deepdecoder.mogan import MOGAN
import keras.backend as K
import numpy as np

import matplotlib.pyplot as plt


class Plotter(keras.callbacks.Callback):
    def __init__(self, data, sample_fn,
                 outdir=TEST_OUTPUT_DIR + "/mogan_plot"):
        super().__init__()
        self.X = data
        self.outdir = outdir
        self.sample_fn = sample_fn
        os.makedirs(outdir, exist_ok=True)

    def on_epoch_begin(self, epoch, logs={}):
        if epoch == 0:
            self._plot("on_begin_0.png")

    def on_epoch_end(self, epoch, logs={}):
        self._plot("on_end_{:04d}.png".format(epoch))

    def _plot(self, outname):
        ys = []
        nb_ys = 0
        selections = []
        while nb_ys <= len(self.X):
            y, selection = self.sample_fn()
            ys.append(y)
            selections.append(selection)
            nb_ys += len(ys[-1])
        Y = np.concatenate(ys)
        S = np.concatenate(selections)[:, 0]
        plt.ylim(-1, 1.5)
        plt.xlim(-1, 1.5)
        plt.scatter(self.X[:, 0], self.X[:, 1], marker='.', c='b', alpha=0.2)
        plt.scatter(Y[S == 0, 0], Y[S == 0, 1], marker='.', c='r', alpha=0.2)
        plt.scatter(Y[S == 1, 0], Y[S == 1, 1], marker='.', c='g', alpha=0.2)
        plt.savefig(os.path.join(self.outdir, outname))
        plt.close()


simple_gan_batch_size = 64
simple_gan_nb_z = 16
simple_gan_nb_cond = 4
simple_gan_nb_out = 4
simple_gan_z_shape = (simple_gan_batch_size, simple_gan_nb_z)


def test_mogan():
    def reconstruction_fn(g_outdict):
        x = g_outdict['output']
        eps = K.random_normal((simple_gan_batch_size, 2))
        return x[:, :2] + eps*x[:, 2:]

    def simple_gan():
        generator = Sequential()
        generator.add(Dense(20, activation='relu',
                            input_dim=simple_gan_nb_z + simple_gan_nb_cond))
        generator.add(Dense(50, activation='relu'))
        generator.add(Dense(50, activation='relu'))
        generator.add(Dense(simple_gan_nb_out))
        generator = asgraph(generator, inputs={
            GAN.z_name: (simple_gan_nb_z, ),
            "cond": (simple_gan_nb_cond, ),
        }, concat_axis=1)
        discriminator = Sequential()
        discriminator.add(Dense(25, activation='relu', input_dim=2))
        discriminator.add(Dense(1, activation='sigmoid'))
        discriminator = asgraph(discriminator, input_name=GAN.d_input)
        return GAN(generator, discriminator, simple_gan_z_shape,
                   reconstruct_fn=reconstruction_fn)
    mu1 = [0., 0.]
    sigma1 = 0.50
    mu2 = [1., 1.]
    sigma2 = 0.25

    def loss_fn(y_true, g_outmap):
        y_predicted = g_outmap['output']
        return mse(y_true, y_predicted)

    def data(bs=1024):
        shape = (bs, 2)
        x1 = np.random.multivariate_normal(mu1, sigma1**2*np.eye(2), shape[:1])
        x2 = np.random.multivariate_normal(mu2, sigma2**2*np.eye(2), shape[:1])
        selection = np.random.binomial(1, 0.5, (shape[0], 1)).repeat(2, axis=1)
        return np.where(selection, x1, x2)

    def expected(bs=1024):
        first = np.array([mu1 + [sigma1, sigma1]]).repeat(bs, axis=0)
        secon = np.array([mu2 + [sigma2, sigma2]]).repeat(bs, axis=0)
        selection = np.random.binomial(1, 0.5, (bs, 1))\
            .repeat(first.shape[1], axis=1)
        return np.where(selection, first, secon), selection

    def generate():
        expec, selection = expected(simple_gan_batch_size)
        return gan.generate(conditionals=[expec]), selection

    bs = simple_gan_batch_size*25

    def optimizer_lambda():
        return Adam(lr=0.0002, beta_1=0.5)

    gan = simple_gan()
    mogan = MOGAN(gan, loss_fn, optimizer_lambda, gan_objective='mse')
    mogan.compile()
    expected_val, _ = expected(bs)
    print(expected_val.shape)
    mogan.mulit_objectives.fit(
            [data(bs), expected_val, expected_val],
            batch_size=gan.batch_size, verbose=1, nb_epoch=100,
            callbacks=[Plotter(data(3000), sample_fn=generate)])
