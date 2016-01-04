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
import time

import keras
import theano
from beesgrid.pybeesgrid import NUM_CONFIGS, NUM_MIDDLE_CELLS
from beras.gan import GAN
from beesgrid import TAG_SIZE, CONFIG_ROTS
from beras.models import asgraph
from keras import initializations
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import Convolution2D
from keras.layers.core import Dense, Layer, Reshape, Activation, Flatten, Lambda
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential, Graph
from keras.objectives import mse
from keras.optimizers import Adam
from theano.sandbox.cuda.basic_ops import gpu_alloc_empty, gpu_contiguous
from theano.sandbox.cuda.dnn import GpuDnnConvDesc, GpuDnnConvGradI

from deepdecoder.mogan import MOGAN
from deepdecoder.utils import loadRealData, binary_mask
import theano.tensor as T
import numpy as np

class Deconvolution2D(Layer):
    def __init__(self, nb_filter, nb_row, nb_col,
                 subsample=(1, 1), border_mode=(0, 0), init='glorot_uniform', **kwargs):
        self.nb_filter = nb_filter
        self.nb_row = nb_row
        self.nb_col = nb_col
        self.init = initializations.get(init)
        self.border_mode = border_mode
        self.subsample = subsample
        super(Deconvolution2D, self).__init__(**kwargs)

    def build(self):
        stack_size = self.input_shape[1]
        self.W_shape = (stack_size, self.nb_filter, self.nb_row, self.nb_col)
        self.W = self.init(self.W_shape)
        self.params = [self.W]

    @property
    def output_shape(self):
        in_rows, in_cols = self.input_shape[2:]
        sr, sc = self.subsample
        return self.input_shape[0], self.nb_filter, sr*in_rows, sr*in_cols

    def get_output(self, train=False):
        """
        sets up dummy convolutional forward pass and uses its grad as deconv
        currently only tested/working with same padding
        """
        img = self.get_input(train)
        kerns = self.W
        sr, sc = self.subsample
        out = gpu_alloc_empty(img.shape[0], kerns.shape[1], img.shape[2]*sr, img.shape[3]*sc)
        desc = GpuDnnConvDesc(
            border_mode=self.border_mode, subsample=self.subsample,
            conv_mode='conv')(out.shape, kerns.shape)
        d_img = GpuDnnConvGradI()(kerns, img, out, desc)
        return d_img


def generator(nb_output_channels=1):
    n = 64
    model = Sequential()
    model.add(Dense(8*n*4*4, input_dim=50))
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


def dcmogan(generator_fn, discriminator_fn, batch_size=128):
    nb_g_z = 20
    nb_grid_config = NUM_CONFIGS + NUM_MIDDLE_CELLS + len(CONFIG_ROTS)
    ff_generator = generator_fn(nb_output_channels=2)

    grid_config_shape = (batch_size, nb_grid_config)
    g = Graph()
    g.add_input("z", (nb_g_z, ))
    g.add_input("grid_config", grid_config_shape[1:])
    g.add_node(Dense(ff_generator.layers[0].input_shape[1], activation='relu'), "dense1",
               inputs=["z", "grid_config"], merge_mode='concat')
    g.add_node(ff_generator, "dcgan", input="dense1")
    g.add_output("output", input="dcgan")
    g.add_node(Dense(1, activation='sigmoid'), "alpha", input="dense1",
               create_output=True)

    def reconstruct(g_outmap):
        g_out = g_outmap["output"]
        alpha = g_outmap["alpha"]
        alpha = 0.5*alpha + 0.5
        alpha = alpha.reshape((batch_size, 1, 1, 1))
        m = g_out[:, :1]
        v = g_out[:, 1:]
        return (alpha * m + (1 - alpha) * v).reshape((batch_size, 1, TAG_SIZE,
                                                     TAG_SIZE))

    grid_loss_weight = theano.shared(np.cast[np.float32](1))

    def grid_loss(grid_idx, g_outmap):
        g_out = g_outmap['output']
        m = g_out[:, :1]
        b = binary_mask(grid_idx, ignore=0.0,  white=1.)
        return grid_loss_weight*mse(b, m)

    gan = GAN(g, asgraph(discriminator_fn(), input_name=GAN.d_input),
              z_shape=(batch_size, nb_g_z),
              reconstruct_fn=reconstruct)
    mogan = MOGAN(gan, grid_loss, lambda: Adam(lr=0.0002, beta_1=0.5),
                  gan_regulizer=GAN.L2Regularizer())

    return mogan, grid_loss_weight


def discriminator():
    n = 64
    model = Sequential()
    model.add(Convolution2D(n, 5, 5, subsample=(2, 2), border_mode='same',
                            input_shape=(1, 64, 64)))
    model.add(LeakyReLU(0.2))

    model.add(Convolution2D(2*n, 5, 5, subsample=(2, 2), border_mode='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(0.2))

    model.add(Convolution2D(4*n, 5, 5, subsample=(2, 2), border_mode='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(0.2))

    model.add(Convolution2D(8*n, 5, 5, subsample=(2, 2), border_mode='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(0.2))

    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))
    return model


def load_gan(weight_dir=None):
    batch_size = 128
    gan = GAN(asgraph(generator()), asgraph(discriminator()),
              z_shape=(batch_size, 50,))
    if weight_dir:
        weight_dir = os.path.realpath(weight_dir)
        if os.path.exists(weight_dir):
            print("Loading weights from: " + weight_dir)
            gan.load_weights(weight_dir + "/{}.hdf5")
    return gan


def train(args):
    X = loadRealData(args.real)
    gan = load_gan(args.weight_dir)
    print("Compiling...")
    start = time.time()

    optimizer = lambda: Adam(lr=0.0002, beta_1=0.5)
    gan.compile(optimizer(), optimizer())
    print("Done Compiling in {0:.2f}s".format(time.time() - start))

    try:
        gan.fit(X, verbose=1, nb_epoch=1)
    finally:
        print("Saving model to {}".format(os.path.realpath(args.weight_dir)))
        output_dir = args.weight_dir + "/{}.hdf5"
        gan.save_weights(output_dir)

