# coding: utf-8


import keras.objectives
import theano.tensor.shared_randomstreams as T_random

from keras.models import Sequential, Graph
from keras.layers.core import Dense, Dropout, Flatten, Reshape, Layer
from keras.layers.convolutional import Convolution2D, MaxPooling2D

from beesgrid import NUM_MIDDLE_CELLS
from beras.gan import GAN
from beras.layers.attention import RotationTransformer
from beras.util import downsample, blur

from deepdecoder.mutliple_objectives import MultipleObjectives
from .utils import *

import keras.backend as K

def bw_background_mask(mask):
    bw = T.zeros_like(mask, dtype=floatX)
    bw = T.set_subtensor(bw[(mask > MASK["IGNORE"]).nonzero()], 1.)

    bg = T.zeros_like(mask, dtype=floatX)
    bg = T.set_subtensor(bg[T.eq(mask, MASK["IGNORE"]).nonzero()], 1.)
    bg = T.set_subtensor(bg[T.eq(mask, MASK["BACKGROUND_RING"]).nonzero()], 1.)
    return T.concatenate([bw, bg], axis=1)


class UniformNoise(Layer):
    def __init__(self, shape, **kwargs):
        super().__init__(**kwargs)
        self.z_shape = shape
        self.rs = T_random.RandomStreams(1334)

    def get_output(self, train=False):
        return self.rs.uniform(self.z_shape, -1, 1)


class AddConvDense(Layer):
    def get_output(self, train=False):
        return 0.5 * self.get_input(train)


class MaskBlackWhite(Layer):
    def get_output(self, train=False):
        return binary_mask(self.get_input(train))
    @property
    def output_shape(self):
        shp = self.input_shape
        return (shp[0], 1,) + shp[2:]

class MaskBWBackground(Layer):
    def get_output(self, train=False):
        return bw_background_mask(self.get_input(train))

    @property
    def output_shape(self):
        shp = self.input_shape
        return (shp[0], 2,) + shp[2:]

class MaskNotIgnore(Layer):
    def get_output(self, train=False):
        mask = self.get_input(train)
        ni = T.zeros_like(mask, dtype=floatX)
        ni = T.set_subtensor(ni[T.neq(mask, MASK["IGNORE"]).nonzero()], 1.)
        return ni


class Downsample(Layer):
    def get_output(self, train=False):
        return downsample(self.get_input(train))
    @property
    def output_shape(self):
        shp = self.input_shape
        return shp[:2] + (shp[2]//2, shp[3]//2)

    
class Blur(Layer):
    def get_output(self, train=False):
        return blur(self.get_input(train))


def get_gan_dense_model(batch_size=128):
    gen = Graph()
    gen.add_input('z', (1, 64, 64))
    gen.add_input('cond_0', (1, 64, 64))
    gen.add_node(MaskBWBackground(), name="mask_bwb", input="cond_0")
    n = 64*64
    gen.add_node(Flatten(input_shape=(3, 64, 64)), name='input_flat',
                 inputs=['z', 'mask_bwb'], merge_mode='concat', concat_axis=1)
    gen.add_node(Dense(n, activation='relu'),
                 name='dense1', input='input_flat')
    
    gen.add_node(Dense(n, activation='relu'), name='dense2', input='dense1')
    gen.add_node(Dense(n, activation='sigmoid'), name='dense3', input='dense2')
    gen.add_node(Reshape((1, 64, 64)), name='dense_reshape', input='dense3')
    gen.add_output('output', input='dense_reshape')
    
    dis = Sequential()
    dis.add(Convolution2D(12, 3, 3, activation='relu', input_shape=(1, 64, 64)))
    dis.add(MaxPooling2D())
    dis.add(Dropout(0.5))
    dis.add(Convolution2D(18, 3, 3, activation='relu'))
    dis.add(MaxPooling2D())
    dis.add(Dropout(0.5))
    dis.add(Convolution2D(18, 3, 3, activation='relu'))
    dis.add(Dropout(0.5))
    dis.add(Flatten())
    dis.add(Dense(256, activation="relu"))
    dis.add(Dropout(0.5))
    dis.add(Dense(1, activation="sigmoid"))
    return GAN(gen, dis, z_shape=(batch_size//2, 1, 64, 64),
               num_gen_conditional=1)


def get_decoder_model():
    size = 64
    rot_model = Sequential()
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


class MOGAN:
    def __init__(self, gan: GAN, loss_fn, optimizer_fn,
                 name="mogan",
                 gan_objective=keras.objectives.binary_crossentropy,
                 gan_regulizer=None):
        assert len(gan.conditionals) >= 1
        y_true = K.placeholder(shape=gan.G.outputs["output"].output_shape)
        v = gan.build_loss(objective=gan_objective)
        inputs = [v.real, y_true] + v.gen_conditionals
        cond_loss = loss_fn(y_true, v.g_outmap)
        gan.build_opt_d(optimizer_fn(), v)

        gan_regulizer = GAN.get_regulizer(gan_regulizer)
        v.g_loss, v.d_loss, v.reg_updates = \
            gan_regulizer.get_losses(gan, v.g_loss, v.d_loss)
        self.build_dict = v
        self.gan = gan
        self.optimizer_fn = optimizer_fn
        outputs_map = {
            "cond_loss": cond_loss.mean(),
            "d_loss": v.d_loss,
            "g_loss": v.g_loss,
        }
        if type(gan_regulizer) == GAN.L2Regularizer:
            outputs_map["l2"] = gan_regulizer.l2_coef
        self.mulit_objectives = MultipleObjectives(
                name, inputs,
                outputs_map=outputs_map,
                params=gan.G.params,
                objectives=[v.g_loss, cond_loss],
                additional_updates=v.d_updates + v.reg_updates)

    def compile(self):
        self.gan._compile_generate(self.build_dict)
        self.mulit_objectives.compile(self.optimizer_fn)
