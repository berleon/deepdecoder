# coding: utf-8
import random
import operator

import os
import time
import itertools
from dotmap import DotMap

import numpy as np
import theano
import theano.tensor as T
import theano.tensor.shared_randomstreams as T_random
import matplotlib.pyplot as plt

from keras import objectives
from keras.models import Sequential, Graph
from keras.layers.core import Dense, Dropout, Activation, Flatten, Reshape, Layer
from keras.layers.convolutional import Convolution2D, MaxPooling2D, UpSample2D
from keras.optimizers import SGD, Adam

from deepdecoder import NUM_CELLS
from beras.gan import GAN
from beras.models import AbstractModel
from beras.layers.attention import RotationTransformer
from beras.util import downsample, blur, BorderReflect

from deepdecoder.generate_grids import MASK, MASK_BLACK, MASK_WHITE, \
    GridGenerator, MaskGridArtist
import deepdecoder.generate_grids as gen_grids

from mask_loss import mask_loss
from .utils import *
import GeneratedGridTrainer
from GeneratedGridTrainer import NetworkArgparser


batch_size = 128




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


def get_gan_dense_model():
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
    return GAN(gen, dis, z_shapes=[(batch_size//2, 1, 64, 64)], num_gen_conditional=1)


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
    model.add(Dense(NUM_CELLS, activation='sigmoid'))
    return model


class MaskGAN(AbstractModel):
    def __init__(self, gan, decoder=None):
        self.gan = gan
        self.decoder = decoder
        self.batch_size = batch_size

        self.compile_functions = {
            'generate': self._compile_generate,
            'debug': self._compile_debug,
            'train_gan': self._compile_train_gan,
            'train_gan_mask': self._compile_train_gan_mask,
            'train_decoder': self._compile_train_decoder,
            'train_mask_only': self._compile_train_mask_only,
        }

    def build_graph(self):
        mask_idx = T.tensor4("mask_idx")
        mask_bw = binary_mask(mask_idx)
        real = T.tensor4("real")
        mask_labels = T.matrix("mask_labels")

        g_loss_on_d, d_loss, d_loss_real, d_loss_fake = self.gan.losses(real, gen_conditional=[mask_idx])
        d_in = self.gan.D.layers[0].input
        fake_image = self.gan.g_out
        fake_image_blur = blur(fake_image)
        mask_loss_dict = mask_loss(mask_idx, fake_image_blur)
        gan_mask_loss = mask_loss_dict['loss']

        if self.decoder:
            decoder_in = self.decoder.get_input(train=False)
            decoder_out = self.decoder.get_output(train=False)

            self.decoder.layers[0].input = fake_image
            decoder_out_on_gan = self.decoder.get_output(train=True)
            decoder_rotation = self.decoder.layers[0].get_output(train=True)
            decoder_loss = objectives.mse(mask_labels,
                                          decoder_out_on_gan).mean()

        g_loss_mask = gan_mask_loss + g_loss_on_d

        local_dict = locals()
        del local_dict['self']
        return DotMap(local_dict)

    train_gan_labels = ['g_on_d', 'd', 'd_real', 'd_fake']

    def _compile_train_gan(self, p, optimizer_factory, mode, **kwargs):
        gan_regularizer = kwargs['gan_regularizer']()
        g_loss, d_loss = gan_regularizer.get_losses(self.gan,
                                                    p.g_loss_on_d, p.d_loss)
        d_updates = optimizer_factory("d").get_updates(
            self.gan.D.params, self.gan.D.constraints, d_loss)

        g_updates = optimizer_factory("g").get_updates(
            self.gan.G.params, self.gan.G.constraints, g_loss)

        self._train_gan = theano.function(
            [p.real, p.mask_idx],
            (g_loss, d_loss, p.d_loss_real, p.d_loss_fake),
            updates=g_updates + d_updates,
            mode=mode)

    train_gan_mask_labels = ["m", "g", 'g_on_d', 'd', 'd_real', 'd_fake', 'd_l2']

    def _compile_train_gan_mask(self, p, optimizer_factory, mode, **kwargs):
        gan_regularizer = kwargs['gan_regularizer']()
        g_loss, d_loss = gan_regularizer.get_losses(self.gan,
                                                    p.g_loss_on_d, p.d_loss)
        d_updates = optimizer_factory("d").get_updates(
            self.gan.D.params, self.gan.D.constraints, d_loss)

        g_loss_with_mask = g_loss + p.gan_mask_loss
        g_updates_mask = optimizer_factory("g").get_updates(
            self.gan.G.params, self.gan.G.constraints, g_loss_with_mask)

        self._train_gan_mask = theano.function(
            [p.real, p.mask_idx],
            (p.gan_mask_loss, g_loss_with_mask, p.g_loss_on_d, d_loss,
             p.d_loss_real, p.d_loss_fake, gan_regularizer.l2_coef),
            updates=g_updates_mask + d_updates,
            mode=mode)

    train_decoder_labels = ["decoder"]

    def _compile_train_decoder(self, p, optimizer_factory, mode, **kwargs):
        decoder_updates = optimizer_factory("decoder").get_updates(
            self.decoder.params, self.decoder.constraints, p.decoder_loss)
        self._train_decoder = theano.function(
            [p.mask_idx, p.mask_labels],
            [p.decoder_loss],
            updates=decoder_updates,
            mode=mode)

    train_mask_only_labels = ["m"]

    def _compile_train_mask_only(self, p, optimizer_factory, mode, **kwargs):
        g_updates_mask_only = optimizer_factory("g").get_updates(
            self.gan.G.params, self.gan.G.constraints, p.gan_mask_loss)
        self._train_mask_only = theano.function(
            [p.mask_idx], [p.gan_mask_loss],
            updates=g_updates_mask_only, mode=mode)

    def _compile_generate(self, p, optimizer_factory, mode, **kwargs):
        self._generate_labels = ["image"]
        self._generate = theano.function([p.mask_idx],
                                         [p.fake_image], mode=mode)

    def _compile_debug(self, p, optimizer_factory, mode, **kwargs):
        ml = p.mask_loss_dict
        ins = [p.real, p.mask_idx]

        self.debug_labels = [
            "real", "mask_bw", "image", "d_out", "mask_loss_per_sample",
            "black_white_loss", "ring_loss", "cell_losses", "m", "g", 'g_on_d',
            'd', 'd_real', 'd_fake']

        outs = [p.real, p.mask_bw, p.fake_image, self.gan.d_out,
             ml["loss_per_sample"], ml['black_white_loss'], ml['ring_loss'],
             T.stack(ml["cell_losses"]), p.gan_mask_loss, p.g_loss_mask,
             p.g_loss_on_d, p.d_loss, p.d_loss_real, p.d_loss_fake]

        if self.decoder:
            ins.append(p.mask_labels)
            outs.extend([p.decoder_out_on_gan, p.decoder_rotation])
            self.debug_labels.extend(["decoder_prediction", "decoder_rot"])

        self._debug = theano.function(ins, outs, mode=mode)

    def compile(self, optimizer_factory, gan_regularizer=None, functions=None, mode=None):
        if functions is None:
            functions = ['train_gan']
        if functions == "all":
            functions = list(self.compile_functions.keys())

        for f in functions:
            assert f in self.compile_functions.keys(), "Unknown function: " + f

        p = self.build_graph()

        for f in functions:
            self.compile_functions[f](p, optimizer_factory, mode,
                                      gan_regularizer=gan_regularizer)

    def fit_gan_enforce(self, X, mask_idx, nb_epoch=100, verbose=0,
                        callbacks=None, shuffle=True):
        if callbacks is None:
            callbacks = []
        n = len(X)

        labels = ["m_{}".format(l) for l in self.train_gan_mask_labels] + \
            self.train_gan_labels

        empty_mask_idx = MASK["IGNORE"] * \
            np.ones((self.batch_size //2, 1, 64, 64), dtype=np.float32)

        def train(model, batch_indicies, batch_index, batch_logs=None):
            if batch_logs is None:
                batch_logs = {}

            s = batch_size//2*batch_index
            e = s + batch_size//2
            outs = self._train_gan_mask(X[batch_indicies], mask_idx[s:e])
            outs += self._train_gan(X[batch_indicies], empty_mask_idx)

            for key, value in zip(labels, outs):
                batch_logs[key] = float(value)

        assert self.batch_size % 2 == 0, "batch_size must be multiple of two."
        self._fit(train, n, batch_size=self.batch_size//2, nb_epoch=nb_epoch,
                  verbose=verbose, callbacks=callbacks, shuffle=shuffle, metrics=labels)

    def fit(self, ins, nb_epoch=100, verbose=0, direct_indexing=None,
            callbacks=None, shuffle=True, train_mode="gan"):
        if callbacks is None:
            callbacks = []
        if direct_indexing is None:
            direct_indexing = [False] * len(ins)

        assert len(ins) >= 1
        assert len(ins) == len(direct_indexing)
        n = len(ins[0])

        train_modes = {
            "gan": lambda: (self._train_gan, self.train_gan_labels),
            "gan_mask": lambda: (self._train_gan_mask, self.train_gan_mask_labels),
            "decoder": lambda: (self._train_decoder, self.train_decoder_labels),
            "mask_only": lambda: (self._train_mask_only, self.train_mask_only_labels),
        }

        if train_mode not in train_modes:
            raise ValueError("Unknown train mode: {}".format(train_mode))
        try:
            fn, labels = train_modes[train_mode]()
        except AttributeError:
            raise ValueError(
                "Function for {} was not compiled. Did you forget to add it "
                "in MaskGAN.compile(..., functions=[<Here>])?")

        def train(model, batch_indicies, batch_index, batch_logs=None):
            if batch_logs is None:
                batch_logs = {}

            batch_ins = []
            s = batch_size//2*batch_index
            e = s + batch_size//2
            for direct, arr in zip(direct_indexing, ins):
                if direct_indexing:
                    batch_ins.append(arr[s:e])
                else:
                    batch_ins.append(arr[batch_indicies])
            outs = fn(*batch_ins)

            if type(outs) != list:
                outs = [outs]
            for key, value in zip(labels, outs):
                batch_logs[key] = float(value)

        assert self.batch_size % 2 == 0, "batch_size must be multiple of two."
        self._fit(train, n, batch_size=self.batch_size//2, nb_epoch=nb_epoch,
                  verbose=verbose, callbacks=callbacks, shuffle=shuffle, metrics=labels)

    def load(self, fname):
        hdf5_tmpl = os.path.join(fname, "{}.hdf5")
        self.gan.G.load_weights(hdf5_tmpl.format("generator"))
        self.gan.D.load_weights(hdf5_tmpl.format("detector"))
        if self.decoder:
            self.decoder.load_weights(hdf5_tmpl.format("decoder"))

    def save(self, directory, overwrite=False):
        os.makedirs(directory, exist_ok=True)
        hdf5_tmpl = os.path.join(directory, "{}.hdf5")
        self.gan.G.save_weights(hdf5_tmpl.format("generator"), overwrite)
        self.gan.D.save_weights(hdf5_tmpl.format("detector"), overwrite)
        if self.decoder:
            self.decoder.save_weights(hdf5_tmpl.format("decoder"), overwrite)
        with open(directory + "/g.json", "w") as f:
            f.write(self.gan.G.to_json())
        with open(directory + "/d.json", "w") as f:
            f.write(self.gan.D.to_json())
        if self.decoder:
            with open(directory + "/decoder.json", "w") as f:
                f.write(self.decoder.to_json())

    def decoder_predict(self, fake_image):
        return self._decoder_predict(fake_image)

    def generate(self, *conditional):
        return self._generate(*conditional)[0]

    def debug(self, X, mask_idx, mask_labels):
        outs = self._debug(X, mask_idx, mask_labels)
        return dict(zip(self.debug_labels, outs))


def loadGAN(weight_dir):
    gan = MaskGAN(get_gan_dense_model())
    weight_dir = os.path.realpath(weight_dir)
    if os.path.exists(weight_dir):
        print("Loading weights from: " + weight_dir)
        gan.gan.load_weights(weight_dir + "/{}.hdf5")
    return gan


def train(args):
    def optimizer(name):
        if name in ["g", "d"]:
            return SGD()
        else:
            return Adam()
    X = loadRealData(args.real)
    gan = loadGAN(args.weight_dir)
    print("Compiling...")
    start = time.time()

    gan.compile(optimizer, functions=['debug', 'train_gan'])
    print("Done Compiling in {0:.2f}s".format(time.time() - start))

    try:
        for i, (masks_idx, masks_idx_small, mask_labels, grid_params) in \
                enumerate(itertools.islice(masks(len(X), scales=[1, 0.5]), 200)):
            print(i)
            gan.fit([X, masks_idx, masks_idx_small],
                    direct_indexing=[False, True, True], verbose=1, nb_epoch=1,
                    train_mode="gan")
    finally:
        print("Saving model to {}".format(os.path.realpath(args.weight_dir)))
        output_dir = args.weight_dir + "/{}.hdf5"
        gan.save_weights(output_dir)


def draw_debug(mask_idx, outs, b=None):
    if b is None:
        b = random.randint(0, 63)
    i = 1
    print(b)
    loss_img = np.zeros_like(mask_idx)
    for name, arr in sorted(outs.items(), key=operator.itemgetter(0)):
        if name == "d_out":
            print("fake: {0:.5f}".format(float(arr[b])))
            print("real: {0:.5f}".format(float(arr[b + batch_size // 2])))
        elif name == "mask_loss_per_sample":
            print("mask_loss: {0:.5f}".format(float(arr[b])))
        elif name == "cell_losses":
            for j, key in enumerate(MASK_BLACK + MASK_WHITE):
                idx = mask_idx == MASK[key]
                loss_img[idx] = arr[j, b]
        elif name in ["image", "mask_bw", "real"]:
            plt.subplot(1, 6, i)
            i += 1
            plt.imshow(arr[b, 0], cmap='gray')           
            plt.colorbar(fraction=0.046, pad=0.04)
            plt.title(name)
        elif name in ["black_white_loss", "ring_loss"]:
            print("{}: {}".format(name, arr[b]))
    
    plt.subplot(1, 6, i)    
    plt.imshow(loss_img[b, 0], cmap='Reds')           
    plt.colorbar(fraction=0.046, pad=0.04)

    plt.show()


def debug(args):
    X = loadRealData(args.real)
    gan = loadGAN(args.weights_dir)
    mask_idx, _, _ = next(masks(1))
    b = 1024
    real = X[b:b+64*6:6]
    outs = gan.debug(real, mask_idx)
    for i in range(64):
        draw_debug(mask_idx, outs, i)


def test(args):
    trainer = GeneratedGridTrainer.GeneratedGridTrainer()
    trainer.fit_data_gen()
    trainer.set_gt_path_file("/home/leon/repos/deeplocalizer_data/gt_images/gt_files.txt")
    real_images, _ = next(trainer.real_batches())
    model = get_decoder_model()
    weights = os.path.join(os.path.realpath(args.weight_dir), "decoder.hdf5")
    print("Loading weights from: " + weights)
    model.load_weights(weights)
    print("Compiling...")
    model.compile("adam", "mse")
    trainer.minibatches_per_epoche = 1
    trainer.minibatch_size = 8
    trainer.real_test(model, n_epochs=1000)

if __name__ == "__main__":
    argparse = NetworkArgparser(train, test)
    argparse.parser_train.add_argument(
        "--real", required=True, type=str,
        help="path to a txt file with paths to hdf5 files")

    argparse.parser_train.add_argument(
        "--train_mode", default="gan", type=str,
        help="possible modes are: gan, mask_only, decoder_only, gan_decoder")
    argparse.parse_args()
