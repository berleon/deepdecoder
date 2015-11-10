# coding: utf-8
import random
import operator
import os
import time
import itertools
import sys
from collections import namedtuple

import h5py
import scipy
import numpy as np
import theano
import theano.tensor as T
import theano.tensor.shared_randomstreams as T_random
import matplotlib.pyplot as plt
from beras.layers.attention import RotationTransformer
from deepdecoder import NUM_CELLS
from keras.models import Sequential, Graph
from keras.layers.core import Dense, Dropout, Activation, Flatten, Reshape, Layer
from keras.layers.convolutional import Convolution2D, MaxPooling2D, UpSample2D
from keras.optimizers import SGD, Adam

from beras.gan import GAN, upsample
from beras.models import AbstractModel
from beras.util import downsample, blur

from deepdecoder.generate_grids import BlackWhiteArtist, MASK, MASK_BLACK,     MASK_WHITE, GridGenerator, MaskGridArtist
import deepdecoder.generate_grids as gen_grids


PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

import GeneratedGridTrainer
from GeneratedGridTrainer import NetworkArgparser
from mask_loss import mask_loss

floatX = theano.config.floatX

batch_size = 128


def bw_mask(mask):
    bw = 0.5*T.ones_like(mask, dtype=floatX)
    bw = T.set_subtensor(bw[(mask > MASK["IGNORE"]).nonzero()], 1.)
    bw = T.set_subtensor(bw[(mask < MASK["IGNORE"]).nonzero()], 0)
    return bw


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
        return bw_mask(self.get_input(train))


class MaskNotIgnore(Layer):
    def get_output(self, train=False):
        mask = self.get_input(train)
        ni = T.zeros_like(mask, dtype=floatX)
        ni = T.set_subtensor(ni[T.neq(mask, MASK["IGNORE"]).nonzero()], 1.)
        return ni


class AddChannelsWeighted(Layer):
    def get_output(self, train=False):
        nb_channels = self.input_shape[1]
        nb_rows, nb_cols = self.input_shape[2:]
        nb_conv_channels = nb_channels - 2                          
        input = self.get_input(train)
        nb_samples = input.shape[0]
        mask = input[:, :1]
        dense = input[:, 1:2]
        conv = input[:, 2:]
        channels = []
        for c in range(nb_conv_channels):
            channel = T.switch(T.eq(mask, 1), conv[:, c:c+1], dense)
            channels.append(channel)
        
        return T.concatenate(channels, axis=1)
    
    @property
    def output_shape(self):
        shp = self.input_shape
        return shp[:1] + (shp[1] - 2,) + shp[2:]


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
    gen.add_node(MaskBlackWhite(), name="mask_bw", input="cond_0")
    n = 64*64
    gen.add_node(Flatten(input_shape=(2, 64, 64)), name='input_flat', 
                 inputs=['z', 'mask_bw'], merge_mode='concat', concat_axis=1)
    gen.add_node(Dense(2*n, activation='relu'),
                 name='dense1', input='input_flat')
    
    gen.add_node(Dense(n, activation='relu'), name='dense2', input='dense1')
    gen.add_node(Dense(n, activation='sigmoid'), name='dense3', input='dense2')
    gen.add_node(Reshape((1, 64, 64)), name='dense_reshape', input='dense3')
    gen.add_output('output', input='dense_reshape')
    
    dis = Sequential()
    dis.add(Convolution2D(12, 3, 3, border_mode='same', activation='relu', input_shape=(1, 64, 64)))
    dis.add(MaxPooling2D())
    dis.add(Dropout(0.5))
    dis.add(Convolution2D(18, 3, 3, border_mode='same', activation='relu'))
    dis.add(MaxPooling2D())
    dis.add(Dropout(0.5))
    dis.add(Flatten())
    dis.add(Dense(256, activation="relu"))
    dis.add(Dropout(0.5))
    dis.add(Dense(1, activation="sigmoid"))
    return GAN(gen, dis, z_shapes=[(batch_size//2, 1, 64, 64)], num_gen_conditional=1)


def get_decoder_model():
    rot_model = Sequential()
    rot_model.add(Convolution2D(12, 5, 5, activation='relu',
                                input_shape=(1, 64, 64)))
    rot_model.add(MaxPooling2D((3, 3)))
    rot_model.add(Dropout(0.5))
    rot_model.add(Flatten())
    rot_model.add(Dense(256))
    rot_model.add(Dropout(0.5))
    rot_model.add(Dense(1, activation='relu'))

    model = Sequential()
    model.add(RotationTransformer(rot_model, input_shape=(1, 64, 64)))
    model.add(Convolution2D(16, 3, 3, border_mode='valid',
                            activation='relu'))
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

    model.add(Flatten())
    model.add(Dense(1200))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(NUM_CELLS))
    model.add(Activation('sigmoid'))
    return model


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    def __getattr__(self, attr):
        return self.get(attr)
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class MaskGAN(AbstractModel):
    def __init__(self, gan, decoder):
        self.gan = gan
        self.decoder = decoder
        self.batch_size = 128

    def build_graph(self, optimizer_factory):
        p = dotdict()
        p.mask_idx = T.tensor4("mask_idx")
        p.mask_bw = bw_mask(p.mask_idx)
        p.x_real = T.tensor4("x_real")

        p.gan_losses = self.gan.losses(p.x_real, gen_conditional=[p.mask_idx])
        p.image = self.gan.g_out
        p.image_blur = blur(p.image)
        p.mask_loss_dict = mask_loss(p.mask_idx, p.image_blur)
        p.gan_mask_loss = p.mask_loss_dict['loss']

        p.g_loss = p.gan_mask_loss + p.gan_losses[0]
        p.g_updates = optimizer_factory().get_updates(
            self.gan.G.params, self.gan.G.constraints, p.g_loss)
        d_loss = p.gan_losses[1]
        p.d_updates = optimizer_factory().get_updates(
            self.gan.D.params, self.gan.D.constraints, d_loss)

        p.mask_updates = optimizer_factory().get_updates(
            self.gan.G.params, self.gan.G.constraints, p.gan_mask_loss)
        return p

    def _build_train(self, p, mode, compile_fn=True):
        self.train_labels = ['g_on_d', 'd', 'd_real', 'd_gen', "m", "g"]
        if compile_fn:
            self._train = theano.function(
                [p.x_real, p.mask_idx],
                p.gan_losses + (p.gan_mask_loss, p.g_loss),
                updates=p.g_updates + p.d_updates, mode=mode)

    def _build_train_mask_only(self, p, mode, compile_fn=True):
        if compile_fn:
            self._train_mask_only = theano.function(
                [p.mask_idx], [p.gan_mask_loss],
                updates=p.mask_updates, mode=mode)
        self.train_mask_only_labels = ["m"]

    def _build_generate(self, p, mode, compile_fn=True):
        self._generate_labels = ["image"]
        self._generate = theano.function([p.mask_idx],
                                                [p.image], mode=mode)

    def _build_debug(self, p, mode, compile_fn=True):
        self.debug_labels = ["real", "mask_bw", "image", "d_out",
                             "mask_loss_per_sample", "black_white_loss",
                             "ring_loss", "cell_losses"] + self.train_labels
        ml = p.mask_loss_dict
        self._debug = theano.function(
            [p.x_real, p.mask_idx],
            [p.x_real, p.mask_bw, p.image, self.gan.d_out,
             ml["loss_per_sample"],
             ml['black_white_loss'],
             ml['ring_loss'],
             T.stack(ml["cell_losses"]), ] + list(p.gan_losses) +
            [p.gan_mask_loss, p.g_loss],
            mode=mode)

    def compile(self, optimizer_factory, functions=None, mode=None):

        all_functions = ['generate', 'debug', 'train_decoder',
                     'train_mask_only', 'train_gan', 'train_all']
        if functions is None:
            functions = ['train_gan']
        if functions == "all":
            functions = all_functions

        for f in functions:
            assert f in all_functions, "Unknown function: " + f

        p = self.build_graph(optimizer_factory)

        self._build_train(p, mode, 'train_gan' in functions)
        self._build_train_mask_only(p, mode, 'train_mask_only' in functions)
        self._build_generate(p, mode, 'generate' in functions)
        self._build_debug(p, mode, 'debug' in functions)

    def fit(self, X, mask_idx, nb_epoch=100, verbose=0,
            callbacks=None,  shuffle=True, train_mode="gan"):
        if callbacks is None:
            callbacks = []

        def train(model, batch_indicies, batch_index, batch_logs=None):
            global last_log
            if batch_logs is None:
                batch_logs = {}
    
            b = batch_index*self.batch_size//2
            e = (batch_index+1)*self.batch_size//2
            masks_idx_batch = mask_idx[b:e]
            if train_mode == "gan":
                outs = self._train(X[batch_indicies], masks_idx_batch)
                for key, value in zip(self.train_labels, outs):
                    batch_logs[key] = float(value)
            elif train_mode == "mask_only":
                outs = self._train_mask_only(masks_idx_batch)
                if type(outs) != list:
                    outs = [outs]
                    
                for key, value in zip(self.train_mask_only_labels, outs):
                    batch_logs[key] = float(value)
            else:
                raise ValueError("unknown train_mode `{}`".format(train_mode))
                
        assert self.batch_size % 2 == 0, "batch_size must be multiple of two."
        self._fit(train, len(X), batch_size=self.batch_size//2, nb_epoch=nb_epoch,
                  verbose=verbose, callbacks=callbacks, shuffle=shuffle, metrics=self.train_labels)

    def load_weights(self, fname):
        self.gan.G.load_weights("generator")
        self.gan.D.load_weights("detector")
        self.gan.D.load_weights("decoder")

    def save_weights(self, fname, overwrite=False):
        self.gan.G.save_weights(fname.format("generator"), overwrite)
        self.gan.D.save_weights(fname.format("detector"), overwrite)
        self.decoder.save_weights(fname.format("decoder"), overwrite)

    def generate(self, *conditional):
        return self._generate(*conditional)[0]

    def debug(self, X, mask_idx):
        outs = self._debug(X, mask_idx)
        return dict(zip(self.debug_labels, outs))


def masks(batch_size):
    mini_batch = 64
    batch_size += mini_batch - (batch_size % mini_batch)
    generator = GridGenerator()
    artist = MaskGridArtist()
    for masks in gen_grids.batches(batch_size, generator, artist=artist,
                                   with_gird_params=True, scales=[1.]):
        yield (masks[0].astype(floatX), masks[1:])


def tags_from_hdf5(fname):
    tags_list = []
    with open(fname) as pathfile:
        for hdf5_name in pathfile.readlines():
            hdf5_fname = os.path.dirname(fname) + "/" + hdf5_name.rstrip('\n')
            f = h5py.File(hdf5_fname)
            data = np.asarray(f["data"])
            labels = np.asarray(f["labels"])
            tags = data[labels == 1]
            tags /= 255.
            tags_list.append(tags)
        
    return np.concatenate(tags_list)


def loadRealData(fname):
    X = tags_from_hdf5(fname)
    X = X[:(len(X)//64)*64]
    return X


def loadGAN(weight_dir):
    gan = MaskGAN(get_gan_dense_model(), get_decoder_model())
    if os.path.exists(weight_dir):
        print("Loading weights from: " + weight_dir)
        gan.gan.load_weights(weight_dir + "{}.hdf5")
    return gan


def train(args):
    X = loadRealData(args.real)
    gan = loadGAN(args.weight_dir)
    print("Compiling...")
    start = time.time()
    gan.compile(lambda: SGD(), functions=["debug", "train_gan", "train_mask_only"])
    print("Done Compiling in {0:.2f}s".format(time.time() - start))

    try:
        for i, masks_idx in enumerate(itertools.islice(masks(len(X)), 100)):
            print(i)
            gan.fit(X, masks_idx[0], verbose=1, nb_epoch=1, train_mode="gan")
    except KeyboardInterrupt:
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
    trainer.set_gt_path_file("/home/leon/uni/vision_swp/deeplocalizer_data/gt_preprocessed/gt_files.txt")
    real_images, _ = next(trainer.real_batches())
    fake_images, _ = next(trainer.batches())
    os.makedirs("real_images", exist_ok=True)
    for i in range(len(real_images)):
        scipy.io.imsave("real_images/{}.jpeg".format(i), real_images[i, 0])
    model = get_decoder()
    model.load_weights(args.weights)
    trainer.minibatches_per_epoche = 1
    trainer.minibatch_size = 8
    trainer.real_test(model, n_epochs=1000)
    trainer.test(model, n_epochs=1)

if __name__ == "__main__":
    argparse = NetworkArgparser(train, test)
    argparse.parser_train.add_argument(
        "--real", required=True, type=str,
        help="path to a txt file with paths to hdf5 files")

    argparse.parser_train.add_argument(
        "--train_mode", default="gan", type=str,
        help="possible modes are: gan, mask_only, decoder_only, gan_decoder")
    argparse.parse_args()
