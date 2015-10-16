#! /usr/bin/env python3
from beras.gan import GAN, stack_laplacian_gans, upsample
from beras.models import AbstractModel
from beras.util import downsample
from deepdecoder.generate_grids import BlackWhiteArtist, MASK, MASK_BLACK, \
    MASK_WHITE, GridGenerator, MaskGridArtist
import deepdecoder.generate_grids as gen_grids
import h5py
import itertools
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D, UpSample2D
import os.path

import base64

import os
import xml.etree.ElementTree as et
import io
import scipy.misc
from deepdecoder import NUM_CELLS
import sys
from keras.optimizers import SGD
from scipy.misc import imsave
import numpy as np
from theano.ifelse import ifelse
import theano
import theano.tensor as T
PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

from beras.layers.attention import RotationTransformer
import GeneratedGridTrainer
from GeneratedGridTrainer import NetworkArgparser

floatX = theano.config.floatX


def mask_loss(mask_image, image):
    white_mean = T.mean(image[(mask_image > MASK["IGNORE"]).nonzero()])
    black_mean = T.mean(image[(mask_image < MASK["IGNORE"]).nonzero()])
    loss = T.maximum(black_mean - white_mean, 0.)
    cell_weight = 1. / (len(MASK_BLACK) + len(MASK_WHITE) - NUM_CELLS - 1)
    for key in MASK_BLACK:
        cell_indicies = T.eq(mask_image, MASK[key]).nonzero()
        cell_mean = T.mean(image[cell_indicies])
        loss += ifelse(T.isnan(cell_mean),
                            # then
                            np.float64(0.),
                            # else
                            cell_weight * T.maximum(cell_mean - white_mean, 0.))
    for key in MASK_WHITE:
        if key == "OUTER_WHITE_RING":
            continue
        cell_indicies = T.eq(mask_image, MASK[key]).nonzero()
        cell_mean = T.mean(image[cell_indicies])
        loss += ifelse(T.isnan(cell_mean),
                            # then
                            np.float64(0.),
                            # else
                            cell_weight * T.maximum(black_mean - cell_mean, 0.))
    return loss

batch_size = 128


def get_alfa():
    gen = Sequential()
    gen.add(Convolution2D(10, 2, 2, 2, border_mode='same', activation='relu'))
    gen.add(Convolution2D(20, 10, 2, 2, border_mode='same', activation='relu'))
    gen.add(Convolution2D(10, 20, 2, 2, border_mode='same', activation='relu'))
    gen.add(Convolution2D(1, 10, 2, 2, border_mode='same', activation='relu'))

    dis = Sequential()
    dis.add(Convolution2D(5, 1, 2, 2, border_mode='same', activation='relu'))
    dis.add(Dropout(0.5))
    dis.add(Flatten())
    dis.add(Dense(16*16*5, 512, activation="relu"))
    dis.add(Dense(512, 1, activation="sigmoid"))
    return GAN(gen, dis, z_shape=(batch_size//2, 1, 16, 16), num_gen_conditional=1)


def get_bravo():
    gen = Sequential()
    gen.add(Convolution2D(10, 3, 2, 2, border_mode='same', activation='relu'))
    gen.add(Convolution2D(20, 10, 2, 2, border_mode='same', activation='relu'))
    gen.add(Convolution2D(10, 20, 2, 2, border_mode='same', activation='relu'))
    gen.add(Convolution2D(1, 10, 2, 2, border_mode='same', activation='sigmoid'))

    dis = Sequential()
    dis.add(Convolution2D(5, 2, 2, 2, border_mode='same', activation='relu'))
    dis.add(Dropout(0.5))
    dis.add(Flatten())
    dis.add(Dense(32*32*5, 512, activation="relu"))
    dis.add(Dense(512, 1, activation="sigmoid"))
    return GAN(gen, dis, z_shape=(batch_size//2, 1, 32, 32), num_gen_conditional=2, num_dis_conditional=1)


def get_charlie():
    gen = Sequential()
    gen.add(Convolution2D(10, 3, 2, 2, border_mode='same', activation='relu'))
    gen.add(Convolution2D(20, 10, 2, 2, border_mode='same', activation='relu'))
    gen.add(Convolution2D(10, 20, 2, 2, border_mode='same', activation='relu'))
    gen.add(Convolution2D(1, 10, 2, 2, border_mode='same', activation='sigmoid'))

    dis = Sequential()
    dis.add(Convolution2D(5, 2, 2, 2, border_mode='same', activation='relu'))
    dis.add(Dropout(0.5))
    dis.add(Flatten())
    dis.add(Dense(64*64*5, 512, activation="relu"))
    dis.add(Dense(512, 1, activation="sigmoid"))
    return GAN(gen, dis, z_shape=(batch_size//2, 1, 64, 64), num_gen_conditional=2, num_dis_conditional=1)


def connect_lapgans(*gans, train=True):
    for gan in gans:
        out = gan.get_output(train)


def generate_masks():
    pass


def gaussian_pyramid(input, nb_layers=5):
    pyramid = [input]
    for i in range(nb_layers):
        pyramid.append(downsample(pyramid[-1]))
    return pyramid


def lapacian_pyramid(gauss_pyramid):
    prev_layer = None
    lap_pyramid = []
    for layer in gauss_pyramid:
        if prev_layer is not None:
            lap_pyramid.append(prev_layer - upsample(layer))
        prev_layer = layer

    lap_pyramid.append(gauss_pyramid[-1])
    return lap_pyramid


def downsample_image(image, steps):
    downsampled = []
    curr_image = image
    for i in range(steps):
        curr_image = downsample(curr_image)
        downsampled.append(curr_image)
    downsampled.reverse()
    return downsampled


class MaskLAPGAN(AbstractModel):
    def __init__(self):
        self.names = ["alfa", "bravo", "charlie"]
        self.alfa = get_alfa()
        self.bravo = get_bravo()
        self.charlie = get_charlie()
        self.gans = [self.alfa, self.bravo, self.charlie]

    @staticmethod
    def bw_mask(mask):
        bw = T.zeros_like(mask, dtype=floatX)
        bw = T.set_subtensor(bw[(mask > MASK["IGNORE"]).nonzero()], 1.)
        return bw

    def compile(self, optimizer_factory, functions=['train', 'generate']):
        masks_idx = [T.tensor4("{}_mask_idx".format(name)) for name in self.names]

        masks_bw = [self.bw_mask(m) for m in masks_idx]
        x_real = T.tensor4("x_real")
        gaussian_pyr = gaussian_pyramid(x_real, nb_layers=3)
        lapacian_pyr = lapacian_pyramid(gaussian_pyr)
        alfa_losses = self.alfa.losses(gaussian_pyr[2], gen_conditional=[masks_bw[0]])
        alfa_image = self.alfa.g_out
        alfa_image_up = upsample(alfa_image)
        alfa_g_loss = alfa_losses[0] + mask_loss(masks_idx[0], alfa_image)
        alfa_g_updates = optimizer_factory().get_updates(
            self.alfa.G.params, self.alfa.G.constraints, alfa_g_loss)
        alfa_d_loss = alfa_losses[1]
        alfa_d_updates = optimizer_factory().get_updates(
            self.alfa.D.params, self.alfa.D.constraints, alfa_d_loss)

        gauss_16_up = upsample(gaussian_pyr[2])
        bravo_gen_images = T.concatenate([alfa_image_up, gauss_16_up])
        bravo_losses = self.bravo.losses(
            lapacian_pyr[1],
            gen_conditional=[masks_bw[1], alfa_image_up],
            dis_conditional=[bravo_gen_images])
        bravo_laplace = self.bravo.g_out
        bravo_image = bravo_laplace + alfa_image_up
        bravo_image_up = upsample(bravo_image)
        bravo_g_loss = bravo_losses[0] + mask_loss(masks_idx[1], bravo_image)
        bravo_g_updates = optimizer_factory().get_updates(
            self.bravo.G.params, self.bravo.G.constraints, bravo_g_loss)
        bravo_d_loss = bravo_losses[1]
        bravo_d_updates = optimizer_factory().get_updates(
            self.bravo.D.params, self.bravo.D.constraints, bravo_d_loss)

        gauss_32_up = upsample(gaussian_pyr[1])
        charlie_gen_images = T.concatenate([bravo_image_up, gauss_32_up])
        charlie_losses = self.charlie.losses(lapacian_pyr[0],
                                    gen_conditional=[masks_bw[2], bravo_image_up],
                                    dis_conditional=[charlie_gen_images])
        charlie_laplace = self.charlie.g_out
        charlie_image = charlie_laplace + bravo_image_up

        charlie_g_loss = charlie_losses[0] + mask_loss(masks_idx[2], charlie_image)
        charlie_g_updates = optimizer_factory().get_updates(
            self.charlie.G.params, self.charlie.G.constraints, charlie_g_loss)
        charlie_d_loss = charlie_losses[1]
        charlie_d_updates = optimizer_factory().get_updates(
            self.charlie.D.params, self.charlie.D.constraints, charlie_d_loss)

        if 'train' in functions:
            self._train_fn = theano.function(
                [x_real] + masks_idx,
                alfa_losses + bravo_losses + charlie_losses,
                updates=alfa_g_updates + alfa_d_updates + bravo_g_updates +
                    bravo_d_updates + charlie_g_updates + charlie_d_updates)

        if 'generate' in functions:
            self._generate = theano.function(
                masks_bw,
                [charlie_image]
            )
        if 'debug' in functions:
            self.debug_labels = ["mask_bw_16", "mask_bw_32", "mask_bw_64",
                      "gauss_64", "gauss_32", "gauss_16", "gauss_8",
                      "gauss_32_up", "gauss_16_up",
                      "laplace_64", "laplace_32",
                      "charlie_laplace", "bravo_laplace",
                      "charlie_img", "bravo_img", "alfa_img",
                      "bravo_img_up", "alfa_img_up"]

            self._debug = theano.function(
                [x_real] + masks_idx,
                masks_bw + gaussian_pyr +
                [gauss_32_up, gauss_16_up] +
                lapacian_pyr[:2] +
                [charlie_laplace, bravo_laplace] +
                [charlie_image, bravo_image, alfa_image] +
                [bravo_image_up, alfa_image_up]
            )

    def fit(self, X, masks_idx, batch_size=128, nb_epoch=100, verbose=0,
            callbacks=None,  shuffle=True):
        if callbacks is None:
            callbacks = []
        losses = ['{}_d_loss', '{}_d_real', '{}_d_gen', '{}_g_loss']
        labels = [loss.format(name) for name in self.names for loss in losses]

        def train(model, batch_ids, batch_index, batch_logs=None):
            if batch_logs is None:
                batch_logs = {}

            b = batch_index*batch_size//2
            e = (batch_index+1)*batch_size//2
            masks_idx_batch = [m[b:e] for m in masks_idx]
            inputs = [X[batch_ids]] + masks_idx_batch
            outs = self._train_fn(*inputs)
            for key, value in zip(labels, outs):
                batch_logs[key] = value

        assert batch_size % 2 == 0, "batch_size must be multiple of two."
        self._fit(train, len(X), batch_size=batch_size, nb_epoch=nb_epoch,
                  verbose=verbose, callbacks=callbacks, shuffle=shuffle, metrics=labels)

    def print_svg(self):
        theano.printing.pydotprint(self._g_train, outfile="train_g.png")
        theano.printing.pydotprint(self._d_train, outfile="train_d.png")

    def load_weights(self, fname):
        for name, gan in zip(self.names, self.gans):
            gan.G.load_weights(fname.format(name, "generator"))
            gan.D.load_weights(fname.format(name, "detector"))

    def save_weights(self, fname, overwrite=False):
        for name, gan in zip(self.names, self.gans):
            gan.G.save_weights(fname.format(name, "generator"), overwrite)
            gan.D.save_weights(fname.format(name, "detector"), overwrite)

    def generate(self, *conditional):
        return self._generate(*conditional)[0]

    def debug(self, X, masks_idx):
        inputs = [X] + masks_idx
        return self._debug(*inputs)


def np_bw_mask(mask_idx):
    bw = np.zeros_like(mask_idx)
    bw[mask_idx > MASK["IGNORE"]] = 1
    return bw


def masks(batch_size):
    generator = GridGenerator()
    artist = MaskGridArtist()
    for masks in gen_grids.batches(batch_size, generator, artist=artist,
                                   scales=[.25, .5, 1.]):
        yield [m.astype(np.float64) for m in masks[:3]]


def tags_from_hdf5(fname):
    f = h5py.File(fname)
    data = np.asarray(f["data"])
    print(data.shape)
    labels = np.asarray(f["labels"])
    tags = data[labels == 1]
    tags /= 255.
    print(tags.shape)
    return tags


def train(args):
    lapgan = MaskLAPGAN()

    X = tags_from_hdf5("/home/leon/uni/vision_swp/deeplocalizer_data/gan_data/gan/gan_0.hdf5")
    X /= 255.
    print("Compiling...")
    lapgan.compile(lambda: SGD(lr=0.02))
    print("Done Compiling")

    for masks_idx in itertools.islice(masks(len(X)), 1):
        print(len(masks_idx))
        lapgan.fit(X, masks_idx, verbose=1, nb_epoch=1)

    lapgan.save_weights("data/lapgan_{}_{}.hdf5", overwrite=True)


def draw(args):
    def encode64_png(np_array):
        output = io.BytesIO()
        scipy.misc.toimage(np_array).save(output, format='PNG')
        return base64.b64encode(output.getvalue())

    def make_image(elem: et.Element, np_array):
        elem.tag = 'image'
        arr_base64 = encode64_png(np_array)
        elem.set('xlink:href', "data:image/png;base64," +
                 arr_base64.decode('utf-8'))

    with open('lapgan.svg') as f:
        tree = et.parse(f)
    lapgan = MaskLAPGAN()

    X = tags_from_hdf5("/home/leon/uni/vision_swp/deeplocalizer_data/gan_data/gan/gan_0.hdf5")
    X /= 255.

    print("Compiling...")
    lapgan.compile(lambda: SGD(lr=0.02), functions=['debug'])
    print("Done Compiling")

    masks_idx = next(masks(64))
    outputs = lapgan.debug(X[:128], masks_idx)
    labels = lapgan.debug_labels
    for elem in tree.iter():
        id = elem.get('id')
        if id is not None:
            id = id.split("XXX")[0]
        if id in labels:
            index = labels.index(id)
            make_image(elem, outputs[index][3, 0])
    tree.write("py_lapgan.svg")


def test(args):
    lapgan = MaskLAPGAN()
    lapgan.load_weights("lapgan_{}_{}.hdf5")
    print("Compiling...")
    lapgan.compile(lambda: SGD(lr=0.02), functions=['generate'])
    print("Done Compiling")
    batch_size = 64
    os.makedirs("data/samples", exist_ok=True)
    for masks_idx in itertools.islice(masks(64), 1):
        masks_bw = [np_bw_mask(m) for m in masks_idx]
        images = lapgan.generate(*masks_bw)
        print(images.shape)
        for i in range(batch_size):
            imsave("data/samples/{}.png".format(i), images[i, 0])


if __name__ == "__main__":
    argparse = NetworkArgparser(train, test)
    parser_draw = argparse.subparsers.add_parser(
        "draw", help="draw the lapgan network")
    parser_draw.set_defaults(func=draw)
    argparse.parse_args()
