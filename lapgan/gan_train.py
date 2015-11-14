# coding: utf-8
import random
import operator
import os
import time
import itertools
from dotmap import DotMap

import h5py
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
from beras.gan import GAN, upsample
from beras.models import AbstractModel
from beras.layers.attention import RotationTransformer
from beras.util import downsample, blur

from deepdecoder.generate_grids import MASK, MASK_BLACK, MASK_WHITE, \
    GridGenerator, MaskGridArtist
import deepdecoder.generate_grids as gen_grids

from mask_loss import mask_loss

import GeneratedGridTrainer
from GeneratedGridTrainer import NetworkArgparser

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
    
    gen.add_node(Dense(2*n, activation='relu'), name='dense2', input='dense1')
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
            'train_decoder': self._compile_train_decoder,
            'train_mask_only': self._compile_train_mask_only,
        }

    def build_graph(self):
        mask_idx = T.tensor4("mask_idx")
        mask_bw = bw_mask(mask_idx)
        real = T.tensor4("real")
        mask_labels = T.matrix("mask_labels")

        gan_losses = self.gan.losses(real, gen_conditional=[mask_idx])
        d_in = self.gan.D.layers[0].input
        fake_image = self.gan.g_out
        fake_image_blur = blur(fake_image)
        mask_loss_dict = mask_loss(mask_idx, fake_image_blur)
        gan_mask_loss = mask_loss_dict['loss']

        decoder_in = self.decoder.get_input(train=False)
        decoder_out = self.decoder.get_output(train=False)

        self.decoder.layers[0].input = fake_image
        decoder_out_on_gan = self.decoder.get_output(train=True)
        decoder_rotation = self.decoder.layers[0].get_output(train=True)
        decoder_loss = objectives.mse(mask_labels, decoder_out_on_gan).mean()

        g_loss_on_d = gan_losses[0]
        g_loss_mask = gan_mask_loss + gan_losses[0]

        d_loss = gan_losses[1]

        return DotMap(locals())

    train_gan_labels = ['g_on_d', 'd', 'd_real', 'd_gen', "m", "g"]

    def _compile_train_gan(self, p, optimizer_factory, mode):
        d_updates = optimizer_factory("d").get_updates(
            self.gan.D.params, self.gan.D.constraints, p.d_loss)
        g_updates_mask = optimizer_factory("g").get_updates(
            self.gan.G.params, self.gan.G.constraints, p.g_loss_mask)
        self._train_gan = theano.function(
            [p.real, p.mask_idx],
            p.gan_losses + (p.gan_mask_loss, p.g_loss_mask),
            updates=g_updates_mask + d_updates,
            mode=mode)

    train_decoder_labels = ["decoder"]

    def _compile_train_decoder(self, p, optimizer_factory, mode):
        decoder_updates = optimizer_factory("decoder").get_updates(
            self.decoder.params, self.decoder.constraints, p.decoder_loss)
        self._train_decoder = theano.function(
            [p.mask_idx, p.mask_labels],
            [p.decoder_loss],
            updates=decoder_updates,
            mode=mode)

    train_mask_only_labels = ["m"]

    def _compile_train_mask_only(self, p, optimizer_factory, mode):
        g_updates_mask_only = optimizer_factory("g").get_updates(
            self.gan.G.params, self.gan.G.constraints, p.gan_mask_loss)
        self._train_mask_only = theano.function(
            [p.mask_idx], [p.gan_mask_loss],
            updates=g_updates_mask_only, mode=mode)

    def _compile_generate(self, p, optimizer_factory, mode):
        self._generate_labels = ["image"]
        self._generate = theano.function([p.mask_idx],
                                         [p.fake_image], mode=mode)

    debug_labels = ["real", "mask_bw", "image", "d_out", "d_in",
                    "mask_loss_per_sample", "black_white_loss",
                    "ring_loss", "cell_losses"] + train_gan_labels + \
                   ["predicted_labels", "decoder_rotation"]

    def _compile_debug(self, p, optimizer_factory, mode):
        ml = p.mask_loss_dict
        self._debug = theano.function(
            [p.real, p.mask_idx],
            [p.real, p.mask_bw, p.fake_image, self.gan.d_out,
             ml["loss_per_sample"],
             ml['black_white_loss'],
             ml['ring_loss'],
             T.stack(ml["cell_losses"]), ] + list(p.gan_losses) +
            [p.gan_mask_loss, p.g_loss_mask] +
            [p.decoder_out_on_gan, p.decoder_rotation],
            mode=mode)

    def compile(self, optimizer_factory, functions=None, mode=None):
        if functions is None:
            functions = ['train_gan']
        if functions == "all":
            functions = list(self.compile_functions.keys())

        for f in functions:
            assert f in self.compile_functions.keys(), "Unknown function: " + f

        p = self.build_graph()

        for f in functions:
            self.compile_functions[f](p, optimizer_factory, mode)

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
            "decoder": lambda: (self._train_decoder, self.train_decoder_labels),
            "mask_only": lambda: (self._train_mask_only, self.train_mask_only_labels),
            "gan_invert": lambda: (self._train_gan_invert, self.train_gan_invert_labels)
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

    def load_weights(self, fname):
        self.gan.G.load_weights("generator")
        self.gan.D.load_weights("detector")
        self.gan.D.load_weights("decoder")

    def save_weights(self, fname, overwrite=False):
        os.makedirs(fname.rstrip("{}.hdf5"), exist_ok=True)
        self.gan.G.save_weights(fname.format("generator"), overwrite)
        self.gan.D.save_weights(fname.format("detector"), overwrite)
        self.decoder.save_weights(fname.format("decoder"), overwrite)

    def decoder_predict(self, fake_image):
        return self._decoder_predict(fake_image)

    def generate(self, *conditional):
        return self._generate(*conditional)[0]

    def debug(self, X, mask_idx, mask_labels):
        outs = self._debug(X, mask_idx, mask_labels)
        return dict(zip(self.debug_labels, outs))


def masks(batch_size):
    mini_batch = 64
    batch_size += mini_batch - (batch_size % mini_batch)
    generator = GridGenerator()
    artist = MaskGridArtist()
    for masks in gen_grids.batches(batch_size, generator, artist=artist,
                                   with_gird_params=True, scales=[1.]):
        yield (masks[0].astype(floatX),) + masks[1:]


def tags_from_hdf5(fname):
    tags_list = []
    with open(fname) as pathfile:
        for hdf5_name in pathfile.readlines():
            hdf5_fname = os.path.dirname(fname) + "/" + hdf5_name.rstrip('\n')
            f = h5py.File(hdf5_fname)
            data = np.asarray(f["data"], dtype=np.float32)
            labels = np.asarray(f["labels"], dtype=np.float32)
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

    gan.compile(optimizer, functions=['train_gan'])
    print("Done Compiling in {0:.2f}s".format(time.time() - start))

    try:
        for i, (masks_idx, mask_labels, grid_params) in \
                enumerate(itertools.islice(masks(len(X)), 200)):
            print(i)
            gan.fit([X, masks_idx], direct_indexing=[False, True], verbose=1, nb_epoch=1,
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
