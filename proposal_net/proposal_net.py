#! /usr/bin/env python3
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
import argparse
import os
import random
from PIL import Image
import h5py
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers.convolutional import Convolution2D
from keras.layers.core import Layer, Dense, Activation, Flatten
from keras.models import Sequential
from keras.objectives import binary_crossentropy, epsilon
from keras.utils.theano_utils import floatX
import math
import skimage

import theano
import theano.tensor as T
from theano.tensor.nnet.conv import conv2d
import matplotlib.pyplot as plt
import numpy as np
import emcee


def hdf5_file_list(fname):
    with open(fname) as f:
        return [os.path.join(os.path.dirname(fname), line.rstrip('\n'))
                for line in f.readlines()]


def get_dataset(hdf5_file):
    f = h5py.File(hdf5_file, 'r')
    assert "data" in f.keys()
    assert "labels" in f.keys()
    # print("loaded dataset from {} with data shape: {} and labels shape: {}"
    #       .format(hdf5_file, f['data'].shape, f['labels'].shape))
    return f['data'], f['labels']


def saliency_error(y_true, y_pred):
    y_pred = T.clip(y_pred, epsilon, 1.0 - epsilon)
    base_loss = T.nnet.binary_crossentropy(y_pred, y_true)
    loss = T.set_subtensor(base_loss[T.eq(y_true, 1)],
                           base_loss[T.eq(y_true, 1)], inplace=False)
    return loss.mean(axis=-1)


def preprocess_img(data):
    data = floatX(data)
    data = data / 255
    return data.reshape((1, 1, data.shape[0], data.shape[1]))


def preprocess(data, labels):
    data = floatX(data) / 255
    # data = data / data.sum(axis=1)[:, np.newaxis]
    return data, floatX(labels)


# from: https://groups.google.com/forum/#!topic/theano-users/iDo3hkiZqko
def im2col(im, psize, n_channels=3):
    """Similar to MATLAB's im2col function.
    Args:
      im - a Theano tensor3, of the form <n_channels, height, width>.
      psize - an int specifying the (square) block size to use
      n_channels - the number of channels in im

    Returns: a 5-tensor of the form <patch_id_i, patch_id_j, n_channels, psize,
             psize>.
    """
    assert im.ndim == 3, "im must have dimension 3."
    im = im[:, ::-1, ::-1]
    res = T.zeros((n_channels, psize * psize, im.shape[1] - psize + 1,
                   im.shape[2] - psize + 1))
    filts = T.reshape(T.eye(psize * psize, psize * psize),
                      (psize * psize, psize, psize))
    filts = T.shape_padleft(filts).dimshuffle((1, 0, 2, 3))

    for i in range(n_channels):
        cur_slice = T.shape_padleft(im[i], n_ones=2)
        res = T.set_subtensor(res[i], T.nnet.conv.conv2d(cur_slice, filts)[0])

    return res.dimshuffle((0, 2, 3, 1)).reshape(
        (n_channels, im.shape[1] - psize + 1, im.shape[2] - psize + 1,
         psize, psize)).dimshuffle((1, 2, 0, 3, 4))


def sample_locations(saliency_map, random_selection_threshold=0.3):
    saliency_threshold = saliency_map > 0.9
    print("num pixels: {}".format(saliency_threshold.sum()))
    random_selection = np.random.sample(saliency_map.shape) < random_selection_threshold
    selected_indicies = (saliency_threshold & random_selection)
    return np.asarray(selected_indicies.nonzero())


def sample_locations_old(saliency_map, nwalkers=1000, walkers_per_iter=50):
    def lnprob(x):
        if x[0] < 0 or x[1] < 0:
            return float("-inf")
        try:
            prob = saliency_map[x[0], x[1]]
            if prob < 0.9:
                return float("-inf")
            else:
                return math.log(prob / saliency_sum)
        except IndexError:
            # TODO: Exceptions are slow
            return float("-inf")
    iterations = nwalkers // walkers_per_iter
    total_samples = walkers_per_iter*iterations
    ndim = 2
    all_pos = np.zeros((total_samples, ndim))
    for i in range(iterations):
        p0 = [[random.randint(0, saliency_map.shape[0]),
               random.randint(0, saliency_map.shape[1])]
              for _ in range(walkers_per_iter)]
        sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob)
        pos, prob, state = sampler.run_mcmc(p0, 50)
        all_pos[i*walkers_per_iter:(i+1)*walkers_per_iter, :] = pos
        for p in pos:
            rr, cc = skimage.draw.circle(p[0], p[1], 4)
            saliency_map[rr, cc] *= 0.8

    return all_pos


class SaliencyNetwork(object):
    ORIGINAL_TAG_SIZE = (60, 60)
    TAG_SIZE = (10, 10)

    def __init__(self, weight_file):
        self.weight_file = weight_file
        train_net = self.build_network(train=True)
        train_net.load_weights(weight_file)
        self.net = self.build_network(train=False)
        self.copy_network_weights(train_net, self.net)
        self._saliency_fn = self.build_saliency(self.net)

    def _resize_size(self, im_size):
        def scale_cooordinate(size, i):
            return int(size * self.TAG_SIZE[i] / self.ORIGINAL_TAG_SIZE[i])

        return scale_cooordinate(im_size[0], 0), \
           scale_cooordinate(im_size[1], 1)

    def saliency_file(self, fname):
        im = Image.open(fname)
        resized = im.resize(self._resize_size(im.size))
        preprocesed = preprocess_img(np.asarray(resized))
        return self.saliency_np(preprocesed), preprocesed

    def saliency_np(self, np_img):
        return self._saliency_fn(np_img)[0]

    @staticmethod
    def build_saliency(net):
        mode = theano.compile.get_default_mode()
        mode = mode.including('conv_fft')
        return theano.function([net.get_input(train=False)],
                               [net.get_output(train=False)], mode=mode)

    @staticmethod
    def build_network(train=True):
        net = Sequential()
        net.add(Convolution2D(5, 1, 3, 3, activation='relu'))
        net.add(Convolution2D(5, 5, 3, 3, activation='relu'))
        net.add(Convolution2D(1, 5, 6, 6, activation='sigmoid'))
        if train:
            net.add(Flatten())
        net.compile('adam', saliency_error)
        return net

    @staticmethod
    def copy_network_weights(train, deploy):
        for tr_layer, dp_layer in zip(train.layers, deploy.layers):
            if type(tr_layer) == Flatten:
                continue
            for tp, dp in zip(tr_layer.params, dp_layer.params):
                dp.set_value(tp.get_value())


def main(args):
    train_hdf5_files = hdf5_file_list(args.train)
    val_hdf5_files = hdf5_file_list(args.val)
    net = SaliencyNetwork.build_network(train=True)
    train_x,  train_y = preprocess(*get_dataset(train_hdf5_files[0]))
    val_x,  val_y = preprocess(*get_dataset(val_hdf5_files[0]))
    print(net.predict(train_x[:128]))
    net.fit(train_x, train_y, nb_epoch=5000, batch_size=128, verbose=1, shuffle=True,
            validation_data=(val_x, val_y),
            callbacks=[ModelCheckpoint("saliency_net_big.hdf5", save_best_only=True),
                       EarlyStopping(patience=5)])

    net.save_weights("saved_saliency_net_big.hdf5")
    dense = net.layers[0]
    weights = dense.params[0].get_value()
    plt.set_cmap('gray')
    plt.imshow(weights.reshape(10, 10))
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='train a simple proposal network')
    parser.add_argument('--train', type=str, help='path to train txt file with hdf5 files in it.')
    parser.add_argument('--val', type=str, help='path to validation txt file with hdf5 files in it.')
    # parser.add_argument('--test', type=str, help='path to test txt file with hdf5 files in it.')
    main(parser.parse_args())
