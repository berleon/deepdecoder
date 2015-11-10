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
from argparse import ArgumentParser
import os
import itertools
from deepdecoder import NUM_CELLS

from keras.preprocessing.image import ImageDataGenerator
import numpy as np

import deepdecoder.generate_grids as gen_grids
import deepdecoder.gt_grids as real_grids


class NetworkArgparser(object):
    def __init__(self, train_cb, test_cb):
        self.parser = ArgumentParser()
        self.subparsers = self.parser.add_subparsers()
        self.parser_train = self.subparsers.add_parser(
            "train", help="train the network")
        self.parser_train.add_argument("--weight-dir", default="weights",
                                      help="directory from where to load weights")
        self.parser_train.set_defaults(func=train_cb)
        self.parser_test = self.subparsers.add_parser(
            "test", help="test the network")
        self.parser_test.add_argument("--weight-dir", default="weights",
                                      help="directory from where to load weights")
        self.parser_test.set_defaults(func=test_cb)

    def parse_args(self):
        args = self.parser.parse_args()
        if hasattr(args, 'func'):
            args.func(args)
        else:
            self.parser.print_help()
            exit(1)


class GeneratedGridTrainer(object):
    def __init__(self):
        self.generator = gen_grids.GridGenerator()
        self.preprocessor = ImageDataGenerator(
            featurewise_center=True, featurewise_std_normalization=False)
        self.minibatch_size = 128
        self.minibatches_per_epoche = 12
        self.save_iter = 20
        self.use_preprocessor = True
        self.artist = gen_grids.BadGridArtist()
        self.gt_files = []

    def set_gt_path_file(self, fname):
        with open(fname, 'r') as f:
            self.gt_files = [l.rstrip('\n')for l in f.readlines()]

    def fit_data_gen(self):
        grids, _ = next(gen_grids.batches(batch_size=2048, generator=self.generator))
        grids = grids.astype(np.float32) / 255
        self.preprocessor.fit(grids)

    def save_weights(self, model, iteration, weight_dir="weights"):
        file = os.path.abspath("{}/{:0>4}_all_degrees_model.hdf5".format(weight_dir, iteration))
        print("saving weights to: " + file)
        model.save_weights(file)

    def preprocessed(self, grids, labels, batch_size):
        if self.use_preprocessor:
            return next(self.preprocessor.flow(grids, labels,
                                               batch_size=batch_size))
        else:
            return grids, labels

    def batches(self, batch_size=32):
        for raw_grids, labels in gen_grids.batches(batch_size, self.generator,
                                                   artist=self.artist):
            scaled = raw_grids.astype(np.float32)/255.
            yield self.preprocessed(scaled, labels, batch_size)

    def real_batches(self, batch_size=32):
        if not self.gt_files:
            raise ValueError("No ground truth files found! Aborting.")
        for raw_grids, raw_labels in real_grids.batches(
                self.gt_files, batch_size, repeat=False):
            scaled = raw_grids.astype(np.float32)/255.
            # swap 0 and 1 in labels
            labels = np.zeros_like(raw_labels)
            labels[raw_labels == 0] = 1
            if self.use_preprocessor:
                yield self.preprocessed(scaled, raw_labels, batch_size)
            else:
                yield scaled, labels

    def train(self, model, weight_dir):
        os.makedirs(weight_dir)
        bs = self.minibatches_per_epoche * self.minibatch_size
        for i, (grids, labels) in enumerate(self.batches(batch_size=bs)):
            print(i)
            if i % self.save_iter == 0:
                self.save_weights(model, i, weight_dir)
            model.fit(grids, labels, nb_epoch=1,
                      batch_size=self.minibatch_size, verbose=1)

    def real_test(self, model, n_epochs=10):
        self._test(model, self.real_batches, n_epochs)

    def test(self, model, n_epochs=10):
        self._test(model, self.batches, n_epochs)

    def _test(self, model, data_generator, n_epochs=10):
        def accuracy_str(accs):
            return "".join(["{:^7.4f}".format(a) for a in accs])

        bs = self.minibatches_per_epoche * self.minibatch_size
        mini_bs = self.minibatch_size
        n_total_rights = np.zeros(NUM_CELLS)
        n_total = 0
        for i, (grids, labels) in itertools.takewhile(
                lambda a: a[0] < n_epochs,
                enumerate(data_generator(batch_size=bs))):
            prediction = model.predict(grids, batch_size=mini_bs, verbose=0)
            print(prediction)
            bit_prediction = prediction > 0.5
            n_rights = np.zeros(NUM_CELLS)
            n_per_epoche = labels.shape[0]
            for j in range(n_per_epoche):
                n_right = (bit_prediction[j] == labels[j]).sum()
                for k in range(n_right):
                    n_rights[k] += 1
            n_total_rights += n_rights
            n_total += n_per_epoche
            accuracies = n_rights / n_per_epoche
            print("#{:<4}: {}".format(i, accuracy_str(accuracies)))

        accuracies = n_total_rights / n_total
        print("total: {}".format(accuracy_str(accuracies)))
