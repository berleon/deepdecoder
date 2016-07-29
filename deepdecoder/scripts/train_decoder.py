#! /usr/bin/env python3
# Copyright 2016 Leon Sixt
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

import matplotlib
matplotlib.use('Agg')  # noqa

from deepdecoder.render_gan import RenderGAN, train, train_callbacks, load_tag3d_network
from deepdecoder.networks import decoder_resnet
from deepdecoder.data import DistributionHDF5Dataset
import argparse
import sys
import os
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, ModelCheckpoint
from diktya.callbacks import AutomaticLearningRateScheduler, HistoryPerBatch
sys.setrecursionlimit(10000)

param_labels = ('z_rot_sin', 'z_rot_cos', 'y_rot', 'x_rot', 'center_x', 'center_y')


def add_noise(tags):
    sigmas = np.clip(np.random.normal(loc=0.07, scale=0.02, size=tags.shape[0]),
                     np.finfo(float).eps, np.inf)
    noised = []
    for idx in range(tags.shape[0]):
        noise = np.random.normal(loc=0., scale=sigmas[idx], size=tags.shape[1:])
        noised.append(np.clip(tags[idx] + noise), -1, 1)
    return np.array(noised)


def augmentation_data_generator(data_generator, batch_size):
    augmentation = ImageDataGenerator(
        rotation_range=30.,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2*np.pi,
        zoom_range=0.2,
        channel_shift_range=0.3)

    augmentation.fit(next(data_generator(batch_size=2048))[0])

    for data, labels in data_generator(batch_size=batch_size):
        yield add_noise(next(augmentation.flow(data, data, batch_size=batch_size))[0]), labels


def dataset_iterator(dataset, batch_size):
    for batch in dataset.iter(batch_size):
        labels = batch.pop['labels']
        for name in labels.dtype.names:
            batch[name] = labels[name]
        yield batch


def _nb_samples_per_iterator(prefetch_batch_size, pos):
    n = len(pos)
    batch_size = prefetch_batch_size // n
    num_samples = [0] * n
    while sum(num_samples) != batch_size:
        not_empty = [k for k in range(n) if pos[k] != prefetch_batch_size]
        choices = np.random.choice(not_empty, batch_size - sum(num_samples))
        num_choosen = [sum(choices == k) for k in range(n)]
        num_samples = [min(num_samples[k] + choosen + pos[k], prefetch_batch_size) - pos[k]
                       for k, choosen in enumerate(num_choosen)]
    return num_samples


def zip_dataset_iterators(dataset_iters, batch_size):
    n = len(dataset_iters)
    prefetch_factor = 10
    prefetch_batch_size = batch_size * prefetch_factor
    iterators = [d_iter(prefetch_batch_size) for d_iter in dataset_iters]

    for batches in zip(*iterators):
        pos = [0] * n
        for i in range(prefetch_factor):
            num_samples = _nb_samples_per_iterator(prefetch_batch_size, pos)
            zipped_batch = {n: np.zeros((batch_size,) + a.shape[1:])
                            for n, a in batches[0].items()}
            zipped_pos = 0
            for i, batch in enumerate(batches):
                for name in batch.keys():
                    zipped_batch[name][zipped_pos:zipped_pos+num_samples[i]] = \
                        batch[name][pos[i]:pos[i]+num_samples[i]]
                pos[i] += num_samples[i]
            yield zipped_batch

def data_generator():
    pass


def run(dataset_fnames,
        gt_fname,
        force, nb_epoch, nb_units, output_dir):
    os.makedirs(output_dir, exist_ok=force)

    datasets = [DistributionHDF5Dataset(fname) for fname in dataset_fnames]
    dataset_iterators = [lambda bs: dataset_iterator(dset, bs) for dset in datasets]
    data_generator = lambda bs: zip_dataset_iterators(dataset_iterators, bs)

    bs = 128
    model = decoder_resnet()

    hist = HistoryPerBatch()
    stopper = EarlyStopping(monitor='val_loss', patience=25, verbose=0, mode='auto')
    scheduler = AutomaticLearningRateScheduler(model.optimizer, 'loss', epoch_patience=3, factor=0.25)
    checkpointer = ModelCheckpoint('decoder.hdf5', monitor='val_loss', verbose=0,
                                   save_best_only=True, mode='auto')

    nb_batches_per_epoch = 1024
    model.fit_generator(
        augmentation_data_generator(data_generator, bs),
        samples_per_epoch=bs*nb_batches_per_epoch, nb_epoch=1000,
        callbacks=[scheduler, checkpointer, stopper, hist],
        validation_data=truth_generator(bs), nb_val_samples=truth_tags.shape[0])

    hist.plot(save_as=os.path.join(output_dir, "loss.png"))


def main():
    parser = argparse.ArgumentParser(
        description='Train the decoder network')
    parser.add_argument('-r', '--real', type=str,
                        help='hdf5 files with the generated data')
    parser.add_argument('--units', default=96, type=int,
                        help='number of units in the decoder.')
    parser.add_argument('--epoch', default=300, type=int,
                        help='number of epoch to train.')
    parser.add_argument('-f', '--force', action='store_true',
                        help='override existing output files')
    parser.add_argument('output', type=str, help='file name where to save the model weights')
    args = parser.parse_args()

    run(args.output_dir, args.nntag3d, args.real, args.force,
        args.epoch, args.gen_units, args.dis_units)

if __name__ == "__main__":
    main()
