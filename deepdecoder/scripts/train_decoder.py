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

import matplotlib.pyplot as plt
from deepdecoder.render_gan import RenderGAN, train, train_callbacks, load_tag3d_network
from deepdecoder.networks import decoder_resnet
from deepdecoder.data import DistributionHDF5Dataset, get_distribution_hdf5_attrs
import argparse
import sys
import os
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, ModelCheckpoint, Callback
from diktya.callbacks import AutomaticLearningRateScheduler, HistoryPerBatch, \
    SaveModelAndWeightsCheckpoint
from diktya.numpy import scipy_gaussian_filter_2d, image_save, tile
import h5py
from skimage.exposure import equalize_hist
import time
import json


param_labels = ('z_rot_sin', 'z_rot_cos', 'y_rot', 'x_rot', 'center_x', 'center_y')


def add_noise(tags):
    sigmas = np.clip(np.random.normal(loc=0.07, scale=0.02, size=tags.shape[0]),
                     np.finfo(float).eps, np.inf)
    noised = []
    for idx in range(tags.shape[0]):
        noise = np.random.normal(loc=0., scale=sigmas[idx], size=tags.shape[1:])
        noised.append(np.clip(tags[idx] + noise, -1, 1))
    return np.array(noised)


def augmentation_data_generator(data_generator, data_name, label_names,
                                noise=True):
    augmentation = ImageDataGenerator(
        rotation_range=0.1,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2*np.pi,
        zoom_range=0.2,
        channel_shift_range=0.3
    )

    # augmentation.fit(next(data_generator(2048))[data_name])

    def wrapper(batch_size):
        for batch in data_generator(batch_size):
            data = batch[data_name]
            augmented = next(augmentation.flow(data, batch_size=batch_size,
                                               shuffle=False))
            labels = []
            for name in label_names:
                labels.append(batch[name])
            if noise:
                augmented = add_noise(augmented)

            yield augmented, labels
    return wrapper


def dataset_iterator(dataset, batch_size, names=None, d_threshold=0.05):
    def len_batch(b):
        return len(b['discriminator'])

    def nb_cached():
        return sum([len_batch(b) for b in cache])

    cache = []
    for batch in dataset.iter(2*batch_size, names):
        labels = batch.pop('labels')
        d_score = batch['discriminator']
        goodenough = d_score.reshape((-1,)) >= d_threshold
        if len(goodenough) == 0:
            continue
        for name in labels.dtype.names:
            batch[name] = labels[name]

        for name, arr in batch.items():
            if len(goodenough) == 1:
                batch[name] = arr[goodenough, np.newaxis]
            else:
                batch[name] = arr[goodenough]
        assert len(batch['discriminator'].shape) == 2
        cache.append(batch)

        if nb_cached() >= batch_size:
            batch = {name: np.zeros((batch_size,) + arr.shape[1:])
                     for name, arr in cache[0].items()}
            new_cache = []
            i = 0
            for cached_batch in cache:
                nb_elems = min(batch_size - i, len_batch(cached_batch))
                new_batch = {}
                for name, arr in cached_batch.items():
                    batch[name][i:i+nb_elems] = arr[i:i+nb_elems]
                    if nb_elems < len_batch(cached_batch):
                        new_batch[name] = arr[i+nb_elems:]
                if new_batch:
                    new_cache.append(new_batch)
                i += nb_elems
                if i == batch_size:
                    yield batch
                    cache = new_cache
                    break


def _nb_samples_per_iterator(prefetch_batch_size, batch_size, pos):
    n = len(pos)
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
            num_samples = _nb_samples_per_iterator(prefetch_batch_size, batch_size, pos)
            zipped_batch = {n: np.zeros((batch_size,) + a.shape[1:])
                            for n, a in batches[0].items()}
            zipped_pos = 0
            for i, batch in enumerate(batches):
                for name in batch.keys():
                    zipped_batch[name][zipped_pos:zipped_pos+num_samples[i]] = \
                        batch[name][pos[i]:pos[i]+num_samples[i]]
                pos[i] += num_samples[i]
                zipped_pos += num_samples[i]
            assert zipped_pos == batch_size
            yield zipped_batch


def bit_split(generator):
    for batch in generator:
        bits = batch.pop('bits')
        bs, nb_bits = bits.shape
        for i in range(nb_bits):
            batch['bit_{}'.format(i)] = (bits[:, i] + 1) / 2.
        yield batch


def truth_generator(h5_truth, batch_size,
                    missing_label_sizes,
                    use_blur=False):
    def int_id_to_binary(id):
        result = np.zeros(12, dtype=np.int)
        a = 11
        while id:
            result[a] = id & 1
            id >>= 1
            a -= 1
        return result

    truth_tags = h5_truth['tags']
    truth_labels = h5_truth['labels']
    idx = 0
    while True:
        if idx + batch_size > truth_tags.shape[0]:
            idx = 0
        tags = (truth_tags[idx:idx+batch_size] / 255.)
        if use_blur:
            tags = scipy_gaussian_filter_2d(tags, 2/3, axis=-1)
        tags = tags * 2 - 1

        labels = list(np.array([int_id_to_binary(i)[::-1]
                                for i in truth_labels[idx:idx+batch_size]]).T)
        missing_labels = [np.zeros((batch_size, size), dtype=np.float32)
                          for _, size in missing_label_sizes]
        yield tags[:, np.newaxis], labels + missing_labels
        idx += batch_size


def hist_equalisation(generator):
    for data, labels in generator:
        eq_imgs = []
        for img in data:
            eq_imgs.append(2*equalize_hist(img) - 1)
        yield np.array(eq_imgs), labels


class PlotHistory(Callback):
    def __init__(self, history_per_batch, outfile):
        self.history_per_batch = history_per_batch
        self.outfile = outfile
        self.val_hist = {}

    def on_epoch_end(self, epoch, log={}):
        fig, ax = plt.subplots()
        for name, loss in log.items():
            if name.startswith('val_bit_'):
                if name not in self.val_hist:
                    self.val_hist[name] = []
                self.val_hist[name].append(loss)

        val_bit_loss = np.mean(list(self.val_hist.values()), axis=0)
        ax.plot(np.arange(1, 1 + len(val_bit_loss)), val_bit_loss, label='val_bits')
        self.history_per_batch.history['bits_loss'] = np.mean([
            hist for n, hist in self.history_per_batch.history.items()
            if n.startswith('bit_')], axis=0)
        self.history_per_batch.plot(metrics=['bits_loss'], fig=fig, axes=ax)
        fig.savefig(self.outfile)
        plt.close(fig)


def save_samples(data, bits, outfname):
    nb_bits = len(bits[0])
    for i in range(len(data)):
        s = 64 // nb_bits
        for j in range(nb_bits):
            data[i, :s, s*j:s*(j+1)] = 2*bits[i, j] - 1
    image_save(outfname, tile(data))


class DecoderTraining:
    def __init__(self, dataset_fnames, gt_fname, force, nb_epoch, nb_units,
                 use_augmentation, use_noise, data_name, output_dir):
        self.dataset_fnames = dataset_fnames
        self.gt_fname = gt_fname
        self.force = force
        self.nb_epoch = nb_epoch
        self.nb_units = nb_units
        self.use_augmentation = use_augmentation
        self.use_noise = use_noise
        self.data_name = data_name
        self.output_dir = output_dir

    def to_config(self):
        return {
            'dataset_fnames':  self.dataset_fnames,
            'gt_fname':  self.gt_fname,
            'force':  self.force,
            'nb_epoch':  self.nb_epoch,
            'nb_units':  self.nb_units,
            'use_augmentation':  self.use_augmentation,
            'use_noise':  self.use_noise,
            'data_name':  self.data_name,
            'output_dir':  self.output_dir,
        }

    def save(self):
        fname = os.path.join(self.output_dir, 'training_params.json')
        with open(fname, 'w+') as f:
            json.dump(self.to_config(), f, indent=2)

    def summary(self):
        header = '#' * 30 + " - Decoder Summary - " + '#' * 30
        print(header)
        print("   Training sets:    {}".format(self.dataset_fnames[0]))
        for fname in self.dataset_fnames[1:]:
            print("                     {}".format(fname))
        print("   GT data:          {}".format(self.gt_fname))
        print("   dataset name:     {}".format(self.data_name))
        print("   output dir:       {}".format(self.output_dir))
        print("   max epochs:       {}".format(self.nb_epoch))
        print("   nb units:         {}".format(self.nb_units))
        print("   use augmentation: {}".format(self.use_augmentation))
        print("   use noise:        {}".format(self.use_noise))
        print('#' * len(header))


def run(training_param):
    tp = training_param
    os.makedirs(tp.output_dir, exist_ok=tp.force)
    tp.save()
    tp.summary()
    h5_truth = h5py.File(tp.gt_fname)

    datasets = [DistributionHDF5Dataset(fname) for fname in tp.dataset_fnames]
    dist = None

    for dset in datasets:
        if dist is None:
            dist = dset.get_tag_distribution()
        else:
            if dist != dset.get_tag_distribution():
                raise Exception("Distribution of datasets must match")

    label_output_sizes = [(n, size) for n, size in dist.norm_nb_elems.items()
                          if n != 'bits']
    all_label_names = ['bit_{}'.format(i) for i in range(12)] + \
        [n for n, _ in label_output_sizes]
    dataset_names = ['labels', 'discriminator', tp.data_name]
    dataset_iterators = [lambda bs: bit_split(dataset_iterator(dset, bs, dataset_names)) for dset in datasets]
    data_generator = lambda bs: zip_dataset_iterators(dataset_iterators, bs)

    aug_gen = augmentation_data_generator(data_generator, tp.data_name, all_label_names)
    nb_vis_samples = 20**2
    vis_out = next(hist_equalisation(aug_gen(nb_vis_samples)))
    nb_bits = 12
    vis_bits = np.array(vis_out[1][:nb_bits]).T
    save_samples(vis_out[0][:, 0], vis_bits, os.path.join(tp.output_dir, "train_samples.png"))

    gt_data, gt_bits = next(hist_equalisation(
        truth_generator(h5_truth, nb_vis_samples, missing_label_sizes=[])))
    gt_bits = np.array(gt_bits).T
    save_samples(gt_data[:, 0], gt_bits, os.path.join(tp.output_dir, "test_samples.png"))

    bs = 128
    model = decoder_resnet(label_output_sizes, nb_filter=tp.nb_units,
                           resnet_depth=[3, 6, 4, 3], optimizer='sgd')
    # model.summary()
    hist = HistoryPerBatch(tp.output_dir)
    stopper = EarlyStopping(monitor='val_loss', patience=25, verbose=0, mode='auto')
    scheduler = AutomaticLearningRateScheduler(
        model.optimizer, 'loss', epoch_patience=3, min_improvement=0.2,
        factor=0.25)
    checkpointer = SaveModelAndWeightsCheckpoint(
        os.path.join('decoder.hdf5'), monitor='val_loss', verbose=0,
        save_best_only=True, hdf5_attrs=get_distribution_hdf5_attrs(dist))
    nb_batches_per_epoch = 1000
    plot_history = PlotHistory(hist, os.path.join(tp.output_dir, 'loss.png'))

    model.fit_generator(
        hist_equalisation(aug_gen(bs)),
        samples_per_epoch=bs*nb_batches_per_epoch, nb_epoch=tp.nb_epoch,
        callbacks=[scheduler, checkpointer, stopper, hist, plot_history],
        validation_data=hist_equalisation(truth_generator(h5_truth, bs, label_output_sizes)),
        nb_val_samples=h5_truth['tags'].shape[0],
        nb_worker=1, max_q_size=4*10, pickle_safe=False
    )


def get_output_dir(output_dir, units, augmentation, noise, dataset_name):
    t = time.gmtime()
    name = "decoder_{}_n{}".format(dataset_name, units)
    if augmentation:
        name += "_aug"
    if noise:
        name += "_noise"
    name += time.strftime("_%Y-%m-%dT%H:%MZ", time.gmtime())

    return os.path.join(output_dir, name)


def main():
    def as_abs_path(x):
        if not os.path.exists(x):
            raise Exception("Path {} does not exists.".format(x))
        if not os.path.isabs(x):
            x = os.path.abspath(x)
        return x

    parser = argparse.ArgumentParser(
        description='Train the decoder network')
    parser.add_argument('-g', '--gt', type=as_abs_path,
                        help='hdf5 files with the real groud truth data')
    parser.add_argument('-t', '--train-set', nargs='+', type=as_abs_path,
                        help='hdf5 files with the artificial train set')
    parser.add_argument('--units', default=16, type=int,
                        help='number of units in the decoder.')
    parser.add_argument('--epoch', default=1000, type=int,
                        help='number of epoch to train.')
    parser.add_argument('--dataset-name', default='fake', type=str,
                        help='get the images from this dataset.')
    parser.add_argument('-a', '--augmentation', action='store_true',
                        help='use augmentation.')
    parser.add_argument('-n', '--noise', action='store_true',
                        help='add gaussian noise to the images.')
    parser.add_argument('-f', '--force', action='store_true',
                        help='override existing output files')
    parser.add_argument('output_dir', type=as_abs_path,
                        help='directory where to save the model weights')
    args = parser.parse_args()
    print(args.train_set)
    output_dir = get_output_dir(
        args.output_dir, args.units, args.augmentation, args.noise, args.dataset_name)

    dt = DecoderTraining(args.train_set, args.gt, args.force, args.epoch, args.units,
                         args.augmentation, args.noise, args.dataset_name, output_dir)
    run(dt)

if __name__ == "__main__":
    main()
