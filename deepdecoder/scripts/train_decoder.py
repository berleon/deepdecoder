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
from deepdecoder.networks import decoder_resnet, decoder_stochastic_wrn, decoder_dummy, \
    decoder_baseline
from deepdecoder.data import DistributionHDF5Dataset, get_distribution_hdf5_attrs
from deepdecoder.augmentation import config as handmade_augmentation_config, \
    stack_augmentations as stack_handmade_augmentations, needed_datanames

import collections
import os
import numpy as np
from keras.callbacks import Callback
from keras.optimizers import SGD
import keras.backend as K
from diktya.callbacks import LearningRateScheduler, HistoryPerBatch, \
    SaveModelAndWeightsCheckpoint, OnEpochEnd, DotProgressBar
from diktya.numpy import image_save, tile
from diktya.preprocessing.image import NoiseAugmentation, WarpAugmentation, \
    CropAugmentation, random_std, chain_augmentations, HistEqualization, \
    ChannelScaleShiftAugmentation
import h5py
from skimage.exposure import equalize_hist
import time
import json
import yaml
import seaborn as sns
import deepdecoder.scripts.evaluate_decoder as evaluate_decoder
import click
import sys


nb_bits = 12


def filter_by_discriminator_score(batch, threshold):
    d_score = batch['discriminator']
    goodenough = d_score.reshape((-1,)) >= threshold
    if len(goodenough) == 0:
        return None
    for name, arr in batch.items():
        if len(goodenough) == 1:
            batch[name] = arr[goodenough, np.newaxis]
        else:
            batch[name] = arr[goodenough]
    return batch


def dataset_iterator(dataset, batch_size, names=None,
                     d_threshold=0.05, shuffle=False):
    def len_batch(b):
        return len(next(iter(b.values())))

    def nb_cached():
        return sum([len_batch(b) for b in cache])

    cache = []
    for batch in dataset.iter(2*batch_size, names, shuffle=shuffle):
        labels = batch.pop('labels')
        for name in labels.dtype.names:
            batch[name] = labels[name]
        if 'discriminator' in batch:
            batch = filter_by_discriminator_score(batch, d_threshold)
            if batch is None:
                continue
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


def zip_dataset_iterators(dataset_iters, batch_size, iter_weights=None):
    if iter_weights is None:
        iter_weights = np.ones((len(dataset_iters))) / len(dataset_iters)
    if type(iter_weights) is not np.ndarray:
        assert(isinstance(iter_weights, collections.Iterable))
        iter_weights = np.array(list(iter_weights))
    assert(np.isclose(np.sum(iter_weights), 1.))

    # largest remainder method (number of samples must exactly equal batch_size)
    iter_samples = iter_weights * batch_size
    fractionals, iter_samples_floor = np.modf(iter_samples)
    num_missing = batch_size - np.sum(iter_samples_floor).astype(np.int)
    ceil_indices = np.argpartition(1 - fractionals, num_missing - 1)[:num_missing]
    iter_samples_floor[ceil_indices] += 1
    iter_samples = iter_samples_floor.astype(np.int)
    assert(np.sum(iter_samples) == batch_size)

    iters = [it(bs) for it, bs in zip(dataset_iters, iter_samples)]
    while True:
        inputs, labels, label_masks = zip(*[next(it) for it in iters])
        inputs = np.concatenate(inputs)
        labels = [np.concatenate(l) for l in zip(*labels)]
        label_masks = [np.concatenate(m) for m in zip(*label_masks)]
        yield inputs, labels, label_masks


def bit_split(generator):
    for batch in generator:
        bits = batch.pop('bits')
        bs, nb_bits = bits.shape

        if bits.min() == -1:
            bits = (bits + 1) / 2.
        for i in range(nb_bits):
            batch['bit_{}'.format(i)] = bits[:, i]
        yield batch


def truth_generator(h5_truth, batch_size,
                    missing_label_sizes, tags_name='tags'):
    print("tags_name", tags_name)
    truth_tags = h5_truth[tags_name]
    truth_bits = (np.array(h5_truth['bits'], dtype=np.float) + 1.) / 2.

    nb_bits = truth_bits.shape[-1]
    idx = 0
    crop = CropAugmentation(0, (64, 64))
    while True:
        if idx + batch_size > truth_tags.shape[0]:
            idx = 0
        tags = crop(truth_tags[idx:idx+batch_size])
        labels = [truth_bits[idx:idx+batch_size][:, i] for i in range(nb_bits)]
        missing_labels = [np.zeros((batch_size, size), dtype=np.float32)
                          for _, size in missing_label_sizes]
        label_mask = [np.ones((l.shape[0], ), dtype=np.float32) for l in labels] + \
                     [np.zeros((l.shape[0], ), dtype=np.float32) for l in missing_labels]
        yield tags, labels + missing_labels, label_mask
        idx += batch_size


def hist_equalisation(generator):
    for data, labels in generator:
        eq_imgs = []
        for img in data:
            eq_imgs.append(2*equalize_hist(img) - 1)
        yield np.array(eq_imgs), labels


class CollectBitsLoss(Callback):
    @staticmethod
    def _collect_loss(logs, start_marker):
        total_loss = 0
        nb_bits = 0
        for name, loss in logs.items():
            if name.startswith(start_marker):
                nb_bits += 1
                total_loss += loss
        return total_loss / nb_bits

    def on_batch_end(self, batch, logs={}):
        logs['bits_loss'] = self._collect_loss(logs, 'bit_')

    def on_epoch_end(self, epoch, logs={}):
        logs['val_bits_loss'] = self._collect_loss(logs, 'val_bit_')


def save_samples(data, bits, outfname):
    print("bits", bits.shape)
    nb_bits = bits.shape[1]
    for i in range(len(data)):
        s = 64 // nb_bits
        for j in range(nb_bits):
            data[i, :s, s*j:s*(j+1)] = 2*bits[i, j] - 1
    image_save(outfname, np.clip(tile(data), -1, 1))


class DecoderTraining:
    def __init__(self, config, default=None):
        self.config = config
        self.default = default or DecoderTraining.default()
        missing = []
        for name, default_value in self.default.items():
            if default_value is None and name not in config:
                missing.append(name)
        if missing:
            raise Exception("Following keys are missing {}."
                            .format(", ".join(missing)))
        wrong_keys = []
        for name in config.keys():
            if name not in self.default:
                wrong_keys.append(name)

        if wrong_keys:
            raise Exception("Following keys are given but not found in default {}."
                            .format(", ".join(wrong_keys)))
        self._label_distribution = None

    @staticmethod
    def default():
        """
        Defaults:
            gt_train (path): hdf5 files with the train real ground truth data.
            gt_test (path): hdf5 files with the test real ground truth data.
            gt_val (path): hdf5 files with the validation real ground truth data.
            train_set (path): hdf5 files with the train set.
            test_set (path): hdf5 files with the artificial test set.
            nb_units (int):number of nb_units in the decoder. (default ``16``)
            epoch (int): number of epoch to train. (default ``120``)
            model (str): decoder model type. either resnet or stochastic_wrn (default ``resnet``).
            data_name (str): get the images from this dataset (default ``fake``).
            marker (str): marker for the output dir. (default ``""`` )
            discriminator_threshold (float): only use sample over this threshold. (default ``0.02``)
            make-json (path): save bb_make file under this path.
            use_combined_data (bool): whether to use ground truth train data
            output_dir (path): directory where to save the model weights.
        """
        return {
            'train_sets': None,
            'test_set': None,
            'gt_train_fname': None,
            'gt_val_fname': None,
            'gt_test_fname': None,
            'nb_epoch': 120,
            'nb_units': 16,
            'nb_batches_per_epoch': 1000,
            'batch_size': 128,
            'use_hist_equalization': False,
            'use_warp_augmentation': False,
            'use_diffeomorphism_augmentation': False,
            'use_noise_augmentation': False,
            'use_handmade_augmentation': False,
            'use_channel_scale_shift_augmentation': False,
            'augmentation_rotation': 0.1 * np.pi,
            'augmentation_scale': (0.8, 1.2),
            'augmentation_shear': (-0.2, 0.2),
            'augmentation_channel_scale': (0.8, 1.5),
            'augmentation_channel_shift': (-0.5, 0.5),
            'augmentation_noise_mean': 0.07,
            'augmentation_noise_std': 0.05,
            'augmentation_diffeomorphism': [(4, 0.3), (8, 0.7), (16, 0.5), (32, 0.5)],
            'handmade_augmentation': None,
            'discriminator_threshold': 0.02,
            'decoder_model': 'resnet',
            'data_name': None,
            'shuffle_data': False,
            'use_combined_data': False,
            'output_dir': None,
            'marker': "",
            'verbose': 0,
        }

    @staticmethod
    def from_config(fname):
        return DecoderTraining.from_yaml_config(fname)

    @staticmethod
    def from_yaml_config(fname):
        with open(fname) as f:
            config = yaml.load(f)
            return DecoderTraining(config, DecoderTraining.default())

    def __getitem__(self, name):
        if name not in self.default:
            raise Exception("Unknown key name {}.".format(name))

        return self.config.get(name, self.default[name])

    def __getattr__(self, name):
        return self[name]

    def get_config(self):
        return {
            'config': self.config,
            'default': self.default,
        }

    def get_model(self, label_output_sizes):
        optimizer = SGD(momentum=0.9, nesterov=True)
        if self.decoder_model == "resnet":
            return decoder_resnet(label_output_sizes,
                                  nb_filter=self.nb_units,
                                  resnet_depth=[3, 6, 4, 3],
                                  optimizer=optimizer)
        elif self.decoder_model == "stochastic_wrn":
            return decoder_stochastic_wrn(label_output_sizes,
                                          optimizer=optimizer)
        elif self.decoder_model == "dummy":
            return decoder_dummy(label_output_sizes, nb_filter=self.nb_units,
                                 optimizer=optimizer)
        elif self.decoder_model == "baseline":
            return decoder_baseline(label_output_sizes, nb_filter=self.nb_units,
                                    optimizer=optimizer)
        else:
            raise Exception("Expected resnet, stochastic_wrn, baseline or dummy for decoder_model. "
                            "Got {}.".format(self.decoder_model))

    def model_fname(self):
        return self.outname("decoder.model")

    def save(self):
        fname = self.outname('training_params.json')
        with open(fname, 'w+') as f:
            json.dump(self.get_config(), f, indent=2)

    def get_label_distributions(self):
        datasets = [DistributionHDF5Dataset(fname) for fname in self.train_sets]
        if self._label_distribution is None:
            for dset in datasets:
                if self._label_distribution is None:
                    self._label_distribution = dset.get_tag_distribution()
                else:
                    if self._label_distribution != dset.get_tag_distribution():
                        raise Exception("Distribution of datasets must match")
        return self._label_distribution

    def get_label_output_sizes(self):
        dist = self.get_label_distributions()
        label_output_sizes = [(n, size) for n, size in dist.norm_nb_elems.items()
                              if n != 'bits']
        return label_output_sizes

    def outname(self, *args):
        return os.path.join(self.output_dir, *args)

    def get_handmade_augmentation(self):
        c = handmade_augmentation_config.configure(self.handmade_augmentation)
        augment = stack_handmade_augmentations(self.data_name, c)
        return augment

    def augmentation(self):
        augmentations = []
        if self.use_warp_augmentation and self.use_diffeomorphism_augmentation:
            aug_warp = WarpAugmentation(
                rotation=(-self.augmentation_rotation, self.augmentation_rotation),
                scale=self.augmentation_scale,
                shear=self.augmentation_shear,
                diffeomorphism=self.augmentation_diffeomorphism,
            )
            augmentations.append(aug_warp)
        elif not self.use_diffeomorphism_augmentation and self.use_warp_augmentation:
            aug_warp = WarpAugmentation(
                rotation=(-self.augmentation_rotation, self.augmentation_rotation),
                scale=self.augmentation_scale,
                shear=self.augmentation_shear,
            )
            augmentations.append(aug_warp)
        elif self.use_diffeomorphism_augmentation and not self.use_warp_augmentation:
            aug_warp = WarpAugmentation(
                diffeomorphism=self.augmentation_diffeomorphism
            )
            augmentations.append(aug_warp)
        if self.use_channel_scale_shift_augmentation:
            augmentations.append(
                ChannelScaleShiftAugmentation(
                    self.augmentation_channel_scale,
                    self.augmentation_channel_shift)
            )
        if self.use_noise_augmentation:
            aug_noise = NoiseAugmentation(
                random_std(self.augmentation_noise_mean,
                           self.augmentation_noise_std))
            augmentations.append(aug_noise)

        aug_crop = CropAugmentation(0, (64, 64))
        augmentations.append(aug_crop)

        if self.use_hist_equalization:
            augmentations.append(HistEqualization())

        return chain_augmentations(*augmentations)

    def summary(self):
        def print_key(key, pad=20):
            val = self.config[key]
            fmt_str = "    {:<" + str(pad) + "}: {}"
            print(fmt_str.format(key, val))

        header = '#' * 30 + " - Decoder Summary - " + '#' * 30
        print(header)
        for key in sorted(self.config.keys()):
            print_key(key)
        print('#' * len(header))

    def iterator_data_names(self):
        if self.use_handmade_augmentation:
            return needed_datanames(self.data_name)
        else:
            return [self.data_name]

    def data_generator_factory(self):
        datasets = [DistributionHDF5Dataset(fname) for fname in self.train_sets]
        dist = None
        for dset in datasets:
            if dist is None:
                dist = dset.get_tag_distribution()
            else:
                if dist != dset.get_tag_distribution():
                    raise Exception("Distribution of datasets must match")
        label_output_sizes = self.get_label_output_sizes()
        all_label_names = ['bit_{}'.format(i) for i in range(12)] + \
            [n for n, _ in label_output_sizes]
        dataset_names = ['labels'] + self.iterator_data_names()
        print("Used datasets: " + str(dataset_names))
        if 'discriminator' in list(dset.keys()):
            dataset_names.append('discriminator')
        dataset_iterators = [lambda bs: bit_split(dataset_iterator(
            dset, bs, dataset_names, self.discriminator_threshold, self.shuffle_data))
            for dset in datasets]

        augmentation = self.augmentation()
        handmade_augmentation = self.get_handmade_augmentation()

        def wrapper(iterator):
            def data_gen(bs, with_batch=False):
                for batch in iterator(bs):
                    data = batch[self.data_name]
                    labels = [batch[l] for l in all_label_names]
                    label_mask = [np.ones(l.shape[0], dtype=np.float32) for l in labels]
                    if self.use_handmade_augmentation:
                        data = handmade_augmentation(batch)

                    to_yield = [augmentation(data), labels, label_mask]
                    if with_batch:
                        to_yield.append(batch)
                    yield to_yield
            return data_gen

        return lambda bs: zip_dataset_iterators(list(map(wrapper, dataset_iterators)), bs)

    def truth_generator_factory(self, h5_truth, missing_label_sizes, tags_name='tags'):
        def wrapper(bs):
            gen = truth_generator(h5_truth, bs, missing_label_sizes, tags_name)
            if self.use_hist_equalization:
                return hist_equalisation(gen)
            else:
                return gen
        return wrapper

    def combined_generator_factory(self, h5_truth, missing_label_sizes, tags_name='tags'):
        gan_data = self.data_generator_factory()
        truth_data = self.truth_generator_factory(h5_truth, missing_label_sizes, tags_name)

        return lambda bs: zip_dataset_iterators((gan_data, truth_data), bs, [.95, .05])

    def check_generator(self, gen, name, plot_samples=100):
        x, y, _ = next(gen(plot_samples))
        fig, ax = plt.subplots()
        sns.distplot(x.flatten(), ax=ax)
        fig.savefig(self.outname("data_histogram_{}.png".format(name)))
        plt.close(fig)

    def train(self):
        marker = os.path.basename(self.output_dir)
        os.makedirs(self.output_dir, exist_ok=True)
        self.save()
        self.summary()
        # save samples
        gt_val = h5py.File(self.gt_val_fname)
        print("bits", gt_val["bits"].shape)
        nb_vis_samples = 20**2

        gt_train = h5py.File(self.gt_train_fname)

        label_output_sizes = self.get_label_output_sizes()

        if self.use_combined_data:
            data_gen = self.combined_generator_factory(gt_train, label_output_sizes)
        else:
            data_gen = self.data_generator_factory()

        truth_gen_only_bits = self.truth_generator_factory(gt_val, missing_label_sizes=[])
        self.check_generator(data_gen, "train")
        self.check_generator(truth_gen_only_bits, "test")
        vis_out = next(data_gen(nb_vis_samples))
        vis_bits = np.array(vis_out[1][:nb_bits]).T
        save_samples(vis_out[0][:, 0], vis_bits,
                     self.outname(marker + "_train_samples.png"))
        gt_data, gt_bits, gt_masks = next(truth_gen_only_bits(nb_vis_samples))
        gt_bits = np.array(gt_bits).T
        print("gt_data", gt_data.shape, gt_data.min(), gt_data.max())
        print("gt_bits", gt_bits.shape, gt_bits.min(), gt_bits.max())
        save_samples(gt_data[:, 0], gt_bits,
                     self.outname(marker + "_val_samples.png"))
        # build model
        bs = self.batch_size
        model = self.get_model(label_output_sizes)
        # setup training
        hist = HistoryPerBatch(self.output_dir, extra_metrics=['bits_loss', 'val_bits_loss'])
        hist_saver = OnEpochEnd(lambda e, l: hist.save(), every_nth_epoch=5)

        def lr_schedule(optimizer):
            lr = K.get_value(optimizer.lr)
            return {
                40: lr / 10.,
            }

        scheduler = LearningRateScheduler(
            model.optimizer, lr_schedule(model.optimizer))
        hdf5_attrs = get_distribution_hdf5_attrs(self.get_label_distributions())
        hdf5_attrs['decoder_uses_hist_equalization'] = self.use_hist_equalization
        checkpointer = SaveModelAndWeightsCheckpoint(
            self.model_fname(), monitor='val_bits_loss',
            verbose=0, save_best_only=True,
            hdf5_attrs=hdf5_attrs)
        plot_history = hist.plot_callback(
            fname=self.outname(marker + '_loss.png'),
            metrics=['bits_loss', 'val_bits_loss'])
        # train
        truth_gen = self.truth_generator_factory(gt_val, label_output_sizes)
        callbacks = [CollectBitsLoss(), scheduler, checkpointer, hist,
                     plot_history, hist_saver]
        if int(self.verbose) == 0:
            callbacks.append(DotProgressBar())

        model.fit_generator(
            data_gen(bs),
            samples_per_epoch=bs*self.nb_batches_per_epoch, nb_epoch=self.nb_epoch,
            callbacks=callbacks,
            verbose=self.verbose,
            validation_data=truth_gen(bs),
            nb_val_samples=gt_val['tags'].shape[0],
            nb_worker=1, max_q_size=4*10, pickle_safe=False
        )
        evaluate_decoder.run(self, cache=False)


def get_output_dir(output_dir, config):
    def is_set(key):
        return key in config and config[key]

    name = "decoder_{}_{}_n{}".format(
        config['decoder_model'], config['data_name'], config['nb_units'])
    if is_set('use_warp_augmentation'):
        name += "_aug"
    if is_set('use_noise_augmentation'):
        name += "_noise"
    if is_set('use_hist_equalization'):
        name += "_hist_eq"

    if is_set('use_diffeomorphism_augmentation'):
        name += "_diff"
    if is_set('use_channel_scale_shift_augmentation'):
        name += "_chncs"
    if is_set('marker'):
        name += "_" + config['marker']
    name += time.strftime("_%Y-%m-%dT%H:%M:%S", time.gmtime())

    return os.path.abspath(os.path.join(output_dir, name))


@click.command('bb_train_decoder')
@click.option('--make-json', required=False, type=click.Path(),
              help='save bb_make file.')
@click.option('--save-default', required=False, type=click.Path(),
              help='save bb_make file.')
@click.option('--gt-train', type=click.Path())
@click.option('--gt-test', type=click.Path())
@click.option('--gt-val', type=click.Path())
@click.option('--train-sets', type=click.Path(), multiple=True)
@click.option('--test-set', type=click.Path())
@click.option('--create-output-dir-in', type=click.Path(resolve_path=True))
@click.argument('config', required=False, type=click.Path(dir_okay=False, exists=True))
def main(make_json, save_default, gt_train, gt_test, gt_val, train_sets, test_set,
         create_output_dir_in, config):
    config_fname = config
    if save_default:
        with open(save_default, 'w') as f:
            yaml_str = yaml.dump(DecoderTraining.default(),
                                 indent=2, default_flow_style=False)
            print("Default yaml is:")
            print('-' * 80)
            print(yaml_str)
            print('-' * 80)
            print("Saved to {}".format(save_default))
            f.write(yaml_str)
        sys.exit(0)

    if not config_fname:
        raise Exception("config argument can only be omitted, if --save-default is given.")

    with open(config_fname) as f:
        config = yaml.load(f)

    config_options = [("gt_train", "gt_train_fname"),
                      ("gt_test", "gt_test_fname"),
                      ("gt_val", "gt_val_fname"),
                      ("train_sets", "train_sets"),
                      ("test_set", "test_set")]

    for flag_name, config_name in config_options:
        value = locals()[flag_name]
        if value is not None:
            config[config_name] = value

    if create_output_dir_in:
        config['output_dir'] = get_output_dir(create_output_dir_in, config)

    dt = DecoderTraining(config)

    dt.train()
    if make_json:
        with open(make_json, 'w') as f:
            json.dump({"path": dt.output_dir}, f)


if __name__ == "__main__":
    main()
