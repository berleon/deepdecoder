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

import os
import numpy as np
from diktya.func_api_helpers import load_model, get_hdf5_attr, predict_wrapper
from diktya.distributions import load_from_json
import h5py
import json
import pickle
from collections import OrderedDict
import time
from progressbar import ProgressBar
from pipeline.stages.processing import Decoder
from deepdecoder.data import DistributionHDF5Dataset
from deepdecoder.networks import ScaleInTestPhase, RandomSwitch
import click


def mse(x, y):
    return ((x - y)**2).mean(axis=-1)


def nth_bit_right_accuracy(bits_true, bits_pred, nth):
    accurate = np.sum(bits_true == bits_pred, axis=1) >= nth
    return accurate.mean()


def hamming_distance(y_true, y_pred):
    return (y_true != y_pred).sum(axis=-1)


def mean_hamming_distance(y_true, y_pred):
    nb_intermediate = np.sum(y_true == 0.5)
    hd = hamming_distance(y_true, y_pred)
    return (np.sum(hd) - nb_intermediate) / len(hd)


def get_marker(data_set):
    return os.path.splitext(os.path.basename(data_set))[0]


def get_predictions(tp: 'DecoderTraining', cache: bool, data_set, tags_name='tags'):
    def _calculate_predictions():
        from deepdecoder.scripts.train_decoder import save_samples
        name = get_marker(data_set)
        print("Loading GT data: {}".format(data_set))
        h5_truth = h5py.File(data_set)
        decoder = Decoder(tp.model_fname(), {
            'ScaleInTestPhase': ScaleInTestPhase,
            'RandomSwitch': RandomSwitch,
        })
        print("Loaded decoder. Got model with {:.2f} million parameters."
              .format(decoder.model.count_params() / 1e6))
        dist = decoder.distribution
        assert decoder.distribution == tp.get_label_distributions()
        fill_zeros = [(n, s) for (n, s) in dist.norm_nb_elems.items()
                      if n != 'bits']

        batch_size = 128
        gen_factory = tp.truth_generator_factory(h5_truth, fill_zeros, tags_name)
        tp.check_generator(gen_factory, os.path.basename(data_set), 400)
        tags, bits, _ = next(gen_factory(20**2))

        nb_bits = 12
        bits = np.array(bits[:12]).T
        normalize_bits = bits.min() == -1
        print(bits.min())
        print(bits.max())
        save_samples(tags[:, 0], bits,
                     tp.outname("evaluate_{}.png".format(name)))
        gen = gen_factory(batch_size)
        bits_true = []
        bits_pred = []
        total_time = 0
        nb_samples = 0
        total_samples = min(len(h5_truth[tags_name]), 100000)

        h5_fname = tp.outname("gt_prediction_{}.hdf5".format(name))
        if os.path.exists(h5_fname):
            os.remove(h5_fname)
        dset_output = DistributionHDF5Dataset(
            h5_fname, decoder.distribution, nb_samples=total_samples)
        with ProgressBar(max_value=total_samples) as pbar:
            for tags, gt, _ in gen:
                if nb_samples + len(tags) > total_samples:
                    break
                nb_samples += len(tags)
                bits = np.array(gt[:nb_bits]).T
                if normalize_bits:
                    bits = bits / 2. + 0.5
                bits_true.append(bits)
                start = time.time()
                outputs = decoder.predict(tags)
                dset_output.append(outputs, tag=tags)
                total_time += time.time() - start
                bits_pred.append(outputs['bits'] / 2. + 0.5)
                pbar.update(nb_samples)

        time_per_sample = total_time / nb_samples
        return np.concatenate(bits_true), np.concatenate(bits_pred), time_per_sample

    cache_path = tp.outname("predictions_on_gt.pickle")
    if os.path.exists(cache_path) and cache:
        print("Loading predictions from: {}".format(cache_path))
        with open(cache_path, 'rb') as f:
            bits_true, bits_pred_probs, time_per_sample = pickle.load(f)
    else:
        outs = _calculate_predictions()
        with open(cache_path, 'wb') as f:
            pickle.dump(outs, f)
        bits_true, bits_pred_probs, time_per_sample = outs

    return bits_true, bits_pred_probs, time_per_sample


def get_confidence(bits_true, bits_pred_probs):
    confidence_results = []
    confidence = np.min(np.abs(0.5 - bits_pred_probs), axis=1) * 2
    bits_pred = np.round(bits_pred_probs).astype(np.int32)
    for confidence_threshold in np.arange(0, 1, 0.002):
        eval_bits_pred = bits_pred[confidence >= confidence_threshold]
        eval_bits_true = bits_true[confidence >= confidence_threshold]
        results = OrderedDict([
            ('mean_hamming_distance', mean_hamming_distance(eval_bits_true, eval_bits_pred)),
            ('mse_id', mse(eval_bits_true, eval_bits_pred).mean())
        ])
        for i in range(1, 12 + 1):
            results['accuracy_{:02d}_bits_right'.format(i)] = \
                nth_bit_right_accuracy(eval_bits_true, eval_bits_pred, i)

        prop_samples = np.count_nonzero(confidence >= confidence_threshold) / confidence.shape[0]
        results['proportion_over_threshold'] = prop_samples
        results['confidence_threshold'] = confidence_threshold
        confidence_results.append(results)
    return confidence_results


def print_results(results):
    for name, value in sorted(results['confidence'][0].items()):
        print('{}: {}'.format(name, value))
    print("Time per sample: {:5f}ms".format(results['time_per_sample']*1e3))


def run(tp: 'DecoderTraining', cache: bool, data_set=None, tags_name='tags'):
    data_set = data_set or tp.gt_test_fname
    bits_true, bits_pred_probs, time_per_sample = get_predictions(tp, cache, data_set, tags_name)
    results = {
        'time_per_sample': time_per_sample,
        'bits_true': bits_true.tolist(),
        'bits_pred_probs': bits_pred_probs.tolist(),
        'mhd': mean_hamming_distance(bits_true, np.round(bits_pred_probs)),
        'confidence': get_confidence(bits_true, bits_pred_probs)
    }
    print_results(results)
    with open(tp.outname("results_{}.json".format(get_marker(data_set))), 'w') as f:
        json.dump(results, f)


@click.command("bb_evaluate_decoder")
@click.option("--cache", is_flag=True, help='use cache for predictions')
@click.option("--data-set", required=False,
              type=click.Path(dir_okay=False, exists=True, resolve_path=True),
              help='use cache for predictions')
@click.option("--tags-name", required=False, default='tags',
              type=str, help='name of the tags in the dataset')
@click.argument('dir', type=click.Path(file_okay=False, exists=True, resolve_path=True))
def main(cache, data_set, tags_name, dir):
    from deepdecoder.scripts.train_decoder import DecoderTraining

    with open(os.path.join(dir, 'training_params.json')) as f:
        config = json.load(f)
    dt = DecoderTraining(**config)
    print(tags_name)
    run(dt, cache, data_set, tags_name)

if __name__ == "__main__":
    main()
