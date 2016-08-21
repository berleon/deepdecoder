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

import argparse
import os
import numpy as np
from diktya.func_api_helpers import load_model, get_hdf5_attr, predict_wrapper
from diktya.distributions import load_from_json
import h5py
import json
from deepdecoder.scripts.train_decoder import DecoderTraining
import pickle
from collections import OrderedDict
import time
from progressbar import ProgressBar

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


def get_model_params(model_fname):
    decoder = load_model(model_fname)
    return decoder.count_params()


def get_predictions(tp: DecoderTraining, no_cache: bool):
    def _calculate_predictions():
        print("Loading GT data: {}".format(tp.gt_fname))
        h5_truth = h5py.File(tp.gt_fname)
        decoder = load_model(tp.model_fname())
        print("Loaded decoder. Got model with {:.2f} million parameters."
              .format(decoder.count_params() / 1e6))
        dist = load_from_json(get_hdf5_attr(tp.model_fname(), 'distribution').decode('utf-8'))
        fill_zeros = [(n, s) for (n, s) in dist.norm_nb_elems.items()
                      if n != 'bits']
        batch_size = 32
        gen = tp.truth_generator_factory(h5_truth, fill_zeros)(batch_size)
        decoder._make_predict_function()
        predict = predict_wrapper(decoder.predict, decoder.output_names)
        bits_true = []
        bits_pred = []
        nb_bits = 12
        total_time = 0
        nb_samples = 0
        total_samples = len(h5_truth['tags'])
        with ProgressBar(max_value=total_samples) as pbar:
            while True:
                tags, gt = next(gen)
                if nb_samples + len(tags) > total_samples:
                    break
                nb_samples += len(tags)
                bits_true.append(np.array(gt[:nb_bits]).T)

                start = time.time()
                outputs = predict(tags, batch_size)
                total_time += time.time() - start
                bits = np.array([v.flatten() for n, v in outputs.items()
                                 if n.startswith("bit")])
                bits_pred.append(bits.T)
                pbar.update(nb_samples)
        time_per_sample = total_time / nb_samples
        return np.concatenate(bits_true), np.concatenate(bits_pred), time_per_sample

    cache_path = tp.outname("predictions_on_gt.pickle")
    if os.path.exists(cache_path) and not no_cache:
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


def run(tp: DecoderTraining, no_cache: bool):
    bits_true, bits_pred_probs, time_per_sample = get_predictions(tp, no_cache)
    results = {
        'time_per_sample': time_per_sample,
        'bits_true': bits_true.tolist(),
        'bits_pred_probs': bits_pred_probs.tolist(),
        'mhd': mean_hamming_distance(bits_true, np.round(bits_pred_probs)),
        'confidence': get_confidence(bits_true, bits_pred_probs)
    }
    print_results(results)
    with open(tp.outname("results.json"), 'w') as f:
        json.dump(results, f)


def main():
    def as_abs_path(x):
        if not os.path.exists(x):
            raise Exception("Path {} does not exists.".format(x))
        if not os.path.isabs(x):
            x = os.path.abspath(x)
        return x

    parser = argparse.ArgumentParser(
        description='Evaluate the decoder network')

    parser.add_argument('--no-cache', action='store_true',
                        help='recompute the predictions')
    parser.add_argument('dir', type=as_abs_path,
                        help='directory where the decoder is stored')
    args = parser.parse_args()
    with open(os.path.join(args.dir, 'training_params.json')) as f:
        config = json.load(f)
    dt = DecoderTraining(**config)
    run(dt, args.no_cache)

if __name__ == "__main__":
    main()
