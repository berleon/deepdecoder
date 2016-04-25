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

from beesgrid import gt_grids, NUM_MIDDLE_CELLS, CONFIG_LABELS, CONFIG_CENTER
import numpy as np


def evaluate_decoder(decoder):
    pass


def hamming_distance(x, y):
    return (x != y).sum(axis=-1)


def nth_bit_right_accuracy(bits_true, bits_pred, nth):
    accurate = np.sum(bits_true == bits_pred, axis=1) >= nth
    return accurate.mean()


def mse(x, y):
    return ((x - y)**2).mean(axis=-1)


class GTEvaluator:
    def __init__(self, gt_files):
        self.gt_files = gt_files
        self.batch_size = 32

    def evaluate(self, predict, bit_threshold=0.5):
        gt_images, bits_true, configs_true = \
            next(gt_grids(self.gt_files, all=True, center='zero'))
        nb_samples = len(gt_images)
        bits_pred = []
        configs_pred = []
        for i in range(0, nb_samples, self.batch_size):
            to = min(i + self.batch_size, nb_samples)
            batch_bits_pred, batch_configs_pred = predict(gt_images[i:to])
            threshold_arr = bit_threshold * np.ones_like(batch_bits_pred)
            batch_bits_pred = batch_bits_pred >= threshold_arr
            batch_bits_pred = batch_bits_pred.astype(np.float32)
            bits_pred.append(batch_bits_pred)
            configs_pred.append(batch_configs_pred)

        bits_pred = np.concatenate(bits_pred, axis=0)
        configs_pred = np.concatenate(configs_pred, axis=0)
        results = {
            'mean_hamming_distance': hamming_distance(
                bits_true, bits_pred).mean(),
            'mse_id': mse(bits_true, bits_pred).mean()
        }
        for i in range(1, NUM_MIDDLE_CELLS + 1):
            results['accuracy_{}_bits_right'.format(i)] = \
                nth_bit_right_accuracy(bits_true, bits_pred, i)

        for i, label in enumerate(CONFIG_LABELS):
            name = 'mse_{}'.format(label)
            results[name] = mse(configs_pred[:, i], configs_true[:, i]).mean()
        return results
