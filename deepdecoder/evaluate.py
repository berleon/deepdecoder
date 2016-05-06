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
from scipy.ndimage.filters import gaussian_filter1d
from collections import OrderedDict


def evaluate_decoder(decoder):
    pass


def hamming_distance(y_true, y_pred):
    return (y_true != y_pred).sum(axis=-1)


def mean_hamming_distance(y_true, y_pred):
    nb_intermediate = np.sum(y_true == 0.5)
    print(nb_intermediate)
    hd = hamming_distance(y_true, y_pred)
    return (np.sum(hd) - nb_intermediate) / len(hd) # (len(hd) - nb_intermediate)


def nth_bit_right_accuracy(bits_true, bits_pred, nth):
    accurate = np.sum(bits_true == bits_pred, axis=1) >= nth
    return accurate.mean()


def mse(x, y):
    return ((x - y)**2).mean(axis=-1)


def denormalize_predict(model, lecture):
    def wrapper(x):
        outs = model.predict(2*x - 1)
        ids = outs[:, :NUM_MIDDLE_CELLS]
        config = outs[:, NUM_MIDDLE_CELLS:]
        config_denorm = lecture.denormalize_config(config)
        config_denorm[:, CONFIG_CENTER] = 0
        return lecture.denormalize_ids(ids), config_denorm
    return wrapper


class GTEvaluator:
    def __init__(self, gt_files, blur_images=True):
        self.gt_files = gt_files
        self.batch_size = 32
        self.gt_images, self.bits_true, self.configs_true = next(gt_grids(
            gt_files, all=True, center='zero'))
        if blur_images:
            self.gt_files = gaussian_filter1d(self.gt_images, 2/3, axis=-1)
            self.gt_files = gaussian_filter1d(self.gt_images, 2/3, axis=-2)
        self.nb_samples = len(self.gt_images)

    def evaluate(self, predict, bit_threshold=0.5):
        bits_pred = []
        configs_pred = []
        for i in range(0, self.nb_samples, self.batch_size):
            to = min(i + self.batch_size, self.nb_samples)
            batch_bits_pred, batch_configs_pred = predict(self.gt_images[i:to])
            threshold_arr = bit_threshold * np.ones_like(batch_bits_pred)
            batch_bits_pred = batch_bits_pred >= threshold_arr
            batch_bits_pred = batch_bits_pred.astype(np.float32)
            bits_pred.append(batch_bits_pred)
            configs_pred.append(batch_configs_pred)

        bits_pred = np.concatenate(bits_pred, axis=0)
        configs_pred = np.concatenate(configs_pred, axis=0)
        results = OrderedDict([
            ('mean_hamming_distance', mean_hamming_distance(
                self.bits_true, bits_pred)),
            ('mse_id', mse(self.bits_true, bits_pred).mean())
        ])
        for i in range(1, NUM_MIDDLE_CELLS + 1):
            results['accuracy_{:02d}_bits_right'.format(i)] = \
                nth_bit_right_accuracy(self.bits_true, bits_pred, i)

        for i, label in enumerate(CONFIG_LABELS):
            name = 'mse_{}'.format(label)
            results[name] = mse(configs_pred[:, i],
                                self.configs_true[:, i]).mean()

        return {k: float(v) for k, v in results.items()}
