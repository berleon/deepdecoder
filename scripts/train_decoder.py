#! /usr/bin/env python
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

from deepdecoder.networks import get_decoder_model
from beras.callbacks import AutomaticLearningRateScheduler, HistoryPerBatch, \
    SaveModels
from keras.models import Model
from keras.optimizers import Adam
from keras.objectives import mse
from keras.regularizers import l1
from keras.engine.topology import Input
from beesgrid import NUM_MIDDLE_CELLS, NUM_CONFIGS, get_gt_files_in_dir
import numpy as np
import os
import h5py
import json
import argparse
import itertools
import theano.tensor as T
import keras.backend as K
from keras.callbacks import Callback
from deepdecoder.evaluate import GTEvaluator, denormalize_predict
import deepdecoder.blending_gan


def generator(fname, batch_size):
    f = h5py.File(fname)
    i = 0
    tags = f['fakes']
    score = f['discriminator']
    params = f['mask_gen_in']
    nb_samples = len(tags)
    nb_to_choose = 2*batch_size
    start = nb_samples - 20*batch_size
    nb_params = NUM_MIDDLE_CELLS + NUM_CONFIGS + 1
    for i in itertools.count(start=start, step=nb_to_choose):
        idx = i % (nb_samples - nb_to_choose)
        batch = slice(idx, idx+nb_to_choose)
        tags_batch = tags[batch]
        score_batch = score[batch]
        params_batch = params[batch, :nb_params]

        sort_idx = np.argsort(score_batch.reshape(-1))
        tags_selected = tags_batch[sort_idx[batch_size:]]
        params_selected = params_batch[sort_idx[batch_size:]]
        assert len(tags_selected) == batch_size, len(tags_selected)
        assert len(params_selected) == batch_size, len(params_selected)
        yield [[tags_selected], params_selected]


class ValidationCB(Callback):
    def __init__(self, gt_files, lecture=None):
        self.evaluator = GTEvaluator(gt_files)
        if lecture is None:
            lecture = deepdecoder.blending_gan.lecture()
        self.lecture = lecture

    def on_epoch_end(self, epoch, log={}):
        results = self.evaluator.evaluate(
            denormalize_predict(self.model, self.lecture))
        print(" - mhd: {}".format(results['mean_hamming_distance']))


def get_class_weights(nb_classes):
    class_weights = np.ones((nb_classes,), dtype=K.floatx())
    class_weights[:NUM_MIDDLE_CELLS+2] *= 30
    class_weights = class_weights / np.sum(class_weights) * nb_classes
    assert int(np.sum(class_weights)) == nb_classes, np.sum(class_weights)
    return class_weights


def weight_mse(class_weights):
    cw = T.as_tensor_variable(class_weights)

    def wrapper(y_true, y_pred):
        se = K.square(y_pred - y_true)
        return K.mean(se * cw.dimshuffle('x', 0), axis=-1)
    return wrapper


def main(nb_units, depth, nb_epoch, dense_units):
    h5_fname = "/home/leon/data/300.hdf5"
    batch_size = 64
    output_dir = "models/decoder_n{}_depth{}_dense{}_e{}/" \
        .format(nb_units, depth, "_".join(map(str, dense_units)), nb_epoch)
    gt_dir = '/home/leon/repos/deeplocalizer_data/images/season_2015'
    gt_files = get_gt_files_in_dir(gt_dir)
    validate_cb = ValidationCB(gt_files)
    print("Creating output dir: {}".format(output_dir))
    os.makedirs(output_dir)
    gen = generator(h5_fname, batch_size)
    nb_output = next(gen)[1].shape[1]
    print(nb_output)
    x = Input(shape=(1, 64, 64))
    decoded_params = get_decoder_model(x, nb_units=nb_units, depth=depth,
                                       nb_output=nb_output, dense=dense_units)
    decoder = Model(x, [decoded_params])

    with open(output_dir + 'decoder.json', 'w+') as f:
        f.write(decoder.to_json())

    optimizer = Adam()
    class_weights = get_class_weights(nb_output)
    decoder.compile(optimizer, weight_mse(class_weights))

    scheduler = AutomaticLearningRateScheduler(
        optimizer, 'loss', epoch_patience=4, min_improvment=0.0002)
    history = HistoryPerBatch()
    save = SaveModels({'{epoch:^03}_decoder.hdf5': decoder},
                      output_dir=output_dir)
    decoder.fit_generator(
        gen, samples_per_epoch=200*batch_size, nb_epoch=nb_epoch, verbose=1,
        callbacks=[scheduler, save, validate_cb])

    decoder.save_weights(output_dir + "decoder.hdf5", overwrite=True)

    with open(output_dir + 'history.json', 'w+') as f:
        json.dump(history.history, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Train the decoder generator network')
    parser.add_argument('--units', default=16, type=int,
                        help='number of units in the layer.')
    parser.add_argument('--depth', default=2, type=int,
                        help='number of conv layers between upsampling.')
    parser.add_argument('--epoch', default=400, type=int,
                        help='number of epoch to train.')
    parser.add_argument('--dense-units', default="1024, 256", type=str,
                        help='comma separated list')
    args = parser.parse_args()
    dense_units = [int(s) for s in args.dense_units.split(',')]
    main(args.units, args.depth, args.epoch, dense_units)
