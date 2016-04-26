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
from beesgrid import NUM_MIDDLE_CELLS
import numpy as np
import os
import h5py
import json
import argparse
import itertools


def generator(fname, batch_size):
    f = h5py.File(fname)
    i = 0
    tags = f['fakes']
    score = f['discriminator']
    params = f['mask_gen_in']
    nb_samples = len(tags)
    nb_to_choose = 2*batch_size
    for i in itertools.count(start=0, step=nb_to_choose):
        idx = (i + nb_to_choose) % nb_samples
        batch = slice(idx, idx+nb_to_choose)
        tags_batch = tags[batch]
        score_batch = score[batch]
        params_batch = params[batch]
        sort_idx = np.argsort(score_batch.reshape(-1))
        tags_selected = tags_batch[sort_idx][batch_size:]
        params_selected = params_batch[sort_idx][batch_size:]
        yield [[tags_selected], params_selected]


def get_class_weights(nb_classes):
    class_weights = np.ones((nb_classes,))
    class_weights[:NUM_MIDDLE_CELLS] *= 10
    class_weights = class_weights / np.linalg.norm(class_weights) * nb_classes
    return list(class_weights)


def main(nb_units, depth, nb_epoch, dense_units):
    h5_fname = "/mnt/storage/leon/models/gan_blending_300/samples/300.hdf5"
    batch_size = 64
    output_dir = "models/decoder_n{}_depth{}_dense{}_e{}/" \
        .format(nb_units, depth, "_".join(map(str, dense_units)), nb_epoch)
    print("Creating output dir: {}".format(output_dir))
    os.makedirs(output_dir)
    gen = generator(h5_fname, batch_size)
    nb_output = next(gen)[1].shape[1]
    print(nb_output)
    x = Input(shape=(1, 64, 64))
    decoded_params = get_decoder_model(x, nb_units=nb_units, depth=depth,
                                       nb_output=nb_output, dense=dense_units)
    decoder = Model(x, [decoded_params])
    optimizer = Adam()

    decoder.compile(optimizer, 'mse')

    scheduler = AutomaticLearningRateScheduler(
        optimizer, 'loss', epoch_patience=4, min_improvment=0.0002)
    history = HistoryPerBatch()
    save = SaveModels({'{epoch:^03}_decoder.hdf5': decoder},
                      output_dir=output_dir)
    class_weights = get_class_weights(nb_output)
    print(class_weights)
    decoder.fit_generator(gen, samples_per_epoch=200*batch_size,
                          nb_epoch=nb_epoch, verbose=1,
                          class_weight=class_weights,
                          callbacks=[scheduler, save])

    decoder.save_weights(output_dir + "decoder.hdf5", overwrite=True)
    with open(output_dir + 'decoder.json', 'w+') as f:
        f.write(decoder.to_json())

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
