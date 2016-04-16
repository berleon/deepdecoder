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

import matplotlib
matplotlib.use('Agg')  # noqa
import matplotlib.pyplot as plt
from deepdecoder.networks import mask_generator
from deepdecoder.utils import zip_visualise_tiles

from beras.callbacks import AutomaticLearningRateScheduler, HistoryPerBatch
from beras.util import collect_layers

from keras.models import Model
from keras.optimizers import Adam
from keras.objectives import mse
from keras.regularizers import l1
from keras.engine.topology import Input
import time
import numpy as np
import os
import h5py
import json
import argparse


def generator(fname, batch_size):
    f = h5py.File(fname)
    i = 0
    tags = f['tags']
    params = f['params']
    nb_samples = len(tags)
    while True:
        yield [params[i:i+batch_size]], tags[i:i+batch_size]
        i += batch_size
        if i >= nb_samples - batch_size:
            i = 0


def main(nb_units, nb_epoch):
    h5_fname = "/home/leon/data/generated_tags_variable.hdf5"
    batch_size = 32
    output_dir = "models/holy/mask_generator_variable_n{}/".format(nb_units)
    use_l1_regularizer = False
    assert not os.path.exists(output_dir)
    gen = generator(h5_fname, batch_size)
    nb_input = next(gen)[0][0].shape[1]

    x = Input(shape=(nb_input,))
    out = mask_generator(x, nb_units=nb_units, dense_factor=3,
                         nb_dense_layers=2)
    if use_l1_regularizer:
        layers = collect_layers(x, out)
        for l in layers:
            if hasattr(l, 'W_regularizer'):
                l.W_regularizer = l1(0.001)
            if hasattr(l, 'b_regularizer'):
                l.b_regularizer = l1(0.001)
    g = Model(x, out)

    print("Compiling...")
    start = time.time()
    optimizer = Adam()
    g.compile(optimizer, mse)
    print("Done Compiling in {0:.2f}s".format(time.time() - start))

    scheduler = AutomaticLearningRateScheduler(
        optimizer, 'loss', epoch_patience=4, min_improvment=0.0002)
    history = HistoryPerBatch()
    g.fit_generator(gen, samples_per_epoch=200*batch_size,
                    nb_epoch=nb_epoch, verbose=1, callbacks=[scheduler])

    ins = next(gen)
    out = g.predict(ins[0])

    def clip(x):
        return np.clip(x, 0, 1)

    print(ins[1].shape)
    print(out.shape)
    zip_visualise_tiles(clip(ins[1]), clip(out), show=False)
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(output_dir + "predict.png")
    g.save_weights(output_dir + "mask_generator.hdf5", overwrite=True)
    with open(output_dir + 'mask_generator.json', 'w+') as f:
        f.write(g.to_json())

    with open(output_dir + 'history.json', 'w+') as f:
        json.dump(history.history, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Train the mask generator network')

    parser.add_argument('--units', default=16, type=int,
                        help='number of units in the layer.')
    parser.add_argument('--epoch', default=400, type=int,
                        help='number of epoch to train.')
    args = parser.parse_args()
    main(args.units, args.epoch)
