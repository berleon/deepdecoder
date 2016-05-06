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
from deepdecoder.networks import mask_generator, mask_generator_extra, \
    mask_generator_all_conv
from deepdecoder.utils import zip_visualise_tiles

from beras.callbacks import AutomaticLearningRateScheduler, HistoryPerBatch, \
    SaveModels

from beras.util import collect_layers

from keras.models import Model
from keras.optimizers import Adam
from keras.objectives import mse
from keras.regularizers import l1
from keras.engine.topology import Input
import numpy as np
import os
import h5py
import json
import argparse
import pylab

pylab.rcParams['figure.figsize'] = (20, 20)


def generator(fname, batch_size):
    f = h5py.File(fname)
    i = 0
    tags = f['tags']
    params = f['params']
    depth_maps = f['depth_map']
    nb_samples = len(tags)
    while True:
        masks = tags[i:i+batch_size]
        depth_map = depth_maps[i:i+batch_size]
        yield [params[i:i+batch_size]], [masks, depth_map]
        i += batch_size
        if i >= nb_samples - batch_size:
            i = 0


def weight_mse(weight):
    def wrapper(x, y):
        return weight*mse(x, y)
    return wrapper


def main(nb_units, depth, nb_epoch, filter_size, project_factor, nb_dense):
    h5_fname = "/home/leon/data/generated_tags_variable_depth_map_normed.hdf5"
    batch_size = 32
    output_dir = "models/holy/mask_generator_var_with_depth_n{}_d{}_e{}/" \
        .format(nb_units, depth, nb_epoch)
    use_l1_regularizer = False
    os.makedirs(output_dir)

    gen = generator(h5_fname, batch_size)
    nb_input = next(gen)[0][0].shape[1]

    x = Input(shape=(nb_input,))
    mask, depth_map = mask_generator_all_conv(x, nb_units=nb_units, depth=depth,
                                           filter_size=filter_size)
    if use_l1_regularizer:
        layers = collect_layers(x, [mask, depth_map])
        for l in layers:
            if hasattr(l, 'W_regularizer'):
                l.W_regularizer = l1(0.001)
            if hasattr(l, 'b_regularizer'):
                l.b_regularizer = l1(0.001)
    g = Model(x, [mask, depth_map])
    optimizer = Adam()


    mask_more_weight = 5
    total_pixels = mask_more_weight*64**2 + 16**2
    weight_depth_map = 16**2 / total_pixels
    weight_mask = mask_more_weight*64**2 / total_pixels
    g.compile(optimizer, [weight_mse(weight_mask),
                          weight_mse(weight_depth_map)])

    scheduler = AutomaticLearningRateScheduler(
        optimizer, 'loss', epoch_patience=4, min_improvment=0.0002)
    history = HistoryPerBatch()
    save = SaveModels({'{epoch:^03}_mask_generator.hdf5': g}, output_dir=output_dir)
    g.fit_generator(gen, samples_per_epoch=200*batch_size,
                    nb_epoch=nb_epoch, verbose=1, callbacks=[scheduler, save])

    ins = next(generator(h5_fname, batch_size))
    predict_masks, predict_depth_map = g.predict(ins[0])
    masks, depth_map = ins[1]

    def clip(x):
        return np.clip(x, 0, 1)

    zip_visualise_tiles(clip(masks),
                        clip(predict_masks), show=False)
    plt.savefig(output_dir + "predict_tags.png")

    zip_visualise_tiles(clip(depth_map),
                        clip(predict_depth_map), show=False)
    plt.savefig(output_dir + "predict_depth_map.png")

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

    parser.add_argument('--depth', default=2, type=int,
                        help='number of conv layers between upsampling.')
    parser.add_argument('--epoch', default=400, type=int,
                        help='number of epoch to train.')

    parser.add_argument('--filter-size', default=3, type=int,
                        help='filter size of the conv kernel.')

    parser.add_argument('--project-factor', default=1, type=int,
                        help='factor of the last dense layer')

    parser.add_argument('--nb-dense', default="256,1024", type=str,
                        help='comma separated list')
    args = parser.parse_args()
    nb_dense = [int(s) for s in args.nb_dense.split(',')]
    main(args.units, args.depth, args.epoch, args.filter_size,
         args.project_factor, nb_dense)
