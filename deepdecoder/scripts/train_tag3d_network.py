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

from deepdecoder.networks import tag3d_network_dense
from deepdecoder.data import DistributionHDF5Dataset

from diktya.callbacks import AutomaticLearningRateScheduler, HistoryPerBatch, \
    SaveModels
from diktya.numpy import zip_tile
from diktya.func_api_helpers import save_model

from keras.models import Model
from keras.optimizers import Adam, SGD, Nadam
from keras.engine.topology import Input

import numpy as np
import scipy.misc
import os
import json
import argparse
import seaborn as sns  # noqa


def run(output_dir, force, tags_3d_hdf5_fname, nb_units, depth,
        nb_epoch, filter_size, project_factor, nb_dense):
    batch_size = 64
    basename = "network_tags3d_n{}_d{}_e{}".format(nb_units, depth, nb_epoch)
    output_basename = os.path.join(output_dir, basename)

    tag_dataset = DistributionHDF5Dataset(tags_3d_hdf5_fname)
    tag_dataset._dataset_created = True
    print("Got {} images from the 3d model".format(tag_dataset.nb_samples))
    weights_fname = output_basename + ".hdf5"
    if os.path.exists(weights_fname) and not force:
        raise OSError("File {} already exists. Use --force to override it")
    elif os.path.exists(weights_fname) and force:
        os.remove(weights_fname)
    os.makedirs(output_dir, exist_ok=True)

    def generator(batch_size):
        for batch in tag_dataset.iter(batch_size):
            labels = []
            for name in batch['labels'].dtype.names:
                labels.append(batch['labels'][name])
            assert not np.isnan(batch['tag3d']).any()
            assert not np.isnan(batch['depth_map']).any()
            labels = np.concatenate(labels, axis=-1)
            yield labels, [batch['tag3d'], batch['depth_map']]

    labels = next(generator(batch_size))[0]
    print("labels.shape ", labels.shape)
    print("labels.dtype ", labels.dtype)
    nb_input = next(generator(batch_size))[0].shape[1]

    x = Input(shape=(nb_input,))
    tag3d, depth_map = tag3d_network_dense(x, nb_units=nb_units, depth=depth,
                                           nb_dense_units=nb_dense)
    g = Model(x, [tag3d, depth_map])
    # optimizer = SGD(momentum=0.8, nesterov=True)
    optimizer = Adam()

    g.compile(optimizer, loss=['mse', 'mse'], loss_weights=[1, 1/3.])

    scheduler = AutomaticLearningRateScheduler(
        optimizer, 'loss', epoch_patience=5, min_improvement=0.001)
    history = HistoryPerBatch()
    save = SaveModels({basename + '_snapshot_{epoch:^03}.hdf5': g}, output_dir=output_dir,
                      hdf5_attrs=tag_dataset.get_distribution_hdf5_attrs())
    history_plot = history.plot_callback(fname=output_basename + "_loss.png",
                                         every_nth_epoch=10)
    g.fit_generator(generator(batch_size), samples_per_epoch=800*batch_size,
                    nb_epoch=nb_epoch, verbose=1,
                    callbacks=[scheduler, save, history, history_plot])

    nb_visualize = 18**2
    vis_labels, (tags_3d, depth_map) = next(generator(nb_visualize))
    predict_tags_3d, predict_depth_map = g.predict(vis_labels)

    def zip_and_save(fname, *args):
        clipped = list(map(lambda x: np.clip(x, 0, 1)[:, 0], args))
        print(clipped[0].shape)
        tiled = zip_tile(*clipped)
        print(tiled.shape)
        scipy.misc.imsave(fname, tiled)

    zip_and_save(output_basename + "_predict_tags.png", tags_3d, predict_tags_3d)
    zip_and_save(output_basename + "_predict_depth_map.png", depth_map, predict_depth_map)

    save_model(g, weights_fname, attrs=tag_dataset.get_distribution_hdf5_attrs())
    with open(output_basename + '.json', 'w+') as f:
        f.write(g.to_json())

    with open(output_basename + '_loss_history.json', 'w+') as f:
        json.dump(history.history, f)

    fig, _ = history.plot()
    fig.savefig(output_basename + "_loss.png")
    print("Saved weights to: {}".format(weights_fname))


def main():
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
    parser.add_argument('-f', '--force', action='store_true',
                        help='override existing output files')
    parser.add_argument('-t', '--3d-tags', type=str,
                        help='path to the hdf5 file with the generated tags')
    parser.add_argument('output', type=str, help='output directory')

    args = parser.parse_args()
    nb_dense = [int(s) for s in args.nb_dense.split(',')]
    tags_3d_hdf5_fname = args.__dict__['3d_tags']
    assert os.path.exists(tags_3d_hdf5_fname)

    run(args.output, args.force, tags_3d_hdf5_fname, args.units, args.depth, args.epoch,
        args.filter_size, args.project_factor, nb_dense)


if __name__ == "__main__":
    main()
