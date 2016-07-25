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

from deepdecoder.render_gan import RenderGAN, train, train_callbacks, load_tag3d_network, \
    save_real_images
import diktya.distributions
from diktya.func_api_helpers import get_hdf5_attr
import argparse
import sys
import pylab
import os

pylab.rcParams['figure.figsize'] = (16, 16)
sys.setrecursionlimit(10000)


def assert_dist_names_match(hdf5_fname):
    dist = diktya.distributions.load_from_json(
        get_hdf5_attr(hdf5_fname, 'distribution').decode('utf-8'))
    assert dist.names == [
        'bits',
        'z_rotation',
        'y_rotation',
        'x_rotation',
        'center',
        'radius',
        'inner_ring_radius',
        'middle_ring_radius',
        'outer_ring_radius',
        'bulge_factor',
        'focal_length',
    ]


def run(output_dir, tag3d_network_weights, real_hdf5_fname, force, nb_epoch,
        nb_gen_units, nb_dis_units):
    os.makedirs(output_dir, exist_ok=force)
    assert_dist_names_match(tag3d_network_weights)
    visualise_dir = os.path.join(output_dir, 'visualise')
    os.makedirs(visualise_dir, exist_ok=force)
    save_real_images(real_hdf5_fname, visualise_dir)

    tag3d_network = load_tag3d_network(tag3d_network_weights)
    labels_shape = tag3d_network.input_layers[0].batch_input_shape
    gan = RenderGAN(tag3d_network, labels_shape=labels_shape[1:])
    callbacks = train_callbacks(
        gan, output_dir, nb_visualise=20**2,
        real_hdf5_fname=real_hdf5_fname,
        hdf5_attrs={
            'distribution': get_hdf5_attr(tag3d_network_weights, 'distribution')
        })

    train(gan, real_hdf5_fname, output_dir, callbacks=callbacks,
          batch_size=16,
          nb_epoch=nb_epoch)


def main():
    parser = argparse.ArgumentParser(
        description='Train the mask blending gan')

    parser.add_argument('-r', '--real', type=str,
                        help='hdf5 file with the real data')
    parser.add_argument('--nntag3d', type=str,
                        help='weights of neuronal network trained on genenerated 3d tags')
    parser.add_argument('--gen-units', default=96, type=int,
                        help='number of units in the generator.')
    parser.add_argument('--dis-units', default=64, type=int,
                        help='number of units in the discriminator.')
    parser.add_argument('--epoch', default=301, type=int,
                        help='number of epoch to train.')
    parser.add_argument('-f', '--force', action='store_true',
                        help='override existing output files')
    parser.add_argument('--output-dir', type=str, help='output dir of the gan')
    args = parser.parse_args()

    run(args.output_dir, args.nntag3d, args.real, args.force,
        args.epoch, args.gen_units, args.dis_units)

if __name__ == "__main__":
    main()
