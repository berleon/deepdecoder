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

import matplotlib.pyplot as plt
from deepdecoder.render_gan import RenderGAN, train, train_callbacks, load_tag3d_network
from deepdecoder.data import get_distribution_hdf5_attrs
import argparse
import os
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from diktya.func_api_helpers import load_model, get_hdf5_attr, predict_wrapper
from diktya.numpy import scipy_gaussian_filter_2d, image_save, tile
from diktya.distributions import load_from_json
import h5py
from skimage.exposure import equalize_hist
import json
from deepdecoder.scripts.train_decoder import truth_generator, hist_equalisation,\
    DecoderTraining


param_labels = ('z_rot_sin', 'z_rot_cos', 'y_rot', 'x_rot', 'center_x', 'center_y')


def hamming_distance(y_true, y_pred):
    return (y_true != y_pred).sum(axis=-1)


def mean_hamming_distance(y_true, y_pred):
    nb_intermediate = np.sum(y_true == 0.5)
    print(nb_intermediate)
    hd = hamming_distance(y_true, y_pred)
    print(hd)
    return (np.sum(hd) - nb_intermediate) / len(hd)


def run(training_param: DecoderTraining):
    tp = training_param
    print("Loading GT data: {}".format(tp.gt_fname))
    h5_truth = h5py.File(tp.gt_fname)
    decoder = load_model(tp.model_fname())
    dist = load_from_json(get_hdf5_attr(tp.model_fname(), 'distribution').decode('utf-8'))
    fill_zeros = [(n, s) for (n, s) in dist.norm_nb_elems.items()
                  if n != 'bits']
    batch_size = 32
    gen = hist_equalisation(truth_generator(h5_truth, batch_size, fill_zeros))
    predict = predict_wrapper(decoder.predict, decoder.output_names)
    gt_bits = []
    pr_bits = []
    nb_bits = 12

    nb_samples = 0
    while True:
        tags, gt = next(gen)
        nb_samples += len(tags)
        if nb_samples > len(h5_truth['tags']):
            break
        gt_bits.append(np.array(gt[:nb_bits]).T)
        outputs = predict(tags, batch_size)
        bits = np.array([v.flatten() for n, v in outputs.items()
                         if n.startswith("bit")])
        pr_bits.append(bits.T)
        # print(pr_bits[-1].shape)
        # print(gt_bits[-1].shape)
        # print(pr_bits[-1][0])
        # print(np.round(pr_bits[-1][0]).astype(np.int))
        # print(gt_bits[-1][0])
    gt_bits = np.concatenate(gt_bits)
    print(gt_bits.shape)
    pr_bits = np.concatenate(pr_bits)
    print(pr_bits.shape)
    print(mean_hamming_distance(gt_bits, np.round(pr_bits)))


def main():
    def as_abs_path(x):
        if not os.path.exists(x):
            raise Exception("Path {} does not exists.".format(x))
        if not os.path.isabs(x):
            x = os.path.abspath(x)
        return x

    parser = argparse.ArgumentParser(
        description='Evaluate the decoder network')
    parser.add_argument('dir', type=as_abs_path,
                        help='directory where the decoder is stored')
    args = parser.parse_args()
    with open(os.path.join(args.dir, 'training_params.json')) as f:
        config = json.load(f)
    dt = DecoderTraining(**config)
    run(dt)

if __name__ == "__main__":
    main()
