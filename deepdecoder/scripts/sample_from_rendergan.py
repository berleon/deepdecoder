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

from diktya.func_api_helpers import load_model, get_hdf5_attr, get_layer, predict_wrapper
from deepdecoder.data import DistributionHDF5Dataset
from deepdecoder.render_gan import render_gan_custom_objects
import diktya.distributions
import argparse
import numpy as np
import os
import progressbar


def run(g_weights_fname, d_weights_fname, selected_outputs, nb_samples, out_fname):
    generator = load_model(g_weights_fname, render_gan_custom_objects())
    discriminator = load_model(d_weights_fname, render_gan_custom_objects())
    generator._make_predict_function()
    discriminator._make_predict_function()
    dist_json = get_hdf5_attr(g_weights_fname, 'distribution').decode('utf-8')
    dist = diktya.distributions.load_from_json(dist_json)
    os.makedirs(os.path.dirname(out_fname), exist_ok=True)
    dset = DistributionHDF5Dataset(out_fname, mode='w', nb_samples=nb_samples,
                                   distribution=dist)
    batch_size = 100
    avialable_datasets = [name for name in generator.output_names
                          if name != 'labels']
    print("Avialable outputs: " + ", ".join(avialable_datasets))
    generator_predict = predict_wrapper(lambda x: generator.predict(x, batch_size),
                                        generator.output_names)

    def sample_generator():
        z_shape = get_layer(generator.inputs[0]).batch_input_shape
        while True:
            z = np.random.uniform(-1, 1, (batch_size, ) + z_shape[1:])
            outs = generator_predict(z)
            raw_labels = outs.pop('labels')
            pos = 0
            labels = np.zeros(len(raw_labels), dtype=dist.norm_dtype)
            for name, size in dist.norm_nb_elems.items():
                labels[name] = raw_labels[:, pos:pos+size]
                pos += size
            deleted_keys = []
            if selected_outputs != 'all':
                for name in list(outs.keys()):
                    if name not in selected_outputs:
                        del outs[name]
                        deleted_keys.append(name)
            if not outs:
                raise Exception("Got no outputs. Removed {}. Selected outputs {}"
                                .format(deleted_keys, selected_outputs))
            outs['labels'] = labels
            outs['discriminator'] = discriminator.predict(outs['fake'])
            yield outs

    bar = progressbar.ProgressBar(max_value=nb_samples)
    for batch in sample_generator():
        pos = dset.append(**batch)
        bar.update(pos)
        if pos >= nb_samples:
            break
    dset.close()
    print("Saved dataset with fakes and labels to: {}".format(out_fname))


def main():
    parser = argparse.ArgumentParser(
        description='Sample from a trained generator. All networks hdf5 files'
        ' must have their json model config in kktoplevel attr a ')

    parser.add_argument('-g', '--generator', type=str,
                        help='generator hdf5 weights with model description')
    parser.add_argument('-d', '--discriminator', type=str,
                        help='discriminator hdf5 weights with model description')
    parser.add_argument('-n', '--nb-samples', default=1000, type=float,
                        help='number of epoch to train.')
    parser.add_argument('-s', '--selected-outputs', default='all', type=str,
                        help='name of datasets to collect.')
    parser.add_argument('output', type=str, help='write samples to this hdf5 filename')
    args = parser.parse_args()
    selected_outputs = args.selected_outputs
    if selected_outputs != 'all':
        selected_outputs = [name.replace(' ', '')
                            for name in selected_outputs.split(",")]
    print("Selected outputs: " + ", ".join(selected_outputs))
    run(args.generator, args.discriminator, selected_outputs,
        int(args.nb_samples), args.output)


if __name__ == "__main__":
    main()
