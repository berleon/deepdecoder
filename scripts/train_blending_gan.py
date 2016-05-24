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

from deepdecoder.blending_gan import BlendingGANExperiment,\
    mask_generator_from_json, build_blending_gan
import argparse
import sys
import os


sys.setrecursionlimit(10000)


def main(output_dir, nb_gen_units, nb_dis_units):
    mask_generator_dir = \
        "models/holy/mask_generator_var_with_depth_n16_d1_e751/"
    mask_generator_json = os.path.join(mask_generator_dir,
                                       "mask_generator.json")
    mask_generator_weights = os.path.join(mask_generator_dir,
                                          "750_mask_generator.hdf5")
    mask_generator = mask_generator_from_json(mask_generator_json)
    mask_generator.load_weights(mask_generator_weights)
    gan = build_blending_gan(lambda x: mask_generator(x),
                             nb_gen_units=nb_gen_units,
                             nb_dis_units=nb_dis_units)
    blending_gan = BlendingGANExperiment(gan, output_dir)
    # blending_gan.save_real_images()
    blending_gan.compile_visualise()
    blending_gan.visualise(fname=blending_gan.add_output_dir('test.png'))
    blending_gan.compile()
    blending_gan.train(301)
    blending_gan.visualise(fname=blending_gan.add_output_dir('end.png'))
    blending_gan.compile_sample()
    blending_gan.sample(301, 10000)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Train the mask blending gan')

    parser.add_argument('--gen-units', default=96, type=int,
                        help='number of units in the generator.')

    parser.add_argument('--dis-units', default=64, type=int,
                        help='number of units in the discriminator.')

    parser.add_argument('--epoch', default=300, type=int,
                        help='number of epoch to train.')

    parser.add_argument('--output-dir', type=str, help='output dir of the gan')
    args = parser.parse_args()
    main(args.output_dir, args.gen_units, args.dis_units)
