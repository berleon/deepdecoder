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

import os
os.environ['THEANO_FLAGS'] = 'device=cpu'  # noqa

import os
from subprocess import Popen, TimeoutExpired, DEVNULL

import click
import random
from random import uniform
import json
from pprint import pprint

import numpy as np

from diktya.random_search import fmin

from deepdecoder.scripts.train_decoder import get_output_dir


def rand_bool():
    return random.randint(0, 1) == 0


def random_aug_space(cl_args, base_output_dir, data_name):
    def wrapper(i):
        scale = uniform(0., 0.3)
        shear = uniform(0., 0.3)
        rotation = uniform(0., 0.2 * np.pi)
        channel_scale = uniform(0., 0.3)
        channel_shift = uniform(0., 0.3)
        noise_mean = uniform(0., 0.10)
        noise_std = uniform(0., 0.07)
        return {
            'decoder_model': 'resnet',
            'nb_batches_per_epoch': 1000,
            'data_name': data_name,
            'marker': "iter{}".format(i),
            'nb_epoch': 50,
            'nb_units': random.choice([16, 32]),
            'use_hist_equalization': rand_bool(),
            'use_warp_augmentation': rand_bool(),
            'use_noise_augmentation': rand_bool(),
            'use_diffeomorphism_augmentation': rand_bool(),
            'use_channel_scale_shift_augmentation': rand_bool(),
            'augmentation_rotation': rotation,
            'augmentation_scale': (1 - scale, 1 + scale),
            'augmentation_shear': (-shear, shear),
            'augmentation_channel_scale': (1 - channel_scale, 1 + channel_scale),
            'augmentation_channel_shift': (-channel_shift, channel_shift),
            'augmentation_noise_mean': noise_mean,
            'augmentation_noise_std': noise_std,
            'augmentation_diffeomorphism': [(4, 0.3),
                                            (8, uniform(0.3, 0.7)),
                                            (16, uniform(0.3, 0.7)),
                                            (32, uniform(0.3, 0.7))],
        }, base_output_dir, cl_args
    return wrapper


def process_env():
    gpu_env = os.environ.copy()
    del gpu_env['THEANO_FLAGS']
    return gpu_env


def train_decoder(input):
    config, base_output_dir, cl_args = input
    output_dir = get_output_dir(base_output_dir, config)
    config['output_dir'] = output_dir
    os.makedirs(output_dir)
    config_fname = os.path.join(config['output_dir'], "decoder_config.json")
    with open(config_fname, 'w') as f:
        json.dump(config, f, indent=2)

    cmd = ['bb_train_decoder'] + cl_args + [config_fname]
    stdout_fname = os.path.join(output_dir, "stdout.log")
    print("\\\n    ".format(cmd))
    with open(stdout_fname, 'wb+') as stdout_log:
        process = Popen(cmd, env=process_env(), stdin=DEVNULL, stdout=stdout_log)
        process.wait()

    with open(os.path.join(output_dir, 'results.json')) as f:
        results = json.load(f)
        return float(results['mhd'])


@click.command()
@click.option('--gt-test', type=click.Path(exists=True))
@click.option('--gt-val', type=click.Path(exists=True))
@click.option('--train-set', '-t', type=click.Path(exists=True))
@click.option('--test-set', type=click.Path(exists=True))
@click.option('--data-name', type=str)
@click.option('--n-jobs', default=2, type=int)
@click.option('--n-trails', default=15, type=int)
@click.argument('output_dir', type=click.Path(resolve_path=True))
def main(gt_test, gt_val, train_set, test_set, data_name,
         output_dir, n_jobs, n_trails):
    os.makedirs(output_dir, exist_ok=False)
    cl_args = [
        '--gt-test', gt_test,
        '--gt-val', gt_val,
        '--train-sets', train_set,
        '--test-set', test_set,
    ]
    results = fmin(train_decoder,
                   random_aug_space(cl_args, output_dir, data_name),
                   n=n_trails, n_jobs=n_jobs)
    mhd, (config, base_output_dir, _) = results[0]
    print("Min MHD: {}".format(mhd))
    print("with config:")
    pprint(config)

    with open(os.path.join(output_dir, "results.json"), 'w') as f:
        json.dump(results, f)
