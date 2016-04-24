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

from deepdecoder.networks import mask_blending_generator, get_mask_driver, \
    get_lighting_generator, get_offset_merge_mask, get_mask_weight_blending, \
    get_offset_back, get_offset_front, get_offset_middle, mask_generator, \
    mask_blending_discriminator, get_mask_postprocess

from beras.visualise import zip_visualise_tiles, visualise_tiles
from deepdecoder.data import nb_normalized_params, real_generator
from beras.gan import GAN, gan_binary_crossentropy, gan_outputs
from beras.callbacks import VisualiseGAN
from beras.transform import tile
from beras.util import collect_layers, from_config
from keras.optimizers import Adam
from keras.callbacks import Callback
from keras.utils.generic_utils import Progbar
from beesgrid import NUM_MIDDLE_CELLS
from deepdecoder.grid_curriculum import Lecture, to_radians, Uniform, \
    Normal, Constant, TruncNormal

from scipy.ndimage.interpolation import zoom
from scipy.ndimage.filters import gaussian_filter1d
import argparse
import numpy as np
import time
import pylab
import matplotlib.pyplot as plt
import sys
import os
import h5py
import json


sys.setrecursionlimit(10000)
pylab.rcParams['figure.figsize'] = (16, 16)


class SampleCallback(Callback):
    def __init__(self, gan, nb_samples, should_sample):
        self.gan = gan
        self.nb_samples = nb_samples
        self.should_sample = should_sample

    def on_epoch_end(self, epoch, log={}):
        if self.should_sample(epoch):
            self.gan.sample(epoch, self.nb_samples)


class SaveGAN(Callback):
    def __init__(self, gan, every_epoch=50,
                 overwrite=True):
        self.gan = gan
        self.every_epoch = every_epoch
        self.overwrite = overwrite

    def on_epoch_end(self, epoch, log={}):
        epoch = epoch
        if epoch % self.every_epoch == 0:
            self.gan.save_weights(epoch, self.overwrite)


def save_image(data, fname, cmap='gray'):
    sizes = np.shape(data)
    height = float(sizes[0])
    width = float(sizes[1])
    fig = plt.figure()
    fig.set_size_inches(width/height, 1, forward=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.imshow(data, cmap=cmap)
    plt.savefig(fname, dpi=height)
    plt.close()


class VisualiseMasks(VisualiseGAN):
    def __init__(self, inputs,  **kwargs):
        self.inputs = inputs
        self.batch_size = 10
        super().__init__(**kwargs)

    def __call__(self, fname=None):
        self.inputs['z'] = self.z
        tiles = []
        for i in range(0, self.nb_samples, self.batch_size):
            to = min(self.nb_samples, i + self.batch_size)
            batch = {}
            for name, input in self.inputs.items():
                batch[name] = input[i:to]

            outs = self.model.debug(batch)
            for mask, blending in zip(self.preprocess(outs['mask']),
                                      self.preprocess(outs['generator'])):
                tiles.append(mask)
                tiles.append(blending)
        tiled = tile(tiles, columns_must_be_multiple_of=2)
        print(tiled.shape)
        if fname is not None:
            save_image(tiled[0], fname)


def lecture():
    z = np.pi
    DISTRIBUTION_PARAMS = {
        'z': {'low': -z, 'high': z},
        'y': {'mean': 0, 'std': to_radians(25)},
        'x': {'mean': 0, 'std': to_radians(25)},
        'center': {'mean': 0, 'std': 2.},
        'radius': {'mean': 24.0, 'std': 1.2},
    }
    lecture = Lecture.from_params(DISTRIBUTION_PARAMS)
    lecture.bulge_factor = Uniform(0.4, 0.8)
    lecture.inner_ring_radius = Uniform(0.42, 0.48)
    lecture.focal_length = Uniform(2, 4)
    lecture.middle_ring_radius = Constant(0.8)
    std_angle = to_radians(60)
    lecture.y = Uniform(-std_angle, std_angle)
    lecture.x = Uniform(-std_angle, std_angle)
    lecture.radius = TruncNormal(-2, 2, 24.0, 1.2)
    lecture.center = TruncNormal(-4, 4, 0, 2.)
    return lecture


class BlendingGAN:
    def __init__(self, output_dir,
                 mask_generator_weights,
                 nb_gen_units=128, nb_dis_units=64,
                 real_hdf5_fname="/home/leon/data/tags_plain_t6.hdf5",
                 ):
        self.nb_fake = 32
        self.nb_real = 32 // 2
        self.nb_gen_units = 48
        self.dis_nb_units = 48
        self.lr = 0.00005
        self.beta_1 = 0.5
        self.nb_input_mask_generator = \
            nb_normalized_params(lecture()) - NUM_MIDDLE_CELLS
        self.z_dim_offset = 50
        self.z_dim_driver = 50
        self.z_dim_bits = 12
        self.z_dim = self.z_dim_offset + self.z_dim_driver + self.z_dim_bits

        self.mask_generator_dir = os.path.dirname(mask_generator_weights)
        self.mask_generator_weights = mask_generator_weights
        self.real_hdf5_fname = real_hdf5_fname
        self.output_dir = output_dir
        os.makedirs(self.output_dir)
        self.histories = []

        self.debug_keys = [
            'mask',
            'mask_depth_map',
            'driver',
            'mask_with_light',
            'blending',
            'mask_post',
            'generator',
        ]

    def _mask_generator(self, x):
        with open(os.path.join(self.mask_generator_dir,
                               "mask_generator.json")) as f:
            config = json.load(f)['config']
            assert len(config['input_layers']) == 1
            name_idx = 0
            input_name = config['input_layers'][0][name_idx]
            _, mask_gen_outs = from_config(config, inputs={input_name: x})
            mask_layers = collect_layers(x, mask_gen_outs)
            for layer in mask_layers:
                layer.trainable = False
            return mask_gen_outs

    def build(self):
        self.g_mask = lambda x: mask_generator(
            x, nb_units=16, dense_factor=7, nb_dense_layers=3,
            depth=4, trainable=False)

        self.d = lambda x: gan_outputs(
            mask_blending_discriminator(x, n=self.dis_nb_units),
            fake_for_gen=(0, self.nb_fake),
            fake_for_dis=(0, self.nb_fake - self.nb_real),
            real=(self.nb_fake, self.nb_fake + self.nb_real))

        def merge16(namespace):
            def call(x):
                return get_offset_merge_mask(
                    x, nb_units=self.nb_gen_units // 3, nb_conv_layers=2,
                    poolings=[True, True], ns=namespace)
            return call

        def merge32(namespace):
            def call(x):
                return get_offset_merge_mask(
                    x, nb_units=self.nb_gen_units // 3, nb_conv_layers=2,
                    poolings=[True, False], ns=namespace)
            return call

        def merge(namespace):
            return lambda x: get_offset_merge_mask(
                x, nb_units=self.nb_gen_units // 3,
                nb_conv_layers=2, ns=namespace)

        self.g = mask_blending_generator(
            mask_driver=lambda x: get_mask_driver(
                x, nb_units=self.nb_gen_units,
                nb_output_units=self.nb_input_mask_generator),
            mask_generator=self._mask_generator,
            light_merge_mask16=merge('light_merge16'),
            offset_merge_light16=merge16('offset_merge_light16'),
            offset_merge_mask16=merge('offset_merge16'),
            offset_merge_mask32=merge('offset_merge32'),
            lighting_generator=lambda x: get_lighting_generator(
                x, self.nb_gen_units // 2),
            offset_front=lambda x: get_offset_front(x, self.nb_gen_units),
            offset_middle=lambda x: get_offset_middle(x, self.nb_gen_units),
            offset_back=lambda x: get_offset_back(x, self.nb_gen_units),
            mask_weight_blending32=lambda x:
                get_mask_weight_blending(x, min=0.15),
            mask_weight_blending64=get_mask_weight_blending,
            mask_generator_weights=self.mask_generator_weights,
            mask_postprocess=lambda x:
                get_mask_postprocess(x, self.nb_gen_units // 3),
            z_for_driver=(0, self.z_dim_driver),
            z_for_offset=(self.z_dim_driver, self.z_dim_driver +
                          self.z_dim_offset),
            z_for_bits=(self.z_dim_driver + self.z_dim_offset,
                        self.z_dim_driver + self.z_dim_offset +
                        self.z_dim_bits))

        self.gan = GAN(self.g, self.d, z_shape=(self.z_dim,),
                       real_shape=(1, 64, 64))

        self.g_optimizer = Adam(self.lr, self.beta_1)
        self.d_optimizer = Adam(self.lr, self.beta_1)
        self.gan.build(self.g_optimizer, self.d_optimizer,
                       gan_binary_crossentropy)
        self._build_callbacks()

    def _build_callbacks(self, ):
        self.save_callback = SaveGAN(self, every_epoch=50)
        self.sample_callback = SampleCallback(
            self, nb_samples=500000,
            should_sample=lambda x: x >= 100 and x % 50 == 0)

        nb_visualise = 50**2 // 2
        real_placeholder = np.zeros((1, 1, 64, 64), dtype='float32')
        self.vis = VisualiseMasks(nb_samples=nb_visualise,
                                  output_dir=self.add_output_dir('visualise'),
                                  show=False,
                                  preprocess=lambda x: np.clip(x, -1, 1),
                                  inputs={'real': real_placeholder})
        self.vis.model = self.gan
        self.vis.on_train_begin(0)

    def compile_debug(self, debug_keys=None):
        print("Compiling Debug...")
        start = time.time()
        if debug_keys is None:
            debug_keys = self.debug_keys
        self.gan.compile_debug(debug_keys)
        print("Done Compiling in {0:.2f}s".format(time.time() - start))

    def compile(self):
        print("Compiling...")
        start = time.time()
        self.gan.compile()
        print("Done Compiling in {0:.2f}s".format(time.time() - start))

    def visualise_debug(self, inputs, fname, debug_keys):
        out = self.gan.debug(inputs)
        visualise = []
        for key in debug_keys:
            arr = out[key]
            assert len(arr.shape) == 4, "Only 4-dim arrays are supported"
            assert arr.shape[-1] == arr.shape[-2], \
                "arrays must have same dimension in height and "
            if arr.shape[-1] != 64:
                scale = 64 / arr.shape[-1]
                arr = zoom(arr, (1, 1, scale, scale))
            visualise.append(np.clip(arr, -1, 1))

        zip_visualise_tiles(*visualise, show=False)
        plt.savefig(self.add_output_dir(fname),
                    bbox_inches='tight', dpi='figure')

    def add_output_dir(self, path):
        return os.path.join(self.output_dir, path)

    def train_data_generator(self, nb_fake, nb_real):
        print(self.real_hdf5_fname)
        for real in real_generator(self.real_hdf5_fname, nb_real,
                                   range=(-1, 1)):
            z = np.random.uniform(
                -1, 1, (nb_fake, self.z_dim)).astype(np.float32)
            real = gaussian_filter1d(real, 2/3, axis=-1)
            real = gaussian_filter1d(real, 2/3, axis=-2)
            yield {
                'real': real,
                'z': z,
            }

    def save_real_images(self, nb_real):
        real = next(self.train_data_generator(0, nb_real))['real']
        tiled = tile(real)
        fname = self.add_output_dir('real_{}.png'.format(nb_real))
        save_image(tiled[0], fname)

    def train(self, nb_epoch):
        self.histories.append(
            self.gan.fit_generator(
                self.train_data_generator(self.nb_fake, self.nb_real),
                nb_batches_per_epoch=100,
                nb_epoch=nb_epoch, batch_size=128,
                verbose=1, callbacks=[
                    self.vis,              # every 10th epoch
                    self.save_callback,    # every 50th epoch
                ])
        )

    def save_weights(self, nb_epoch, overwrite=False):
        dirname = self.add_output_dir('weights/{}/'.format(nb_epoch))
        os.makedirs(dirname, exist_ok=True)
        print("\nSaving weights to: {}".format(dirname))
        self.gan.save_weights(dirname + "/{}.hdf5")

    def sample(self, nb_epoch, nb_samples):
        chunks = 256
        dirname = self.add_output_dir('samples')
        os.makedirs(dirname, exist_ok=True)
        batch_size = 64
        f = h5py.File(dirname + '/{}.hdf5'.format(nb_epoch))
        f.create_dataset("z", shape=(nb_samples, self.z_dim),
                         dtype='float32', chunks=(chunks, self.z_dim),
                         compression='gzip')
        f.create_dataset("fakes", shape=(nb_samples, 1, 64, 64),
                         dtype='float32', chunks=(256, 1, 64, 64),
                         compression='gzip')
        progbar = Progbar(nb_samples)
        for i in range(0, nb_samples, batch_size):
            z = np.random.uniform(-1, 1, (batch_size, self.z_dim))
            fakes = self.gan.generate(inputs={'z': z})
            to = min(nb_samples, i + batch_size)
            nb_from_this_batch = to - i
            f['z'][i:to] = z[:nb_from_this_batch]
            f['fakes'][i:to] = fakes[:nb_from_this_batch]
            progbar.add(nb_from_this_batch)


def main(output_dir, nb_gen_units, nb_dis_units):
    mask_generator_weights = \
        "models/holy/mask_generator_var_with_depth_n16_d1_e751/750_mask_generator.hdf5"
    blending_gan = BlendingGAN(output_dir, mask_generator_weights,
                               nb_gen_units, nb_dis_units)

    blending_gan.save_real_images(50**2)
    blending_gan.build()
    blending_gan.compile_debug()
    blending_gan.vis(fname=blending_gan.add_output_dir('test.png'))
    blending_gan.compile()
    blending_gan.train(300)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Train the mask blending gan')

    parser.add_argument('--gen-units', default=128, type=int,
                        help='number of units in the generator.')

    parser.add_argument('--dis-units', default=128, type=int,
                        help='number of units in the discriminator.')

    parser.add_argument('--epoch', default=300, type=int,
                        help='number of epoch to train.')

    parser.add_argument('--output-dir', type=str, help='output dir of the gan')
    args = parser.parse_args()
    main(args.output_dir, args.gen_units, args.dis_units)
