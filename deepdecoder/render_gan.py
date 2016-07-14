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

from deepdecoder.networks import get_label_generator, \
    get_lighting_generator, get_preprocess, get_blur_factor, \
    get_offset_back, get_offset_front, get_offset_middle, \
    render_gan_discriminator, get_details, \
    MinCoveredRegularizer
from deepdecoder.layers import ThresholdBits, NormSinCosAngle
from beras.visualise import save_image
from deepdecoder.data import real_generator
from beras.gan import GAN
from beras.callbacks import VisualiseGAN, LearningRateScheduler
from beras.transform import tile, zip_tile
from beras.regularizers import with_regularizer
from beras.util import concat
from keras.optimizers import Adam
from keras.callbacks import Callback
from keras.utils.generic_utils import Progbar
from keras.utils.layer_utils import layer_from_config
from keras.engine.topology import Input, merge
from keras.engine.training import Model
from beesgrid import NUM_MIDDLE_CELLS
from collections import OrderedDict

from deepdecoder.transform import PyramidBlending, PyramidReduce, \
    GaussianBlur, AddLighting, Segmentation, HighPass

from beras.layers.core import Split, LinearInBounds, ZeroGradient

from scipy.ndimage.interpolation import zoom
from scipy.ndimage.filters import gaussian_filter1d
import numpy as np
import time
import pylab
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


class VisualiseMasks(VisualiseGAN):
    def __init__(self, inputs, visualise_fn, **kwargs):
        self.inputs = inputs
        self.batch_size = 10
        self.visualise_fn = visualise_fn
        super().__init__(**kwargs)

    def should_visualise(self, i):
        return i % 10 == 0 or i < 20

    def __call__(self, fname=None):
        self.inputs['z'] = self.z
        tiles = []
        fn = self.visualise_fn()
        assert fn is not None
        for i in range(0, self.nb_samples, self.batch_size):
            to = min(self.nb_samples, i + self.batch_size)
            batch = {}
            for name, input in self.inputs.items():
                batch[name] = input[i:to]

            outs = fn(batch)
            for mask, blending in zip(self.preprocess(outs['mask']),
                                      self.preprocess(outs['generator'])):
                tiles.append(2*np.clip(mask, 0, 1) - 1)
                tiles.append(blending)
        tiled = tile(tiles, columns_must_be_multiple_of=2)
        if fname is not None:
            save_image(tiled[0], fname)


def render_gan_custom_objects():
    return {
        'GAN': GAN,
        'Split': Split,
        'HighPass': HighPass,
        'AddLighting': AddLighting,
        'Segmentation': Segmentation,
        'GaussianBlur': GaussianBlur,
        'ZeroGradient': ZeroGradient,
        'LinearInBounds': LinearInBounds,
        'NormSinCosAngle': NormSinCosAngle,
        'PyramidBlending': PyramidBlending,
        'PyramidReduce': PyramidReduce
    }


def load_tag3d_network(weight_fname):
    f = h5py.File(weight_fname, 'r')
    config = json.loads(f.attrs['model'].decode('utf-8'))
    model = layer_from_config(config)
    for layer in model.layers:
        layer.trainable = False
    return model


class RenderGANBuilder:
    def __init__(self,
                 tag3d_network,
                 z_dim_offset=50, z_dim_labels=50, z_dim_bits=12,
                 discriminator_units=32, generator_units=32,
                 data_shape=(1, 64, 64),
                 labels_shape=(22,),
                 nb_bits=12,
                 generator_optimizer=Adam(lr=0.0002, beta_1=0.5),
                 discriminator_optimizer=Adam(lr=0.0002, beta_1=0.5)
                 ):
        self.tag3d_network = tag3d_network
        self.z_dim_offset = z_dim_offset
        self.z_dim_labels = z_dim_labels
        self.z_dim_bits = z_dim_bits
        self.z_dim = z_dim_offset + z_dim_labels + z_dim_bits
        self.discriminator_units = discriminator_units
        self.generator_units = generator_units
        self.preprocess_units = self.generator_units // 2
        self.data_shape = data_shape
        self.labels_shape = labels_shape
        self.nb_bits = nb_bits
        self.generator_optimizer = generator_optimizer
        self.discriminator_optimizer = discriminator_optimizer

        self._build()

    def _build_discriminator(self):
        x = Input(shape=self.data_shape, name='data')
        d = render_gan_discriminator(x, n=self.discriminator_units, conv_repeat=2, dense=[512])
        self.discriminator = Model([x], [d])

    def _build_generator_given_z_offset_and_labels(self):
        labels = Input(shape=self.labels_shape, name='input_labels')
        z_offset = Input(shape=(self.z_dim_offset,), name='input_z_offset')

        outputs = OrderedDict()
        labels_without_bits = Split(self.nb_bits, self.labels_shape[0], axis=1)(labels)
        tag3d, tag3d_depth_map = self.tag3d_network(labels)

        outputs['tag3d'] = tag3d

        outputs['tag3d_depth_map'] = tag3d_depth_map

        segmentation = Segmentation(threshold=-0.08, smooth_threshold=0.2,
                                    sigma=1.5, name='segmentation')

        tag3d_downsampled = PyramidReduce()(tag3d)
        tag3d_segmented = segmentation(tag3d)
        outputs['tag3d_segmented'] = tag3d_segmented

        out_offset_front = get_offset_front([z_offset, ZeroGradient()(labels_without_bits)],
                                            self.generator_units)

        light_depth_map = get_preprocess(tag3d_depth_map, self.preprocess_units,
                                         nb_conv_layers=2)
        light_outs = get_lighting_generator([out_offset_front, light_depth_map],
                                            self.generator_units)
        tag3d_lighten = AddLighting(
            scale_factor=0.6, shift_factor=0.75,
            name='tag3d_lighten')([tag3d] + light_outs)

        outputs['tag3d_lighten'] = tag3d_lighten

        offset_depth_map = get_preprocess(tag3d_depth_map, self.preprocess_units,
                                          nb_conv_layers=2)

        offset_middle_light = get_preprocess(concat(light_outs), self.preprocess_units,
                                             resize=['down', 'down'])
        out_offset_middle = get_offset_middle(
            [out_offset_front, offset_depth_map, offset_middle_light], self.generator_units)

        offset_back_tag3d_downsampled = get_preprocess(tag3d_downsampled,
                                                       self.preprocess_units,
                                                       nb_conv_layers=2)

        offset_back_feature_map, out_offset_back = get_offset_back(
            [out_offset_middle, offset_back_tag3d_downsampled], self.generator_units)

        mask_weight64 = get_blur_factor(out_offset_middle)
        blending = PyramidBlending(offset_pyramid_layers=2,
                                   mask_pyramid_layers=2,
                                   mask_weights=['variable', 1],
                                   offset_weights=[1, 1],
                                   use_selection=[True, True],
                                   name='blending')(
                [out_offset_back, tag3d_lighten, tag3d_segmented,
                 mask_weight64])

        details = get_details(
            [blending, tag3d_segmented, tag3d, out_offset_back,
             offset_back_feature_map] + light_outs, self.generator_units)

        outputs['details_offset'] = details
        details_high_pass = HighPass(4, nb_steps=4, name='details_high_pass')(details)
        fake = LinearInBounds(-1.2, 1.2, name='fake')(
            merge([details_high_pass, blending], mode='sum'))
        outputs['fake'] = fake
        self.generator_given_z_and_labels = Model([z_offset, labels], [fake])
        # errors: fake
        # ok: tag3d, light_depth_map, lights_outs[0], tag3d_lighten
        self.generator_given_z_and_labels = Model([z_offset, labels], [fake])
        self.sample_generator_given_z_and_labels_output_names = list(outputs.keys())
        self.sample_generator_given_z_and_labels = Model([z_offset, labels],
                                                         list(outputs.values()))

    @property
    def pos_z_labels(self):
        return (0, self.z_dim_labels)

    @property
    def pos_z_offset(self):
        return (self.z_dim_labels, self.z_dim_labels + self.z_dim_offset)

    @property
    def pos_z_bits(self):
        return (self.z_dim_labels + self.z_dim_offset,
                self.z_dim_labels + self.z_dim_offset + self.z_dim_bits)

    def _build_generator_given_z(self):
        z = Input(shape=(self.z_dim,), name='z')

        z_labels = Split(*self.pos_z_labels, axis=1)(z)
        z_offset = Split(*self.pos_z_offset, axis=1)(z)
        z_bits = Split(*self.pos_z_bits, axis=1)(z)
        bits = ThresholdBits()(z_bits)
        nb_labels_without_bits = self.labels_shape[0] - self.nb_bits
        generated_labels = get_label_generator(
            z_labels, self.generator_units, nb_output_units=nb_labels_without_bits)

        labels_normed = NormSinCosAngle(0)(generated_labels)
        labels = concat([bits, labels_normed], name='labels')
        fake = self.generator_given_z_and_labels([z_offset, labels])
        self.generator_given_z = Model([z], [fake])

        sample_tensors = self.sample_generator_given_z_and_labels([z_offset, labels])
        self.sample_generator_given_z_output_names = ['labels'] + \
            self.sample_generator_given_z_and_labels_output_names
        self.sample_generator_given_z = Model([z], [labels] + sample_tensors)

    def _build_gan(self):
        self.generator_given_z.compile(self.generator_optimizer, 'binary_crossentropy')
        self.discriminator.compile(self.discriminator_optimizer, 'binary_crossentropy')
        self.gan = GAN(self.generator_given_z, self.discriminator)

    def _build(self):
        self._build_discriminator()
        self._build_generator_given_z_offset_and_labels()
        self._build_generator_given_z()
        self._build_gan()


class RenderGANExperiment:
    def __init__(self, gan, output_dir,
                 real_hdf5_fname):
        self.nb_fake = 32
        self.nb_real = 32 // 2
        self.lr = 0.0002
        self.beta_1 = 0.5
        self.gan = gan
        self.tag3d_nb_inputs = nb_normalized_params(tag_distribution())
        self.nb_mask_driver_output_units = self.tag3d_nb_inputs - NUM_MIDDLE_CELLS
        self.z_dim_offset = 50
        self.z_dim_labels = 50
        self.z_dim_bits = 12
        self.z_dim = self.z_dim_offset + self.z_dim_labels + self.z_dim_bits
        self.real_hdf5_fname = real_hdf5_fname
        self.output_dir = output_dir
        os.makedirs(self.output_dir)
        self.histories = []
        self.nb_visualise = 20**2
        self.visualise_layers = [
            'tag3d',
            'mask_depth_map',
            'mask_gen_input',
            'selection',
            'mask_with_lighting',
            'blending',
            'mask_post',
            'mask_post_high',
            'generator',
            'discriminator'
        ]
        self.g_optimizer = Adam(self.lr, self.beta_1)
        self.d_optimizer = Adam(self.lr, self.beta_1)
        self.gan.build(self.g_optimizer, self.d_optimizer,
                       gan_binary_crossentropy)

        with open(self.join_output_dir("gan.json"), 'w') as f:
            config = self.gan.get_config()
            json.dump(config, f)
        self._build_callbacks()

    def _real_placeholder(self):
        return np.zeros((1, 1, 64, 64), dtype='float32')

    def _build_callbacks(self):
        self.save_callback = SaveGAN(self, every_epoch=10)
        self.sample_callback = SampleCallback(
            self, nb_samples=500000,
            should_sample=lambda x: x >= 100 and x % 50 == 0)

        self.vis = VisualiseMasks(nb_samples=self.nb_visualise // 2,
                                  visualise_fn=lambda: self.visualise_fn,
                                  output_dir=self.join_output_dir('visualise'),
                                  show=False,
                                  preprocess=lambda x: np.clip(x, -1, 1),
                                  inputs={'real': self._real_placeholder()})
        self.vis.model = self.gan
        self.vis.on_train_begin(0)

        schedule = {
            50: self.lr / 4,
            150: self.lr / 4**2,
            250: self.lr / 4**3,
        }
        self.lr_schedulers = [
            LearningRateScheduler(self.g_optimizer, schedule),
            LearningRateScheduler(self.d_optimizer, schedule)
        ]

    def compile_sample(self):
        layers = ['generator', 'mask_gen_input', 'discriminator']
        start = time.time()
        self._sample_fn = self.gan.compile_custom_layers(layers)
        print("Done Compiling in {0:.2f}s".format(time.time() - start))

    def compile_visualise(self, visualise_layers=None):
        start = time.time()
        if visualise_layers is None:
            visualise_layers = self.visualise_layers
        self.visualise_fn = self.gan.compile_custom_layers(visualise_layers)
        print("Done Compiling in {0:.2f}s".format(time.time() - start))

    def compile(self):
        print("Compiling...")
        start = time.time()
        self.gan.compile()
        print("Done Compiling in {0:.2f}s".format(time.time() - start))

    def visualise(self, fname, inputs=None, selected_layers=None):
        nb_visualise = 20
        if inputs is None:
            inputs = {
                'real': self._real_placeholder(),
                'z': np.random.uniform(-1, 1, (nb_visualise, self.z_dim)),
            }
        if selected_layers is None:
            selected_layers = self.visualise_layers
        out = self.visualise_fn(inputs)
        visualise = []
        for layer in selected_layers:
            arr = out[layer]
            if len(arr.shape) != 4:
                continue
            assert arr.shape[-1] == arr.shape[-2], \
                "arrays must have same dimension in height and width"
            if arr.shape[-1] != 64:
                scale = 64 / arr.shape[-1]
                arr = zoom(arr, (1, 1, scale, scale))
            visualise.append(np.clip(arr, -1, 1))

        tiled = zip_tile(*visualise)
        save_image(tiled, fname)

    def join_output_dir(self, path):
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

    def save_real_images(self):
        real = next(self.train_data_generator(0, self.nb_visualise))['real']
        tiled = tile(real)
        fname = self.join_output_dir('real_{}.png'.format(self.nb_visualise))
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
                ] + self.lr_schedulers)
        )

    def save_weights(self, label, overwrite=False):
        dirname = self.join_output_dir('weights/{}/'.format(label))
        os.makedirs(dirname, exist_ok=True)
        print("\nSaving weights to: {}".format(dirname))
        self.gan.save_weights(dirname + "/{}.hdf5")

    def sample(self, label, nb_samples):
        chunks = 256
        dirname = self.join_output_dir('samples')
        os.makedirs(dirname, exist_ok=True)
        batch_size = 64
        f = h5py.File(dirname + '/{}.hdf5'.format(label))
        f.create_dataset("z", shape=(nb_samples, self.z_dim),
                         dtype='float32', chunks=(chunks, self.z_dim),
                         compression='gzip')
        f.create_dataset("fakes", shape=(nb_samples, 1, 64, 64),
                         dtype='float32', chunks=(256, 1, 64, 64),
                         compression='gzip')
        f.create_dataset("mask_gen_input", dtype='float32',
                         shape=(nb_samples, self.tag3d_nb_inputs),
                         chunks=(256, self.tag3d_nb_inputs),
                         compression='gzip')
        f.create_dataset("discriminator", dtype='float32',
                         shape=(nb_samples, 1),
                         chunks=(256, 1),
                         compression='gzip')
        progbar = Progbar(nb_samples)
        real = self._real_placeholder()
        for i in range(0, nb_samples, batch_size):
            z = np.random.uniform(-1, 1, (batch_size, self.z_dim))
            outs = self._sample_fn(inputs={'z': z, 'real': real})
            fakes = outs['generator']
            mask_gen_input = outs['mask_gen_input']
            dis_out = outs['discriminator']
            to = min(nb_samples, i + batch_size)
            nb_from_this_batch = to - i
            f['z'][i:to] = z[:nb_from_this_batch]
            f['fakes'][i:to] = fakes[:nb_from_this_batch]
            f['mask_gen_input'][i:to] = mask_gen_input[:nb_from_this_batch]
            f['discriminator'][i:to] = dis_out[:nb_from_this_batch]
            progbar.add(nb_from_this_batch)
