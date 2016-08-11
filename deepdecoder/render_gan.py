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

from deepdecoder.networks import get_label_generator, \
    get_lighting_generator, get_preprocess, get_blur_factor, \
    get_offset_back, get_offset_front, get_offset_middle, \
    render_gan_discriminator, get_details
from deepdecoder.layers import ThresholdBits, NormSinCosAngle
from deepdecoder.data import real_generator, DistributionHDF5Dataset, get_distribution_hdf5_attrs
from diktya.gan import GAN
from diktya.callbacks import VisualiseGAN, LearningRateScheduler, HistoryPerBatch, \
    OnEpochEnd, SampleGAN
from diktya.numpy import tile, zip_tile, scipy_gaussian_filter_2d
from diktya.func_api_helpers import concat, save_model, load_model, predict_wrapper, \
    name_tensor

from keras.optimizers import Adam
from keras.callbacks import Callback
from keras.engine.topology import Input, merge
from keras.engine.training import Model
import keras.backend as K

import scipy.misc
import matplotlib.pyplot as plt
import seaborn as sns
from collections import OrderedDict

from deepdecoder.transform import Background, PyramidReduce, \
    GaussianBlur, AddLighting, Segmentation, HighPass, BlendingBlur

from diktya.layers.core import Subtensor, InBounds, ZeroGradient

import numpy as np
import os
from os.path import join


class SaveGAN(Callback):
    def __init__(self, rendergan, output_format, every_epoch=50,
                 hdf5_attrs=None,
                 overwrite=True):
        self.rendergan = rendergan
        self.every_epoch = every_epoch
        self.overwrite = overwrite
        if hdf5_attrs is None:
            hdf5_attrs = {}
        self.hdf5_attrs = hdf5_attrs
        self.output_format = output_format

    def on_epoch_end(self, epoch, log={}):
        epoch = epoch
        if epoch % self.every_epoch == 0:
            fname_format = self.output_format.format(epoch=epoch, name="{name}")
            self.rendergan.save_weights(
                fname_format, overwrite=self.overwrite, attrs=self.hdf5_attrs)


class VisualiseTag3dAndFake(VisualiseGAN):
    def call(self, samples):
        tiles = []
        for tag3d, fake in zip(self.preprocess(samples['tag3d']),
                               self.preprocess(samples['fake'])):
            tiles.append(2*np.clip(tag3d, 0, 1) - 1)
            tiles.append(fake)
        return tile(tiles, columns_must_be_multiple_of=2)


class VisualiseFakesSorted(VisualiseGAN):
    def call(self, samples):
        fakes = samples['fake']
        scores = samples['discriminator_on_fake']
        idx = np.argsort(scores.flatten())[::-1]
        fakes_sorted = fakes[idx]
        return tile(self.preprocess(fakes_sorted))


class VisualiseRealsSorted(VisualiseGAN):
    def call(self, samples):
        real = samples['real']
        scores = samples['discriminator_on_real']
        idx = np.argsort(scores.flatten())[::-1]
        real_sorted = real[idx]
        return tile(self.preprocess(real_sorted))


class VisualiseAll(VisualiseGAN):
    def call(self, samples):
        out_arrays = [self.preprocess(o) for o in samples.values() if o.shape[1:] == (1, 64, 64)]
        return zip_tile(*out_arrays)


def render_gan_custom_objects():
    return {
        'GAN': GAN,
        'ThresholdBits': ThresholdBits,
        'Subtensor': Subtensor,
        'HighPass': HighPass,
        'AddLighting': AddLighting,
        'Segmentation': Segmentation,
        'GaussianBlur': GaussianBlur,
        'ZeroGradient': ZeroGradient,
        'InBounds': InBounds,
        'NormSinCosAngle': NormSinCosAngle,
        'Background': Background,
        'PyramidReduce': PyramidReduce,
        'BlendingBlur': BlendingBlur,
    }


def load_tag3d_network(weight_fname):
    model = load_model(weight_fname)
    for layer in model.layers:
        layer.trainable = False
    return model


class RenderGAN:
    def __init__(self,
                 tag3d_network,
                 z_dim_offset=50, z_dim_labels=50, z_dim_bits=12,
                 discriminator_units=32, generator_units=32,
                 data_shape=(1, 64, 64),
                 labels_shape=(24,),
                 nb_bits=12,
                 generator_optimizer=Adam(lr=0.0004, beta_1=0.5),
                 discriminator_optimizer=Adam(lr=0.0004, beta_1=0.5)
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
        d = render_gan_discriminator(x, n=self.discriminator_units,
                                     conv_repeat=2, dense=[512])
        self.discriminator = Model([x], [d])

    def _build_generator_given_z_offset_and_labels(self):
        labels = Input(shape=self.labels_shape, name='input_labels')
        z_offset = Input(shape=(self.z_dim_offset,), name='input_z_offset')

        outputs = OrderedDict()
        labels_without_bits = Subtensor(self.nb_bits, self.labels_shape[0], axis=1)(labels)
        tag3d, tag3d_depth_map = self.tag3d_network(labels)

        outputs['tag3d'] = tag3d

        outputs['tag3d_depth_map'] = tag3d_depth_map

        segmentation = Segmentation(threshold=-0.08, smooth_threshold=0.2,
                                    sigma=1.5, name='segmentation')

        tag3d_downsampled = PyramidReduce()(tag3d)
        tag3d_segmented = segmentation(tag3d)
        outputs['tag3d_segmented'] = tag3d_segmented
        tag3d_segmented_blur = GaussianBlur(sigma=3.0)(tag3d_segmented)
        outputs['tag3d_segmented_blur'] = tag3d_segmented_blur

        out_offset_front = get_offset_front([z_offset, ZeroGradient()(labels_without_bits)],
                                            self.generator_units)

        light_depth_map = get_preprocess(tag3d_depth_map, self.preprocess_units,
                                         nb_conv_layers=2)
        light_outs = get_lighting_generator([out_offset_front, light_depth_map],
                                            self.generator_units)

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

        tag3d_lighten = AddLighting(
            scale_factor=0.75, shift_factor=0.75)([tag3d] + light_outs)

        blur_factor = get_blur_factor(out_offset_middle, min=0.05, max=1.0)
        outputs['tag3d_lighten'] = tag3d_lighten
        tag3d_blur = BlendingBlur(sigma=1)([tag3d_lighten, blur_factor])
        outputs['tag3d_blur'] = tag3d_blur
        blending = Background(name='blending')([out_offset_back, tag3d_blur, tag3d_segmented_blur])

        outputs['fake_without_noise'] = blending
        details = get_details(
            [blending, tag3d_segmented_blur, tag3d, out_offset_back,
             offset_back_feature_map] + light_outs, self.generator_units)
        outputs['details_offset'] = details
        details_high_pass = HighPass(4, nb_steps=4)(details)
        outputs['details_high_pass'] = details_high_pass
        fake = InBounds(-1.0, 1.0)(
            merge([details_high_pass, blending], mode='sum'))
        outputs['fake'] = fake
        for name in outputs.keys():
            outputs[name] = name_tensor(outputs[name], name)

        self.generator_given_z_and_labels = Model([z_offset, labels], [fake])
        self.sample_generator_given_z_and_labels_output_names = list(outputs.keys())
        self.sample_generator_given_z_and_labels = Model([z_offset, labels],
                                                         list(outputs.values()))

    @property
    def pos_z_bits(self):
        return (0, self.z_dim_bits)

    @property
    def pos_z_labels(self):
        return (self.z_dim_bits, self.z_dim_bits + self.z_dim_labels)

    @property
    def pos_z_offset(self):
        return (self.z_dim_labels + self.z_dim_bits,
                self.z_dim_labels + self.z_dim_bits + self.z_dim_offset)

    def _build_generator_given_z(self):
        z = Input(shape=(self.z_dim,), name='z')

        z_bits = Subtensor(*self.pos_z_bits, axis=1)(z)
        z_labels = Subtensor(*self.pos_z_labels, axis=1)(z)
        z_offset = Subtensor(*self.pos_z_offset, axis=1)(z)
        bits = ThresholdBits()(z_bits)
        nb_labels_without_bits = self.labels_shape[0] - self.nb_bits
        generated_labels = get_label_generator(
            z_labels, self.generator_units, nb_output_units=nb_labels_without_bits)

        labels_normed = NormSinCosAngle(0)(generated_labels)
        labels = concat([bits, labels_normed], name='labels')
        fake = self.generator_given_z_and_labels([z_offset, labels])
        self.generator_given_z = Model([z], [fake])

        sample_tensors = self.sample_generator_given_z_and_labels([z_offset, labels])
        sample_tensors = [name_tensor(t, n)
                          for t, n in zip(sample_tensors,
                                          self.sample_generator_given_z_and_labels.output_names)]
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

    def save_weights(self, fname_format, overwrite=False, attrs={}):
        def save(name):
            model = self.__dict__[name]
            fname = fname_format.format(name=name)
            os.makedirs(os.path.dirname(fname), exist_ok=True)
            save_model(model, fname, overwrite=False, attrs=attrs)

        save("sample_generator_given_z")
        save("sample_generator_given_z_and_labels")
        save("generator_given_z_and_labels")
        save("generator_given_z")
        save("discriminator")


class StoreSamples(Callback):
    def __init__(self, output_dir, distribution, overwrite=False):
        self.output_dir = output_dir
        self.distribution = distribution
        self.overwrite = overwrite

    def store(self, outputs, fname):
        store_outputs = {}
        for name, arr in outputs.items():
            if name == 'labels':
                labels = arr.view(dtype=self.distribution.norm_dtype)
                store_outputs['labels'] = labels
            else:
                store_outputs[name] = arr
        nb_samples = len(store_outputs['fake'])
        if os.path.exists(fname) and self.overwrite:
            os.remove(fname)
        dset = DistributionHDF5Dataset(
            fname, self.distribution, nb_samples=nb_samples)

        dset.append(**store_outputs)
        dset.close()

    def on_train_begin(self, epoch, logs={}):
        if 'samples' in logs:
            self.store(
                logs['samples'],
                os.path.join(self.output_dir, "on_train_begin.hdf5"))

    def on_epoch_end(self, epoch, logs={}):
        if 'samples' in logs:
            self.store(
                logs['samples'],
                os.path.join(self.output_dir, "{:05d}.hdf5".format(epoch)))


class DScoreHistogram(Callback):
    def __init__(self, output_dir):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def plot_discriminator_score(self, samples):
        fig, axes = plt.subplots(figsize=(10, 10))
        sns.distplot(samples['discriminator_on_fake'], bins=20, ax=axes,
                     color='red', label='number of fake images',
                     axlabel='discriminator score')
        sns.distplot(samples['discriminator_on_real'], bins=20, ax=axes, color='green',
                     label='number of real images',
                     axlabel='discriminator score')
        axes.legend()
        return fig, axes

    def on_epoch_end(self, epoch, logs={}):
        if 'samples' in logs:
            fig, _ = self.plot_discriminator_score(logs['samples'])
            fname = join(self.output_dir, "{:03d}.png".format(epoch))
            fig.savefig(fname)
            plt.close(fig)


def train_callbacks(rendergan, output_dir, nb_visualise, real_hdf5_fname,
                    distribution, lr_schedule=None, overwrite=False):
    save_gan_cb = SaveGAN(rendergan, join(output_dir, "models/{epoch:03d}/{name}.hdf5"),
                          every_epoch=10, hdf5_attrs=get_distribution_hdf5_attrs(distribution))
    nb_score = 1000

    sample_fn = predict_wrapper(rendergan.sample_generator_given_z.predict,
                                rendergan.sample_generator_given_z_output_names)

    real = next(train_data_generator(real_hdf5_fname, nb_score, 1))['data']

    vis_cb = VisualiseTag3dAndFake(
        nb_samples=nb_visualise // 2,
        output_dir=join(output_dir, 'visualise_tag3d_fake'),
        show=False,
        preprocess=lambda x: np.clip(x, -1, 1)
    )
    vis_all = VisualiseAll(
        nb_samples=nb_visualise // len(rendergan.sample_generator_given_z.outputs),
        output_dir=join(output_dir, 'visualise_all'),
        show=False,
        preprocess=lambda x: np.clip(x, -1, 1))

    vis_fake_sorted = VisualiseFakesSorted(
        nb_samples=nb_visualise,
        output_dir=join(output_dir, 'visualise_fakes_sorted'),
        show=False,
        preprocess=lambda x: np.clip(x, -1, 1))

    vis_real_sorted = VisualiseRealsSorted(
        nb_samples=nb_visualise,
        output_dir=join(output_dir, 'visualise_reals_sorted'),
        show=False,
        preprocess=lambda x: np.clip(x, -1, 1))

    def default_lr_schedule(lr):
        return {
            150: lr / 4,
            200: lr / 4**2,
            250: lr / 4**3,
        }

    def lr_scheduler(opt):
        return LearningRateScheduler(opt, lr_schedule(float(K.get_value(opt.lr))))

    if lr_schedule is None:
        lr_schedule = default_lr_schedule

    g_optimizer = rendergan.gan.g_optimizer
    d_optimizer = rendergan.gan.d_optimizer
    lr_schedulers = [
        lr_scheduler(g_optimizer),
        lr_scheduler(d_optimizer),
    ]
    hist_dir = join(output_dir, "history")
    os.makedirs(hist_dir, exist_ok=True)
    hist = HistoryPerBatch(hist_dir)

    def history_plot(e, logs={}):
        fig, _ = hist.plot(save_as="{:03d}.png".format(e), metrics=['g_loss', 'd_loss'])
        plt.close(fig)  # allows fig to be garbage collected
    hist_save = OnEpochEnd(history_plot, every_nth_epoch=20)

    sample_outdir = join(output_dir, 'samples')
    os.makedirs(sample_outdir, exist_ok=True)
    store_samples_cb = StoreSamples(sample_outdir, distribution, overwrite)

    dscore_outdir = join(output_dir, 'd_score_hist')
    os.makedirs(dscore_outdir, exist_ok=True)
    dscore = DScoreHistogram(dscore_outdir)

    nb_sample = max(nb_score, nb_visualise)
    sample_cb = SampleGAN(sample_fn, rendergan.discriminator.predict,
                          rendergan.gan.random_z(nb_sample), real,
                          callbacks=[vis_cb, vis_fake_sorted, vis_all, vis_real_sorted,
                                     dscore, store_samples_cb])
    return [sample_cb, save_gan_cb, hist, hist_save] + lr_schedulers


def train_data_generator(real_hdf5_fname, batch_size, z_dim):
    for data in real_generator(real_hdf5_fname, batch_size, range=(-1, 1)):
        z = np.random.uniform(-1, 1, (3*batch_size, z_dim)).astype(np.float32)
        # data = scipy_gaussian_filter_2d(real, sigma=2/3.)
        yield {
            'data': data,
            'z': z,
        }


def save_real_images(real_hdf5_fname, output_dir, nb_visualise=20**2):
    real = next(train_data_generator(real_hdf5_fname, nb_visualise, 10))['data']
    tiled = tile(real)
    fname = join(output_dir, 'real_{}.png'.format(nb_visualise))
    scipy.misc.imsave(fname, tiled[0])


def train(rendergan, real_hdf5_fname, output_dir, callbacks=[],
          batch_size=32, nb_epoch=100):
    data_generator = train_data_generator(real_hdf5_fname, batch_size, rendergan.z_dim)
    rendergan.gan.fit_generator(
        data_generator,
        nb_batches_per_epoch=100,
        nb_epoch=nb_epoch, batch_size=batch_size*4,
        verbose=1, callbacks=callbacks)
