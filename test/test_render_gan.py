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

from deepdecoder.render_gan import RenderGANExperiment, \
    RenderGANBuilder, render_gan_custom_objects
from deepdecoder.networks import tag3d_network_dense

from keras.utils.layer_utils import layer_from_config
from keras.engine.topology import Input
from keras.engine.training import Model
from keras.layers.core import Dense
from beras.util import sequential
import json
import numpy as np


def test_model_multiple_calls():
    x1 = Input(shape=(20,))

    y1 = sequential([
        Dense(10),
        Dense(1),
    ])(x1)
    m1 = Model(x1, y1)

    x2 = Input(shape=(25,))
    y2 = sequential([
        Dense(20),
        m1
    ])(x2)
    m2 = Model(x2, y2)
    m2.compile('adam', 'mse')

    x3 = Input(shape=(20,))
    y3 = sequential([
        Dense(25),
        m2
    ])(x3)
    m3 = Model(x3, y3)
    m3.compile('adam', 'mse')
    m3.train_on_batch(np.zeros((32, 20)), np.zeros((32, 1)))


def data(builder, bs):
    z = np.random.uniform(-1, 1, (bs, builder.z_dim))
    z_offset = np.random.uniform(-1, 1, (bs, builder.z_dim_offset))
    labels = np.random.uniform(-1, 1, (bs, ) + builder.labels_shape)
    return z, z_offset, labels


def test_render_gan_builder_gan_train_on_batch():
    builder = RenderGANBuilder(lambda x: tag3d_network_dense(x, nb_units=4),
                               generator_units=4, discriminator_units=4,
                               labels_shape=(27,))
    bs = 19
    z, z_offset, labels = data(builder, bs)

    builder.gan.train_on_batch(g_inputs={
        'z': z
    }, d_inputs={
        'real': np.random.uniform(-1, 1, (bs,) + builder.data_shape),
    })


def test_render_gan_builder_generate():
    builder = RenderGANBuilder(lambda x: tag3d_network_dense(x, nb_units=4),
                               generator_units=4, discriminator_units=4,
                               labels_shape=(27,))
    bs = 19
    z, z_offset, labels = data(builder, bs)
    fakes = builder.generator_given_z_and_labels.predict([z_offset, labels])
    assert fakes.shape == (bs,) + builder.data_shape

    fakes = builder.generator_given_z.predict(z)
    assert fakes.shape == (bs,) + builder.data_shape

    outs = builder.sample_generator_given_z.predict(z)
    assert len(outs) == len(builder.sample_generator_given_z_output_names)


def test_render_gan_builder_generator_train_on_batch():
    builder = RenderGANBuilder(lambda x: tag3d_network_dense(x, nb_units=4),
                               generator_units=4, discriminator_units=4,
                               labels_shape=(27,))
    bs = 19
    z, z_offset, labels = data(builder, bs)
    real = np.zeros((bs,) + builder.data_shape)

    builder.generator_given_z.compile('adam', 'mse')
    builder.generator_given_z.train_on_batch(z, real)

    builder.generator_given_z_and_labels.compile('adam', 'mse')
    builder.generator_given_z_and_labels.train_on_batch([z_offset, labels], real)


def test_render_gan_builder_generator_extended():
    labels_shape = (27,)
    z_dim_offset = 50
    builder = RenderGANBuilder(lambda x: tag3d_network_dense(x, nb_units=4),
                               generator_units=4, discriminator_units=4,
                               z_dim_offset=z_dim_offset,
                               labels_shape=(27,))
    bs = 19
    z, z_offset, labels = data(builder, bs)
    real = np.zeros((bs,) + builder.data_shape)

    labels_input = Input(shape=labels_shape)
    z = Input(shape=(z_dim_offset,))
    fake = builder.generator_given_z_and_labels([z, labels_input])
    m = Model([z, labels_input], [fake])
    m.compile('adam', 'mse')
    m.train_on_batch([z_offset, labels], real)


def test_model_serialization(tmpdir):
    return
    output_dir = tmpdir.join('output_dir')
    gan = RenderGANBuilder(lambda x: tag3d_network_dense(x, nb_units=4),
                           generator_units=4, discriminator_units=4)
    experiment = RenderGANExperiment(gan, str(output_dir))

    experiment.compile_visualise()
    gan_config = gan.get_config()
    f = tmpdir.join('gan.json').open('w+')
    json.dump(gan_config, f, indent=2)
    experiment.save_weights("test")
    experiment.visualise(str(tmpdir.join("test.png")))
    gan_loaded = layer_from_config(
        gan_config, custom_objects=render_gan_custom_objects())

    experiment_loaded = RenderGANExperiment(
        gan_loaded, str(tmpdir.join("loaded")))
    experiment_loaded.compile_visualise()

    all_equal = True
    for l1, l2 in zip(gan.layers, gan_loaded.layers):
        if len(l1.trainable_weights) >= 1:
            assert len(l1.trainable_weights) == len(l2.trainable_weights)
            if not all([(w1.get_value() == w2.get_value()).all()
                        for w1, w2 in zip(
                                l1.trainable_weights, l2.trainable_weights)]):
                all_equal = False
    assert not all_equal

    gan_loaded.load_weights(str(output_dir.join('weights', 'test', '{}.hdf5')))
    experiment_loaded.visualise(str(tmpdir.join("test_loaded.png")))

    for l1, l2 in zip(gan.layers, gan_loaded.layers):
        for w1, w2 in zip(l1.trainable_weights, l2.trainable_weights):
            assert (w1.get_value() == w2.get_value()).all()
