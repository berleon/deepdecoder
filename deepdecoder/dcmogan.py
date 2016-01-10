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


import theano
from keras.models import Graph
from keras.layers.core import Dense
from keras.objectives import mse
from keras.optimizers import Adam
import numpy as np

from beras.gan import GAN
from beras.models import asgraph
from beesgrid import TAG_SIZE, CONFIG_ROTS
from beesgrid.pybeesgrid import NUM_CONFIGS, NUM_MIDDLE_CELLS
from deepdecoder.mogan import MOGAN
from deepdecoder.utils import binary_mask


def dcmogan(generator_fn, discriminator_fn, batch_size=128):
    nb_g_z = 20
    nb_grid_config = NUM_CONFIGS + NUM_MIDDLE_CELLS + len(CONFIG_ROTS)
    ff_generator = generator_fn(input_dim=2*nb_g_z, nb_output_channels=2)

    g = Graph()
    g.add_input("z", (nb_g_z, ))
    g.add_input("grid_config", (nb_grid_config, ))
    g.add_node(Dense(nb_g_z, activation='relu'), "dense1",
               input="grid_config")
    g.add_node(ff_generator, "dcgan", inputs=["z", "dense1"],
               merge_mode='concat')
    g.add_output("output", input="dcgan")
    g.add_node(Dense(1, activation='sigmoid'), "alpha", input="dense1",
               create_output=True)

    def reconstruct(g_outmap):
        g_out = g_outmap["output"]
        alpha = g_outmap["alpha"]
        alpha = 0.5*alpha + 0.5
        alpha = alpha.reshape((batch_size, 1, 1, 1))
        m = g_out[:, :1]
        v = g_out[:, 1:]
        return (alpha * m + (1 - alpha) * v).reshape((batch_size, 1, TAG_SIZE,
                                                      TAG_SIZE))

    grid_loss_weight = theano.shared(np.cast[np.float32](1))

    def grid_loss(grid_idx, g_outmap):
        g_out = g_outmap['output']
        m = g_out[:, :1]
        b = binary_mask(grid_idx, ignore=0.0,  white=1.)
        return grid_loss_weight*mse(b, m)

    gan = GAN(g, asgraph(discriminator_fn(), input_name=GAN.d_input),
              z_shape=(batch_size, nb_g_z),
              reconstruct_fn=reconstruct)
    mogan = MOGAN(gan, grid_loss, lambda: Adam(lr=0.0002, beta_1=0.5),
                  gan_regulizer=GAN.L2Regularizer())

    return mogan, grid_loss_weight
