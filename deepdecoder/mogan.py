# Copyright 2015 Leon Sixt
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


from beras.gan import GAN, flatten
from deepdecoder.mutliple_objectives import MultipleObjectives
import keras.backend as K
from keras.objectives import binary_crossentropy
from keras.optimizers import SGD
from copy import copy


def mogan(self, gan: GAN, loss_fn, d_optimizer, name="mogan",
          gan_objective=binary_crossentropy, gan_regulizer=None,
          cond_true_ndim=4):
    assert len(gan.conditionals) >= 1

    g_dummy_opt = SGD()
    d_optimizer = d_optimizer
    v = gan.build(g_dummy_opt, d_optimizer, gan_objective)
    del v['g_updates']

    cond_true = K.placeholder(ndim=cond_true_ndim)
    inputs = copy(gan.graph.inputs)
    inputs['cond_true'] = cond_true

    cond_loss = loss_fn(cond_true, v.g_outmap)

    metrics = {
        "cond_loss": cond_loss.mean(),
        "d_loss": v.d_loss,
        "g_loss": v.g_loss,
    }

    params = flatten([n.trainable_weights
                      for n in gan.get_generator_nodes().values()])

    return MultipleObjectives(
        name, inputs, metrics=metrics, params=params,
        objectives={'g_loss': v['g_loss'], 'cond_loss': cond_loss},
        additional_updates=v['d_updates'] + gan.updates)
