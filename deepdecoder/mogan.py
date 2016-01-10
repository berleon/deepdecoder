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


from beras.gan import GAN
from deepdecoder.mutliple_objectives import MultipleObjectives
import keras.backend as K
from keras.objectives import binary_crossentropy


class MOGAN:
    def __init__(self, gan: GAN, loss_fn, optimizer_fn,
                 name="mogan",
                 gan_objective=binary_crossentropy,
                 gan_regulizer=None):
        assert len(gan.conditionals) >= 1
        y_true = K.placeholder(shape=gan.G.outputs["output"].output_shape)
        v = gan.build_loss(objective=gan_objective)
        inputs = [v.real, y_true] + v.gen_conditionals
        cond_loss = loss_fn(y_true, v.g_outmap)
        gan.build_opt_d(optimizer_fn(), v)
        gan_regulizer = GAN.get_regulizer(gan_regulizer)
        v.g_loss, v.d_loss, v.reg_updates = \
            gan_regulizer.get_losses(gan, v.g_loss, v.d_loss)
        self.build_dict = v
        self.gan = gan
        self.optimizer_fn = optimizer_fn
        outputs_map = {
            "cond_loss": cond_loss.mean(),
            "d_loss": v.d_loss,
            "g_loss": v.g_loss,
        }
        if type(gan_regulizer) == GAN.L2Regularizer:
            outputs_map["l2"] = gan_regulizer.l2_coef
        self.mulit_objectives = MultipleObjectives(
                name, inputs,
                outputs_map=outputs_map,
                params=gan.G.params,
                objectives=[v.g_loss, cond_loss],
                additional_updates=v.d_updates + v.reg_updates)

    def compile(self):
        self.gan._compile_generate(self.build_dict)
        self.mulit_objectives.compile(self.optimizer_fn)
