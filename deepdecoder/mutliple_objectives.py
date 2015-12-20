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
import keras
import keras.constraints
from beras.models import AbstractModel

import keras.backend as K
from dotmap import DotMap
from keras import optimizers


class MultipleObjectives(AbstractModel):
    def __init__(self, name, inputs, outputs_map, params,
                 objectives,
                 constraints=None,
                 additional_updates=None,
                 debug_map=None,
                 output_loss=True):
        self.name = name
        self.inputs = inputs
        self.outputs_map = outputs_map
        self.objectives = objectives
        self.params = params
        if constraints is None:
            constraints = [keras.constraints.identity()
                           for _ in range(len(self.params))]
        self.constraints = constraints

        self.debug_map = debug_map
        self.output_loss = output_loss
        if additional_updates is None:
            additional_updates = []
        self.additional_updates = additional_updates
        self._train_fn = None
        self._debug_fn = None

    @staticmethod
    def sum_objectives(objectives):
        return sum([o.mean() for o in objectives])

    def compile(self, optimizer_lambda):
        scalar_objective = self.sum_objectives(self.objectives)
        if self.output_loss:
            self.outputs_map[self.name] = scalar_objective
        optimizer = optimizers.get(optimizer_lambda())
        updates = optimizer.get_updates(self.params, self.constraints,
                                        scalar_objective)
        self._train_fn = K.function(
                self.inputs, list(self.outputs_map.values()),
                updates=updates + self.additional_updates)
        if self.debug_map:
            self._debug_fn = K.function(
                self.inputs, list(self.debug_map.values()),
                updates=updates + self.additional_updates)

    def fit(self, inputs, batch_size=128, nb_epoch=100, verbose=0,
            nb_iterations=None,
            callbacks=None, shuffles=True):

        if callbacks is None:
            callbacks = []
        if type(shuffles) == bool:
            if shuffles:
                shuffles = [True] * len(inputs)
            else:
                shuffles = [False] * len(inputs)

        assert len(inputs) >= 1 or nb_iterations is not None
        assert len(inputs) == len(shuffles)
        if nb_iterations is None:
            nb_samples = len(inputs[0])
        else:
            nb_samples = nb_iterations * batch_size

        def train(model, batch_indicies, batch_index, batch_logs=None):
            if batch_logs is None:
                batch_logs = {}

            batch_ins = []
            s = batch_size//2*batch_index
            e = s + batch_size//2
            for shuffle, arr in zip(shuffles, inputs):
                if shuffle:
                    batch_ins.append(arr[batch_indicies])
                else:
                    batch_ins.append(arr[s:e])
            if len(batch_ins) == 0:
                outs = self._train_fn([])
            else:
                outs = self._train_fn(batch_ins)

            if type(outs) != list:
                outs = [outs]
            for key, value in zip(self.outputs_map.keys(), outs):
                batch_logs[key] = float(value)

        self._fit(train, nb_samples, batch_size=batch_size, nb_epoch=nb_epoch,
                  verbose=verbose, callbacks=callbacks, shuffle=shuffles,
                  metrics=self.outputs_map.keys())

    def debug(self, inputs):
        outs = self._debug_fn(*inputs)
        out_map = DotMap()
        for name, out in zip(self.debug_map.keys(), outs):
            out_map[name] = out
        return out_map
