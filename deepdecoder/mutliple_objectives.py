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
from collections import defaultdict
import keras
import keras.constraints
from beras.models import AbstractModel

import keras.backend as K
from dotmap import DotMap
from keras.optimizers import clip_norm

import copy
import numpy as np
import theano


class MultipleObjectives(AbstractModel):
    def __init__(self, name, inputs, params,
                 objectives,
                 metrics=None,
                 constraints=None,
                 additional_updates=None,
                 debug_map=None,
                 output_loss=True):
        """
        :param name: name of the objective
        :param inputs: dict of input variables
        :param objectives: {name: variable} objectives to optimize
        :param metrics: {label: variable} print this values during training
        :param params: list of all weights that will be optimized
        """
        self.name = name
        self.inputs = inputs
        self.input_order = list(sorted(iter(self.inputs.keys())))
        if metrics is None:
            metrics = {}
        self.metrics = metrics

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

    def compile(self, multi_optimizer):
        updates = multi_optimizer.get_updates(
            self.params, self.constraints, self.objectives)

        ins = [self.inputs[name] for name in self.input_order]
        self._train_fn = K.function(
            ins, list(self.metrics.values()),
            updates=updates + self.additional_updates)
        if self.debug_map:
            self._debug_fn = K.function(ins, list(self.debug_map.values()))

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

        def train(model, batch_index, batch_logs=None):
            if batch_logs is None:
                batch_logs = {}

            batch_ins = []
            s = batch_size//2*batch_index
            e = s + batch_size//2
            for arr in zip(inputs):
                batch_ins.append(arr[s:e])
            outs = self._train_fn(batch_ins)

            if type(outs) != list:
                outs = [outs]
            for key, value in zip(self.metrics.keys(), outs):
                batch_logs[key] = float(value)

        nb_batches = nb_samples // batch_size
        self._fit(train, nb_samples, nb_batches=nb_batches,
                  batch_size=batch_size, nb_epoch=nb_epoch,
                  verbose=verbose, callbacks=callbacks, shuffle=shuffles,
                  metrics=sorted(self.metrics.keys()))

    def fit_generator(self, generator, samples_per_epoch,
                      nb_epoch, batch_size=128, verbose=1, callbacks=[]):
        def train(model, batch_ids, batch_index, batch_logs=None):
            inputs = next(generator)
            outs = self._train_fn(inputs)
            for key, value in zip(self.metrics.keys(), outs):
                batch_logs[key] = float(value)

        self._fit(train, samples_per_epoch, batch_size=batch_size,
                  nb_epoch=nb_epoch, verbose=verbose, callbacks=callbacks,
                  shuffle=False, metrics=sorted(self.metrics.keys()))

    def compile_get_grads(self):
        scalar_objective = self.sum_objectives(self.objectives)
        self.grads_positions = [{} for _ in self.params]
        objectives = copy.copy(self.objectives)
        objectives[self.name] = scalar_objective
        self.evaluate_objective_names = list(objectives.keys())
        gradients = []
        for name, objective in objectives.items():
            for i, grad in enumerate(K.gradients(objective, self.params)):
                self.grads_positions[i][name] = len(gradients)
                gradients.append(grad)
        self._gradients_fn = theano.function(self.inputs, gradients)

    def get_grads(self, inputs):
        assert hasattr(self, '_gradients_fn'), \
            "Did you call compile_evaluate_grads?"
        outs = self._gradients_fn(*inputs)
        gradients = []
        for out_pos_dict in self.grads_positions:
            gradients.append({name: np.array(outs[pos])
                              for name, pos in sorted(out_pos_dict.items())})
        return gradients

    def debug(self, inputs):
        outs = self._debug_fn(*inputs)
        out_map = DotMap()
        for name, out in zip(self.debug_map.keys(), outs):
            out_map[name] = out
        return out_map


class MultiOptimizer:
    '''Abstract optimizer base class.

    Note: this is the parent class of all optimizers, not an actual optimizer
    that can be used for training models.

    All Keras optimizers support the following keyword arguments:

        clipnorm: float >= 0. Gradients will be clipped
            when their L2 norm exceeds this value.
        clipvalue: float >= 0. Gradients will be clipped
            when their absolute value exceeds this value.
    '''
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        self.updates = []

    def get_state(self):
        return [K.get_value(u[0]) for u in self.updates]

    def set_state(self, value_list):
        assert len(self.updates) == len(value_list)
        for u, v in zip(self.updates, value_list):
            K.set_value(u[0], v)

    def get_updates(self, params, constraints, losses):
        raise NotImplementedError

    def get_gradients(self, loss, params):
        grads = K.gradients(loss, params)
        if hasattr(self, 'clipnorm') and self.clipnorm > 0:
            norm = K.sqrt(sum([K.sum(K.square(g)) for g in grads]))
            grads = [clip_norm(g, self.clipnorm, norm) for g in grads]
        if hasattr(self, 'clipvalue') and self.clipvalue > 0:
            grads = [K.clip(g, -self.clipvalue, self.clipvalue) for g in grads]
        return grads

    def get_config(self):
        return {"name": self.__class__.__name__}


class SumLossOptimizer(MultiOptimizer):
    def __init__(self, optimizer, weights=None):
        self.optimizer = optimizer
        if weights is None:
            weights = defaultdict(lambda: 1)
        self.weights = weights

    def get_updates(self, params, constraints, losses):
        summed_loss = sum([self.weights[name]*loss
                           for name, loss in losses.items()])
        return self.optimizer.get_updates(params, constraints, summed_loss)


class ParetoOptimizer():
    def __init__(self, optimizers):
        self.optimizers = optimizers

    def get_updates(self, params, constraints, losses):
        updates_per_params = [{} for _ in params]
        for name, loss in losses.items():
            optimizer = self.optimizers[name]
            updates = optimizer.get_updates(params, constraints, loss)
            for update in updates_per_params:
                updates_per_params[name] = update

        for updates in updates_per_params:
            for name, (var, new_value) in updates.items():
                pass
