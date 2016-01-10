#! /usr/bin/env python
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

import pytest
from deepdecoder.model_utils import plot_weights_histogram, \
    add_uniform_noise
from keras.models import Sequential
from keras.layers.core import Dense


def test_plot_weights_histogram():
    model = Sequential()
    model.add(Dense(1000, input_dim=20))
    model.add(Dense(1000))
    plot_weights_histogram(model)


def test_add_uniform_noise():
    model = Sequential()
    model.add(Dense(1000, input_dim=20, init='zero'))
    model.add(Dense(1000, init='zero'))
    add_uniform_noise(model, 0.5)
    for layer in model.layers:
        for weight in layer.get_weights():
            assert (weight != 0).any()


if __name__ == "__main__":
    pytest.main(__file__)
