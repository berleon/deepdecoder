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

from test.config import visual_debug

from keras.optimizers import Adam, SGD

from deepdecoder.mutliple_objectives import MultipleObjectives
import keras.backend as K
import numpy as np


def test_multiple_objectives():
    x = K.variable(np.cast[np.float32](20.))
    inputs = []
    parable = x**2
    line = -x + 2
    optima = 0.5
    outputs = {'parable': parable, 'line': line}
    params = [x]
    mo = MultipleObjectives('par-line', inputs, outputs, params,
                            objectives=[parable, line])
    mo.compile(lambda: SGD(lr=0.1))
    mo.fit([], nb_epoch=10, nb_iterations=10, verbose=1)

    x_value = K.get_value(mo.params[0])
    assert np.allclose(x_value, optima, rtol=1e-3)