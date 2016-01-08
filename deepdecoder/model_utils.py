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

import numpy as np
import matplotlib.pyplot as plt



def add_uniform_noise(model, std):
    for layer in model.layers:
        weights = layer.get_weights()
        for weight in weights:
            weight += np.random.normal(0, std, weight.shape)
        layer.set_weights(weights)


def weights_histogram(model, bins=50):
    hists = []
    for i, layer in enumerate(model.layers):
        name = str(type(layer)) + "_{}".format(i)
        weights = layer.get_weights()
        for wi, weight in enumerate(weights):
            hist, bin_edges = np.histogram(weight, bins, density=True)
            hists.append((
                name + "_{}".format(wi),
                hist, bin_edges
            ))
    return hists


def plot_weights_histogram(model, bins=50):
    hists = weights_histogram(model, bins)
    for name, hist, bin_edges in hists:
        plt.title(name)
        print(bin_edges[1:].shape)
        print(hist.shape)
        plt.plot(bin_edges[1:], hist)
        plt.show()

