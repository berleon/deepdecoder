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


import matplotlib.pyplot as plt
import numpy as np
from itertools import combinations


def plt_hist(x, label, num_bins=50):
    hist, bins = np.histogram(x, bins=num_bins)
    width = 0.7 * (bins[1] - bins[0])
    center = (bins[:-1] + bins[1:]) / 2
    plt.bar(center, hist, align='center', width=width, label=label, alpha=0.5)


def plot_multi_objective_grads(params, grads):
    for i, grad_dict in enumerate(grads):
        fig = plt.figure()
        print(params[i])
        print(i)
        fig.add_subplot(2, 1, 2)
        for plt_idx, (name, grad) in enumerate(grad_dict.items(), 1):
            plt.title(name)
            plt_hist(grad, name)
        plt.show()
        print("multiplication")
        fig.add_subplot(2, 1, 2)
        for plt_idx, ((name_a, a), (name_b, b)) in enumerate(
             combinations(grad_dict.items(), 2), 1):
            plt.title(name)
            plt_hist(a*b, "{}-{}".format(name_a, name_b))
        plt.show()
