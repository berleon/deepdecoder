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


import matplotlib
import os

visual_debug = False

if not visual_debug:
    matplotlib.use('Agg')


import theano
import matplotlib.pyplot as plt
import scipy.misc


def on_gpu():
    return 'gpu' in theano.config.device


TEST_OUTPUT_DIR = "tests_out"
os.makedirs(TEST_OUTPUT_DIR, exist_ok=True)

if not on_gpu():
    collect_ignore = []
    for path, dirs, files in os.walk("test"):
        for f in files:
            if f.startswith("test_gpu_only") and f.endswith(".py"):
                collect_ignore.append(os.path.join(path, f))


def imsave(fname, image):
    abs_fname = os.path.join(TEST_OUTPUT_DIR, fname)
    os.makedirs(os.path.dirname(abs_fname), exist_ok=True)
    scipy.misc.imsave(abs_fname, image)


def plt_save_and_maybe_show(fname):
    abs_fname = os.path.join(TEST_OUTPUT_DIR, fname)
    os.makedirs(os.path.dirname(abs_fname), exist_ok=True)
    plt.savefig(abs_fname)
    if visual_debug:
        plt.show()
    else:
        plt.clf()
