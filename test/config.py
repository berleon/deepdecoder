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
import os

import matplotlib

visual_debug = False

if not visual_debug:
    matplotlib.use('Agg')

import matplotlib.pyplot as plt

TEST_OUTPUT_DIR = "tests_out"
os.makedirs(TEST_OUTPUT_DIR, exist_ok=True)


def plt_save_and_maybe_show(fname):
    plt.savefig(os.path.join(TEST_OUTPUT_DIR, fname))
    if visual_debug:
        plt.show()
