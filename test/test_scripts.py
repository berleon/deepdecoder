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

from conftest import plt_save_and_maybe_show
from deepdecoder.scripts.train_decoder import _nb_samples_per_iterator


def test_nb_sample_per_iterator():
    bs = 32
    factor = 10
    pos = [0]*10
    nb_samples = _nb_samples_per_iterator(bs*factor, pos)
    assert sum(nb_samples) == bs
    assert all([n <= bs / 2 for n in nb_samples])

    pos = [bs*factor]*10
    pos[0] = bs*(factor-1)
    nb_samples = _nb_samples_per_iterator(bs*factor, pos)
    assert sum(nb_samples) == bs
    assert nb_samples[0] == bs
    assert all([n == 0 for n in nb_samples[1:]])
