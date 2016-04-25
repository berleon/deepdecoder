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
from deepdecoder.evaluate import GTEvaluator
from beesgrid import NUM_MIDDLE_CELLS, CONFIG_LABELS, get_gt_files_in_dir


def test_evaluation():
    def predict(x):
        return np.random.uniform(size=(len(x), NUM_MIDDLE_CELLS)), \
               np.random.uniform(size=(len(x), len(CONFIG_LABELS)))

    gt_dir = '/home/leon/repos/deeplocalizer_data/images/season_2015'
    gt_files = get_gt_files_in_dir(gt_dir)
    evaluator = GTEvaluator(gt_files)
    results = evaluator.evaluate(predict)
