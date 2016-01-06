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

from beesgrid.pybeesgrid import BlackWhiteArtist
from os.path import join
from scipy.misc import imsave

from test.config import TEST_OUTPUT_DIR
from beesgrid import CONFIG_ROTS, CONFIG_RADIUS
from deepdecoder.utils import tile
import numpy as np


def test_curriculum_config():
    from deepdecoder.adaptive_sampling import _config, config_distributions

    batch_size = 4096
    d = config_distributions(0)
    configs = _config(d, batch_size)
    assert configs[:, CONFIG_ROTS].mean() <= 1e-5
    assert abs(configs[:, CONFIG_RADIUS].mean() - d.radius_mean) <= 0.05

    for h in (0, 0.1, 0.5, 0.7, 1):
        d = config_distributions(h)
        configs = _config(d, batch_size)
        assert configs[:, CONFIG_ROTS].mean() <= 0.05
        assert (configs[:, CONFIG_ROTS[1]].std() - d.y_std) <= 0.05
        assert (configs[:, CONFIG_ROTS[2]].std() - d.x_std) <= 0.05
        assert abs(configs[:, CONFIG_RADIUS].mean() - d.radius_mean) <= 0.05


def test_curriculum_ids():
    from deepdecoder.adaptive_sampling import _sample_ids, curriculum_grids
    batch_size = 128
    output_dir = join(TEST_OUTPUT_DIR, "curriculum_grids")
    os.makedirs(output_dir, exist_ok=True)
    for h in np.linspace(0, 1, num=11):
        _, grids = curriculum_grids(h, batch_size, artist=BlackWhiteArtist())
        grid_img = tile(grids)
        print(grid_img.shape)

        imsave(join(output_dir, "{}.png".format(str(h).replace('.', ''))),
               grid_img[0])




