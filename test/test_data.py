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

from math import pi
import numpy as np
from deepdecoder.data import normalize_angle, bins_for_z, generated_3d_tags
from beesgrid import BlackWhiteArtist
from pipeline.distributions import examplary_tag_distribution, DistributionCollection


def test_data_normalize_angle():
    z = np.array([pi, -pi/2, 3*pi, 2*pi, 4.5*pi])
    z_norm = normalize_angle(z)
    assert (0 <= z_norm).all()
    assert (z_norm < 2*pi).all()
    assert z_norm[0] == pi
    assert z_norm[1] == 1.5*pi
    assert z_norm[2] == pi
    assert z_norm[3] == 0
    assert abs(z_norm[4] - 0.5*pi) <= 1e-7


def test_data_bins_for_z():
    n = 256
    z = np.random.uniform(-4*pi, 4*pi, (n,))
    z_norm = normalize_angle(z, lower_bound=-pi/4)
    bins, diffs = bins_for_z(z)
    assert (bins.astype(np.int) == bins).all()
    assert (0 <= bins).all() and (bins <= 3).all()
    assert (abs(pi/2*bins - z_norm) <= pi / 4).all()


def test_generated_3d_tags():
    artist = BlackWhiteArtist(0, 255, 0, 1)
    label_dist = DistributionCollection(examplary_tag_distribution())
    bs = 8
    labels, grids = generated_3d_tags(label_dist, batch_size=128, artist=artist)
    assert grids.shape == (bs, 1, 64, 64)
    for label_key in labels.dtype.names:
        assert label_key in label_dist.keys
