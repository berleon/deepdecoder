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

import numpy as np
from deepdecoder.data import generated_3d_tags, \
    HDF5Dataset, DistributionHDF5Dataset
from beesgrid import BlackWhiteArtist
from diktya.distributions import examplary_tag_distribution, DistributionCollection
import pytest

def test_generated_3d_tags():
    artist = BlackWhiteArtist(0, 255, 0, 1)
    label_dist = DistributionCollection(examplary_tag_distribution())
    bs = 8
    labels, grids = generated_3d_tags(label_dist, batch_size=128, artist=artist)
    assert grids.shape == (bs, 1, 64, 64)
    for label_key in labels.dtype.names:
        assert label_key in label_dist.keys


def test_hdf5_dataset(tmpdir):
    dset = HDF5Dataset(str(tmpdir.join('dataset.hdf5')), nb_samples=1000)
    dset.append(name=np.random.random((500, 1, 8, 8)))
    dset.append(name=np.random.random((500, 1, 8, 8)))
    assert dset['name'].dtype == np.float64
    assert dset['name'].shape == (1000, 1, 8, 8)

    dset = HDF5Dataset(str(tmpdir.join('dataset_big.hdf5')), nb_samples=200000)
    dset.append(name=np.random.random((500, 1, 8, 8)))
    dset.append(name=np.random.random((500, 1, 8, 8)))
    assert dset['name'].dtype == np.float64
    assert dset['name'].shape == (200000, 1, 8, 8)
    # between 1MB and 8MB
    assert 1e6 <= np.prod(dset['name'].chunks) <= 8e6


def test_distribution_hdf5_dataset(tmpdir):
    with pytest.raises(Exception):
        DistributionHDF5Dataset(
            str(tmpdir.join('dataset_no_distribution.hdf5')), nb_samples=1000)

    dist = DistributionCollection(examplary_tag_distribution(nb_bits=12))
    labels = dist.sample(32)
    image = np.random.random((32, 1, 8, 8))
    dset = DistributionHDF5Dataset(
        str(tmpdir.join('dataset.hdf5')), distribution=dist, nb_samples=1000)
    dset.append(labels=labels, image=image)
    for name in dist.names:
        assert name in dset
    for batch in dset.iter(batch_size=32):
        for name in dist.names:
            assert name not in batch
        assert 'labels' in batch
        assert batch['labels'].dtype == dist.norm_dtype
        break


def test_hdf5_dataset_nearset_power_of_2(tmpdir):
    assert HDF5Dataset._nearest_power_of_two(2) == 2
    assert HDF5Dataset._nearest_power_of_two(10) == 8
    assert HDF5Dataset._nearest_power_of_two(12) == 16
    assert HDF5Dataset._nearest_power_of_two(13) == 16
    assert HDF5Dataset._nearest_power_of_two(30) == 32
    assert HDF5Dataset._nearest_power_of_two(50) == 64
    assert HDF5Dataset._nearest_power_of_two(250) == 256
    assert HDF5Dataset._nearest_power_of_two(1000) == 1024
    assert HDF5Dataset._nearest_power_of_two(1600) == 2048
