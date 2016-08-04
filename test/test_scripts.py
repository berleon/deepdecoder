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
from diktya.distributions import examplary_tag_distribution, DistributionCollection
from deepdecoder.data import DistributionHDF5Dataset
from deepdecoder.scripts.train_decoder import _nb_samples_per_iterator, \
    zip_dataset_iterators, augmentation_data_generator, dataset_iterator, \
    bit_split
import numpy as np
from itertools import count

def test_nb_sample_per_iterator():
    bs = 32
    factor = 10
    pos = [0]*10
    nb_samples = _nb_samples_per_iterator(bs*factor, bs, pos)
    assert sum(nb_samples) == bs
    assert all([n <= bs / 2 for n in nb_samples])

    bs = 32
    factor = 10
    pos = [0]
    nb_samples = _nb_samples_per_iterator(bs*factor, bs, pos)
    assert nb_samples == [bs]

    pos = [bs*factor]*10
    pos[0] = bs*(factor-1)
    nb_samples = _nb_samples_per_iterator(bs*factor, bs, pos)
    assert sum(nb_samples) == bs
    assert nb_samples[0] == bs
    assert all([n == 0 for n in nb_samples[1:]])


def test_zip_dataset_itertors():
    def generator(bs):
        for i in count(0, step=bs):
            yield {
                'data': np.arange(start=i, stop=i+bs),
                'labels': np.arange(start=i, stop=i+bs),
            }

    bs = 32
    for i, batch in enumerate(zip_dataset_iterators([generator], bs)):
        assert (batch['data'] == batch['labels']).all()
        assert (batch['data'] == np.arange(i*bs, (i+1)*bs)).all()
        if i == 20:
            break


def test_augmentation_data_generator(tmpdir):
    dist = DistributionCollection(examplary_tag_distribution())
    dset_fname = str(tmpdir.join("dset.hdf5"))
    samples = 6000
    dset = DistributionHDF5Dataset(dset_fname, nb_samples=samples,
                                   distribution=dist)
    labels = dist.sample(samples)
    labels = dist.normalize(labels)
    fake = np.random.random((samples, 1, 8, 8))
    discriminator = np.random.random((samples, 1))
    dset.append(labels=labels, fake=fake, discriminator=discriminator)
    dset.close()

    dset = DistributionHDF5Dataset(dset_fname)
    bs = 32
    names = ['labels', 'fake']
    assert 'labels' in next(dset.iter(bs, names))
    assert next(dset.iter(bs))['labels'].dtype.names == tuple(dist.names)

    dset_iters = [lambda bs: bit_split(dataset_iterator(dset, bs))]
    data_gen = lambda bs: zip_dataset_iterators(dset_iters, bs)
    label_names = ['bit_0', 'bit_11', 'x_rotation']
    aug_gen = augmentation_data_generator(data_gen, 'fake', label_names)
    outs = next(aug_gen(bs))
    assert len(outs[0]) == 32
    assert len(outs[1]) == len(label_names)

    gen = aug_gen(bs)
    for i, batch in enumerate(gen):
        if i == 2*samples // bs:
            break
        assert batch is not None
        assert batch[0].shape == (bs, 1, 8, 8)
        assert len(batch[1]) == len(label_names)
