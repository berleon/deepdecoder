# Copyright 2016 Leon Sixt
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

from beesgrid import MaskGridArtist, DepthMapArtist, draw_grids

import numpy as np
import h5py

from beesgrid import MASK, BlackWhiteArtist, TAG_LABEL_NAMES
from diktya.data_utils import HDF5Tensor
from itertools import count
import diktya.distributions
import scipy.ndimage.interpolation
import scipy.ndimage


def np_binary_mask(mask, black=0., ignore=0.5,  white=1.):
    bw = ignore * np.ones_like(mask, dtype=np.float32)
    bw[mask > MASK["IGNORE"]] = white
    bw[mask < MASK["IGNORE"]] = black
    return bw


def load_real_hdf5_tags(fname, batch_size):
    h5 = h5py.File(fname, 'r')
    nb_tags = h5['tags'].shape[0]
    nb_tags = (nb_tags // batch_size)*batch_size
    tags = HDF5Tensor(fname, 'tags', 0, nb_tags)
    assert len(tags) % batch_size == 0
    return tags


def real_generator(hdf5_fname, nb_real, use_mean_image=False, range=(0, 1)):
    low, high = range
    tags = load_real_hdf5_tags(hdf5_fname, nb_real)
    nb_tags = len(tags)
    print("Got {} real tags".format(nb_tags))
    mean_end = min(nb_tags, 2000)
    mean_image = (tags[0:mean_end] / 255).mean(axis=0)

    for i in count(step=nb_real):
        ti = i % nb_tags
        tag_batch = tags[ti:ti+nb_real] / 255
        if use_mean_image:
            tag_batch -= mean_image
        yield (high - low)*tag_batch + low


def z_generator(z_shape):
    while True:
        yield np.random.uniform(-1, 1, z_shape).astype(np.float32)


def zip_real_z(real_gen, z_gen):
    for real, z in zip(real_gen, z_gen):
        yield {'real': real, 'z': z}


def resize_mask(masks, order=1, sigma=0.66, scale=0.5):
    resized = []
    for mask in masks:
        smoothed = scipy.ndimage.gaussian_filter(mask[0], sigma=sigma)
        small = scipy.ndimage.interpolation.zoom(smoothed, (scale, scale),
                                                 order=order)
        resized.append(small)
    new_size = int(masks.shape[-1] * scale)
    return np.stack(resized).reshape((len(masks), 1, new_size, new_size))


def generator_3d_tags_with_depth_map(tag_distribution, batch_size=128,
                                     antialiasing=4, depth_scale=1/4):
    ignore = -0.25
    background = 0
    black = 51
    white = 255
    need_size = (1 - ignore)
    scale = 255 / need_size

    bw_artist = BlackWhiteArtist(black, white, background, antialiasing)
    depth_map_artist = DepthMapArtist()
    while True:
        labels = tag_distribution.sample(batch_size)
        grids, = draw_grids(labels, artist=bw_artist)
        depth_maps, = draw_grids(labels, artist=depth_map_artist,
                                 scales=[depth_scale])
        depth_map_f32 = (depth_maps / 255.).astype(np.float32)
        grids_f32 = (grids / scale + ignore).astype(np.float32)
        norm_labels = tag_distribution.normalize(labels)
        yield norm_labels, grids_f32, depth_map_f32


def generated_3d_tags(tag_distribution, batch_size=128, artist=None, scale=1.):
    if artist is None:
        artist = MaskGridArtist()
    labels = tag_distribution.sample(batch_size)
    grids = draw_grids(labels, scales=[scale], artist=artist)
    assert len(grids) == 1
    return labels, grids[0]



class HDF5Dataset(h5py.File):
    def __init__(self, name, nb_samples=None, **kwargs):
        super().__init__(name, **kwargs)
        self.dataset_created = '__dataset_created' in self.attrs
        if nb_samples is None:
            if '__nb_samples' not in self.attrs:
                raise Exception("Argument nb_samples not given and not found in hdf5 file.")
            self.nb_samples = self.attrs['__nb_samples']
        else:
            self.nb_samples = nb_samples
            self.attrs['__nb_samples'] = nb_samples
        self.chunk_size = 64*64*256*4   # ~4MB
        self.create_dataset_kwargs = {
            'compression': 'gzip'
        }
        self._append_pos = None

    @staticmethod
    def _nearest_power_of_two(x):
        return int(2**np.round(np.log(x) / np.log(2)))

    def _create_dataset(self, **kwargs):
        if self.dataset_created:
            raise Exception("Datasets allready created.")

        self.attrs['dataset_names'] = [k.encode('utf-8') for k in kwargs.keys()]
        for name, array in kwargs.items():
            shape = array.shape[1:]
            size_one_elem = np.prod(shape)
            nb_chunks = self._nearest_power_of_two(self.chunk_size / size_one_elem)
            nb_chunks = min(nb_chunks, self.nb_samples)
            self.create_dataset(name,
                                shape=(self.nb_samples,) + shape,
                                dtype=str(array.dtype),
                                chunks=(nb_chunks,) + shape,
                                compression='gzip')
        self.dataset_created = True
        self._append_pos = 0

    def append(self, **kwargs):
        """
        Append the arrays to the hdf5 file. Must always be called with the
        same keys for one HDF5Dataset.

        Args:
            **kwargs: Dictonary of name to numpy array.
        """

        if not self.dataset_created:
            self._create_dataset(**kwargs)
        if self._append_pos >= self.nb_samples:
            raise Exception("Dataset is allready full!")

        batch_size = len(next(iter(kwargs.values())))
        begin = self._append_pos
        end = min(self._append_pos+batch_size, self.nb_samples)
        for name, array in kwargs.items():
            if len(array) != batch_size:
                raise Exception("Arrays must have the same number of samples."
                                " Got {} and {}".format(batch_size, len(array)))
            self[name][begin:end] = array
        self._append_pos += batch_size
        return self._append_pos

    def append_generator(self, generator):
        """
        Consumes a generator. The generator must yield dictionaries.
        They are put into the :py:meth:`append`.
        """
        while True:
            self.append(**next(generator))
            if self._append_pos >= self.nb_samples:
                break

    def iter(self, batch_size):
        i = 0
        dataset_names = [n.decode('utf-8') for n in self.attrs['dataset_names']]
        while True:
            yield {
                name: self[name][i:i+batch_size]
                for name in dataset_names
            }
            i = (i + batch_size) % (self.nb_samples - batch_size)


class DistributionHDF5Dataset(HDF5Dataset):
    def __init__(self, name, distribution=None, **kwargs):
        super().__init__(name, **kwargs)

        if distribution is None:
            if 'distribution' not in self.attrs:
                raise Exception("distribution argument not given and not found"
                                " in hdf5 file.")
        else:
            self.attrs['distribution'] = distribution.to_json().encode('utf8')
            self.attrs['label_names'] = [l.encode('utf8') for l in distribution.names]

    def get_tag_distribution(self):
        dist_json = self.attrs['distribution'].decode('utf-8')
        return diktya.distributions.load_from_json(dist_json)

    def get_label_names(self):
        return [l.decode('utf-8') for l in self.attrs['label_names']]

    def append(self, labels, **kwargs):
        for label_name in labels.dtype.names:
            kwargs[label_name] = labels[label_name]
        return super().append(**kwargs)

    def iter(self, batch_size):
        label_names = self.get_label_names()
        dist = self.get_tag_distribution()
        for batch in super().iter(batch_size):
            labels = [batch.pop(name) for name in label_names]
            labels = np.concatenate(labels, axis=1)
            batch['labels'] = labels.astype(dist.norm_dtype)
            yield batch
