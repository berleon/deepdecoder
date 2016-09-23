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
    nb_tags = h5['rois'].shape[0]
    nb_tags = (nb_tags // batch_size)*batch_size
    tags = HDF5Tensor(fname, 'rois', 0, nb_tags)
    assert len(tags) % batch_size == 0
    return tags


def real_generator(hdf5_fname, batch_size, range=(0, 1)):
    low, high = range
    tags = load_real_hdf5_tags(hdf5_fname, batch_size)
    nb_tags = len(tags)
    print("Got {} real tags".format(nb_tags))
    for i in count(step=batch_size):
        ti = i % nb_tags
        assert ti + batch_size <= nb_tags, \
            "end: {}, nb_tags: {}".format(ti + batch_size, nb_tags)
        tag_batch = tags[ti:ti+batch_size] / 255.
        tag_batch = (high - low)*tag_batch + low
        yield tag_batch


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
    def __init__(self, name, nb_samples=None, chunks=None, **kwargs):
        super().__init__(name, **kwargs)
        self.dataset_created = '__dataset_created' in self.attrs
        if nb_samples is None:
            if '__nb_samples' in self.attrs:
                self.nb_samples = self.attrs['__nb_samples']
            else:
                self.nb_samples = None
        else:
            self.nb_samples = nb_samples
            self.attrs['__nb_samples'] = nb_samples

        if nb_samples is None and chunks is None:
            chunks = 256
        self.chunks = chunks
        self._append_pos = None

    @staticmethod
    def _nearest_power_of_two(x):
        return int(2**np.round(np.log(x) / np.log(2)))

    def dataset_names(self):
        if not self.dataset_created:
            raise Exception("No Datasets  created.")
        if not hasattr(self, '_dataset_names'):
            self._dataset_names = [n.decode('utf-8')
                                   for n in self.attrs['dataset_names']]
        return self._dataset_names

    def _create_dataset(self, **kwargs):
        if self.dataset_created:
            raise Exception("Datasets allready created.")

        self.attrs['dataset_names'] = [k.encode('utf-8') for k in kwargs.keys()]
        for name, array in kwargs.items():
            shape = array.shape[1:]
            if self.nb_samples:
                self.create_dataset(name,
                                    shape=(self.nb_samples,) + shape,
                                    dtype=str(array.dtype))
            else:
                self.create_dataset(name,
                                    shape=(1,) + shape,
                                    chunks=(self.chunks,) + shape,
                                    maxshape=(None,) + shape,
                                    dtype=str(array.dtype))

        self.dataset_created = True
        self._append_pos = 0

    def _ensure_enough_space_for(self, size):
        for name in self.dataset_names():
            if len(self[name]) < size:
                self[name].resize(size, axis=0)

    def append(self, **kwargs):
        """
        Append the arrays to the hdf5 file. Must always be called with the
        same keys for one HDF5Dataset.

        Args:
            **kwargs: Dictonary of name to numpy array.
        """

        if not self.dataset_created:
            self._create_dataset(**kwargs)
        if self.nb_samples and self._append_pos >= self.nb_samples:
            raise Exception("Dataset is allready full!")

        batch_size = len(next(iter(kwargs.values())))
        begin = self._append_pos
        if self.nb_samples:
            end = min(begin+batch_size, self.nb_samples)
        else:
            end = begin+batch_size

        nb_from_batch = end - begin
        for name, array in kwargs.items():
            self._ensure_enough_space_for(end)
            if len(array) != batch_size:
                raise Exception("Arrays must have the same number of samples."
                                " Got {} and {}".format(batch_size, len(array)))
            self[name][begin:end] = array[:nb_from_batch]
        self._append_pos += nb_from_batch
        return self._append_pos

    def append_generator(self, generator):
        """
        Consumes a generator. The generator must yield dictionaries.
        They are put into the :py:meth:`append`.
        """
        while True:
            self.append(**next(generator))
            if self.nb_samples and self._append_pos >= self.nb_samples:
                break

    def iter(self, batch_size, names=None, shuffle=False):
        i = 0
        if names is None:
            names = [n.decode('utf-8') for n in self.attrs['dataset_names']]

        if self.nb_samples is None:
            nb_samples = len(self[names[0]])
        else:
            nb_samples = self.nb_samples

        indicies = np.arange(nb_samples)
        if shuffle:
            np.random.shuffle(indicies)

        while True:
            size = batch_size
            batch = {name: [] for name in names}
            while size > 0:
                nb = min(nb_samples, i + size) - i
                for name in names:
                    idx = indicies[i:i + nb]
                    idx = np.sort(idx)
                    batch[name].append(self[name][idx, :])
                size -= nb
                i = (i + nb) % nb_samples
            yield {name: np.concatenate(arrs) for name, arrs in batch.items()}


def h5_add_distribution(f, distribution):
    if hasattr(f, 'attrs'):
        h5 = f
        close = False
    elif type(f) == str:
        h5 = h5py.File(f)
        close = True
    for name, value in get_distribution_hdf5_attrs(distribution).items():
        h5.attrs[name] = value
    if close:
        h5.close()


def get_distribution_hdf5_attrs(distribution):
    return {
        'distribution': distribution.to_json().encode('utf8'),
        'label_names': [l.encode('utf8') for l in distribution.names]
    }


class DistributionHDF5Dataset(HDF5Dataset):
    def __init__(self, name, distribution=None, **kwargs):
        super().__init__(name, **kwargs)

        if distribution is None:
            if 'distribution' not in self.attrs:
                raise Exception("distribution argument not given and not found"
                                " in hdf5 file.")
        else:
            h5_add_distribution(self, distribution)

    def get_distribution_hdf5_attrs(self):
        return {
            name: self.attrs[name]
            for name in ('distribution', 'label_names')
        }

    def get_tag_distribution(self):
        dist_json = self.attrs['distribution'].decode('utf-8')
        return diktya.distributions.load_from_json(dist_json)

    def get_label_names(self):
        return [l.decode('utf-8') for l in self.attrs['label_names']]

    def append(self, labels, **kwargs):
        for label_name in labels.dtype.names:
            kwargs[label_name] = labels[label_name]
        return super().append(**kwargs)

    def iter(self, batch_size, names=None, shuffle=False):
        label_names = self.get_label_names()
        dist = self.get_tag_distribution()
        if names is None:
            names = [n.decode('utf-8') for n in self.attrs['dataset_names']]
        if 'labels' in names:
            del names[names.index('labels')]
            names += dist.names
            uses_labels = True
        else:
            uses_labels = False
        if set(dist.names) < set(names):
            uses_labels = True
        elif any([n in names for n in dist.names]):
            raise Exception(
                "Got name some label names {}. But not all {}."
                .format(', '.join([n for n in names if n in dist.names]), dist.names))

        for batch in super().iter(batch_size, names, shuffle):
            if uses_labels:
                labels = [(name, batch.pop(name)) for name in label_names]
                batch['labels'] = np.zeros(len(labels[0][1]), dtype=dist.norm_dtype)
                for name, label in labels:
                    batch['labels'][name] = label
            yield batch
