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
from beras.data_utils import HDF5Tensor
from itertools import count
import pipeline.distributions
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


class SynthesisedDataset:
    def __init__(self, fname, mode=None):
        self.fname = fname
        self.f = h5py.File(fname, mode)

    def get_tag_distribution(self):
        dist_json = self.f.attrs['distribution'].decode('utf-8')
        return pipeline.distributions.load_from_json(dist_json)

    def set_tag_distribution(self, distribution):
        self.f.attrs['distribution'] = distribution.to_json().encode('utf8')
        self.f.attrs['label_names'] = [l.encode('utf8') for l in distribution.names]

    def get_label_names(self):
        label_names = [l.decode('utf-8') for l in self.f.attrs['label_names']]
        assert label_names == TAG_LABEL_NAMES


class Tag3dDataset(SynthesisedDataset):
    shape = (64, 64)

    def __init__(self, fname, mode=None):
        super().__init__(fname, mode)
        self._append_pos = 0
        self._max_samples = 0

    def create_datasets(self, nb_samples, labels_size):
        shape = self.shape
        self.f.create_dataset("tags", shape=(nb_samples, 1, shape[0], shape[1]), dtype='float32',
                              chunks=(256, 1, shape[0], shape[1]), compression='gzip')

        self.f.create_dataset("depth_map", shape=(nb_samples, 1, 16, 16), dtype='float32',
                              chunks=(256, 1, 16, 16), compression='gzip')

        for label_name, size in labels_size.items():
            self.f.create_dataset(label_name, shape=(nb_samples, size), dtype='float32',
                                  chunks=(256, size), compression='gzip')
        self._append_pos = 0
        self._max_samples = nb_samples

    def append(self, labels, tags, depth_map):
        i = self._append_pos
        batch_size = len(tags)
        size = min(i+batch_size, self._max_samples) - i
        if size == 0:
            return i

        self.f['tags'][i:i+size] = tags[:size]
        self.f['depth_map'][i:i+size] = depth_map[:size]
        for label_name in labels.dtype.names:
            self.f[label_name][i:i+size] = labels[label_name][:size]

        self._append_pos += size
        return self._append_pos

    def iter(self, batch_size):
        i = 0
        tags = self.f['tags']
        depth_maps = self.f['depth_map']
        nb_samples = len(tags)
        while True:
            masks = tags[i:i+batch_size]
            depth_map = depth_maps[i:i+batch_size]
            labels = []
            for l in TAG_LABEL_NAMES:
                labels.append(self.f[l][i:i+batch_size])
            yield [np.concatenate(labels, axis=1), [masks, depth_map]]
            i += batch_size
            if i >= nb_samples - batch_size:
                i = 0
