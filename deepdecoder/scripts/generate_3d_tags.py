#! /usr/bin/env python3
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


import matplotlib
matplotlib.use('Agg')  # noqa

from deepdecoder.data import generator_3d_tags_with_depth_map
from beras.transform import tile
import matplotlib.pyplot as plt
import os
import h5py
import argparse
from keras.utils.generic_utils import Progbar
from scipy.ndimage.interpolation import zoom
from scipy.ndimage.filters import gaussian_filter1d
from scipy.misc import imsave
from deepdecoder.scripts.default_3d_tags_distribution import default_tag_distribution


def generator(tag_dist, batch_size, antialiasing=1):
    s = antialiasing
    depth_scale = 1/2
    for param, mask, depth_map in generator_3d_tags_with_depth_map(
            tag_dist, batch_size, antialiasing=s, depth_scale=depth_scale):
        depth_map = gaussian_filter1d(depth_map, 2/6/depth_scale, axis=-1, mode='constant')
        depth_map = gaussian_filter1d(depth_map, 2/6/depth_scale, axis=-2, mode='constant')
        depth_map = zoom(depth_map, (1., 1., depth_scale, depth_scale))
        yield param, mask, depth_map


def plot_anitaliasing(tag_dist, fname, a, nb_samples=64):
    _, masks, depth_map = next(generator(tag_dist, nb_samples, antialiasing=a))
    tiled = tile(masks)[0]
    imsave(fname.format(a), tiled)


def run(tag_dist, output_fname, force, nb_samples):
    os.makedirs(os.path.dirname(output_fname), exist_ok=True)
    if os.path.exists(output_fname) and force:
        print("Deleted {}".format(output_fname))
        os.remove(output_fname)
    else:
        assert not os.path.exists(output_fname), \
            "File {} already exists. Use --force to override it"
    basename, _ = os.path.splitext(output_fname)
    anit_name = basename + "_anti_{}.png"
    hist_name = basename + "_hist_{}.png"
    plot_anitaliasing(tag_dist, anit_name, 1)
    plot_anitaliasing(tag_dist, anit_name, 2)
    plot_anitaliasing(tag_dist, anit_name, 4)
    plot_anitaliasing(tag_dist, anit_name, 8)

    labels, masks, _ = next(generator(tag_dist, 10000, antialiasing=2))
    for key in labels.dtype.names:
        m = labels[key].mean()
        s = labels[key].std()
        print("{}: {:.3f}, {:.3f}".format(key, m, s))
        assert abs(m) <= 0.03

    for label_name in sorted(set(labels.dtype.names) - set(['bits'])):
        x = labels[label_name]
        plt.hist(x.flatten(), bins=40, normed=True)
        plt.savefig(hist_name.format(label_name))
        plt.clf()

    f = h5py.File(output_fname, 'w')
    shape = (64, 64)
    f.create_dataset("tags", shape=(nb_samples, 1, shape[0], shape[1]), dtype='float32',
                     chunks=(256, 1, shape[0], shape[1]), compression='gzip')
    f.create_dataset("depth_map", shape=(nb_samples, 1, 16, 16), dtype='float32',
                     chunks=(256, 1, 16, 16), compression='gzip')
    for label_name in labels.dtype.names:
        n_values = labels[label_name].shape[1]
        f.create_dataset(label_name, shape=(nb_samples, n_values), dtype='float32',
                         chunks=(256, n_values), compression='gzip')

    f.attrs['label_names'] = [l.encode('utf8') for l in labels.dtype.names]
    print(f.attrs['label_names'])
    i = 0
    progbar = Progbar(nb_samples)
    batch_size = min(20000, nb_samples)
    for labels, tags, depth_map in generator(tag_dist, batch_size, antialiasing=4):
        size = min(i+batch_size, nb_samples) - i
        if size == 0:
            break
        f['tags'][i:i+size] = tags[:size]
        f['depth_map'][i:i+size] = depth_map[:size]
        for label_name in labels.dtype.names:
            f[label_name][i:i+size] = labels[label_name][:size]
        i += size
        progbar.update(i)

    f.flush()
    print("Saved images to: {}".format(output_fname))
    dist_fname = basename + "_distribution.json"
    with open(dist_fname, "w+") as f:
        f.write(tag_dist.to_json())
        print("Saved distribution to: {}".format(dist_fname))


def main():
    parser = argparse.ArgumentParser(
        description='Generate images and depth maps from the 3d object model of the tag')
    parser.add_argument('output', type=str, help='output file name')
    parser.add_argument('-f', '--force', action='store_true',
                        help='override existing output file')
    parser.add_argument('-d', '--dist', type=str, default=default_tag_distribution(),
                        help='Json params of the distribution')
    parser.add_argument('-n', '--nb-samples', type=float, required=True,
                        help='Number of samples to generate')
    args = parser.parse_args()
    run(args.dist, args.output, args.force, int(args.nb_samples))

if __name__ == "__main__":
    main()
