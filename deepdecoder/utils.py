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

import theano
from beesgrid.generate_grids import MASK, MASK_BLACK, MASK_WHITE, \
    GridGenerator, MaskGridArtist
import beesgrid.generate_grids as gen_grids
import h5py
import numpy as np
import theano.tensor as T
floatX = theano.config.floatX


def binary_mask(mask, ignore_value=0.5):
    bw = ignore_value*T.ones_like(mask, dtype=floatX)
    bw = T.set_subtensor(bw[(mask > MASK["IGNORE"]).nonzero()], 1.)
    bw = T.set_subtensor(bw[(mask < MASK["BACKGROUND_RING"]).nonzero()], 0)
    return bw

def masks(batch_size, scales=[1.]):
    mini_batch = 64
    batch_size += mini_batch - (batch_size % mini_batch)
    generator = GridGenerator()
    artist = MaskGridArtist()
    for masks in gen_grids.batches(batch_size, generator, artist=artist,
                                   with_gird_params=True, scales=scales):
        yield (masks[0].astype(floatX),) + masks[1:]


def tags_from_hdf5(fname):
    tags_list = []
    with open(fname) as pathfile:
        for hdf5_name in pathfile.readlines():
            hdf5_fname = os.path.dirname(fname) + "/" + hdf5_name.rstrip('\n')
            f = h5py.File(hdf5_fname)
            data = np.asarray(f["data"], dtype=np.float32)
            labels = np.asarray(f["labels"], dtype=np.float32)
            tags = data[labels == 1]
            tags /= 255.
            tags_list.append(tags)

    return np.concatenate(tags_list)


def loadRealData(fname):
    X = tags_from_hdf5(fname)
    X = X[:(len(X)//64)*64]
    return X
