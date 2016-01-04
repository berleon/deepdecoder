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
import math
import theano
import theano.tensor as T
from beesgrid import gt_grids, draw_grids, MaskGridArtist

import matplotlib.pyplot as plt
import seaborn as sns
from deepdecoder import TAG_SIZE

from deepdecoder.utils import np_binary_mask, zip_visualise_tiles


def evaluate_loss(gt_files, name, loss_fn, visualise=True, *args, **kwargs):
    nb_views = 16
    if type(visualise) == int:
        nb_views = visualise
        visualise = True

    print("Evaluating {} with args {} and kwargs {}:".format(name, args,
                                                             kwargs))
    calc_loss, loss_dict = compile_loss_fn(loss_fn, visualise, *args, **kwargs)
    batch_size = 16
    loss = 0
    total_grids = 0
    for gt, ids, configs in gt_grids(gt_files, batch_size):
        masks, = draw_grids(ids, configs, artist=MaskGridArtist())
        outs = calc_loss(masks, gt) * len(gt)

        if visualise:
            out_dict = {"1-masks": np_binary_mask(masks), "0-img": gt}
            for l, o in zip(loss_dict.visual, outs[1:]):
                out_dict[l] = o
            visualise_loss(out_dict, nb_views)
            visualise = False
        l = outs[0]
        loss += l
        total_grids += len(gt)

    loss /= total_grids
    print("Total mean loss is: {}".format(loss))


def visualise_loss(out_dict, nb_views=16):
    bs = None
    size = len(out_dict)
    cols = min(size, 4)
    rows = math.ceil(size / 4)
    for out in out_dict.values():
        if bs is None:
            bs = len(out)
        assert bs == len(out)
    for i in range(nb_views):
        print(i)
        fig = plt.figure(figsize=(18, 10))
        for plt_i, (name, out) in enumerate(sorted(out_dict.items())):
            fig.add_subplot(rows, cols, plt_i+1)
            plt.imshow(out[i, 0], cmap='gray')
            plt.colorbar(fraction=0.046, pad=0.04)
            plt.title(name)
        plt.show()


def benchmark_loss_fn(loss_fn, *args, **kwargs):
    batch_size = 128
    mask = theano.shared(np.random.sample((batch_size, 1, TAG_SIZE,
                                             TAG_SIZE)))
    predicted = T.tensor4()
    loss_dict = loss_fn(mask, predicted, *args, **kwargs)
    outs = [loss_dict.loss]
    if visualise:
        outs.extend(loss_dict.visual.values())
    return theano.function([mask, predicted], outs), loss_dict


def compile_loss_fn(loss_fn, visualise=True, *args, **kwargs):
    mask = T.tensor4()
    predicted = T.tensor4()
    loss_dict = loss_fn(mask, predicted, *args, **kwargs)
    outs = [loss_dict.loss]
    if visualise:
        outs.extend(loss_dict.visual.values())
    return theano.function([mask, predicted], outs), loss_dict

