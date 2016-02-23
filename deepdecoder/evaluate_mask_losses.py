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
from beesgrid import gt_grids, draw_grids, MaskGridArtist, NUM_MIDDLE_CELLS, \
    CONFIG_CENTER, CONFIG_ROTS

import numpy as np
import matplotlib.pyplot as plt
from beesgrid import TAG_SIZE
from deepdecoder.utils import np_binary_mask


def evaluate_loss(gt_files, name, loss_fn, loss_dict, visualise=True,
                  bit_flips=0, translation=0, z_rotation=0):
    def outs_to_dict(outs, loss_dict):
        num_visual = len(loss_dict.visual)
        num_shape = len(loss_dict.shape)
        visual_dict = dict(zip(loss_dict.visual.keys(), outs[:num_visual]))
        shape_dict = dict(zip(loss_dict.shape.keys(),
                              outs[num_visual:num_visual+num_shape]))
        print_dict = dict(zip(loss_dict.print.keys(),
                              outs[num_visual+num_shape:]))
        return visual_dict, shape_dict, print_dict

    def alter_params(ids, configs):
        bs = len(configs)
        new_ids = []
        for id in ids:
            flipped_bits = np.random.choice(
                NUM_MIDDLE_CELLS, bit_flips, replace=False)
            id[flipped_bits] += 1
            id[id == 2] = 0
            new_ids.append(id)
        angle = np.random.uniform(-np.pi, np.pi, (bs, 1))
        configs[:, CONFIG_CENTER] += translation * \
            np.concatenate((np.cos(angle), np.sin(angle)), axis=1)
        configs[:, CONFIG_ROTS[0]] += z_rotation
        return np.stack(new_ids), configs

    nb_views = 16
    if type(visualise) == int:
        nb_views = visualise
        visualise = True

    batch_size = 8
    loss = 0
    num_batches = 0
    for gt, ids, configs in gt_grids(gt_files, batch_size):
        configs[:, CONFIG_CENTER] = 0
        ids, configs = alter_params(ids, configs)
        masks, = draw_grids(ids, configs, artist=MaskGridArtist())
        outs = loss_fn(masks, gt)
        visual_dict, shape_dict, print_dict = outs_to_dict(outs[1:], loss_dict)
        if visualise:
            out_dict = {"1-masks": np_binary_mask(masks), "0-img": gt}
            visual_dict.update(out_dict)
            visualise_loss(visual_dict, shape_dict, print_dict, nb_views)
            visualise = False
        l = outs[0]
        loss += l
        num_batches += 1


    loss /= num_batches
    return loss


def visualise_loss(out_dict, shape_dict, print_dict, nb_views=16):
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

        for (name, out) in sorted(shape_dict.items()):
            print("{}: {}".format(name, out.shape))

        for (name, out) in sorted(print_dict.items()):
            print("{}: {}".format(name, out[i]))
        plt.show()


def benchmark_loss_fn(loss_fn, *args, **kwargs):
    batch_size = 128
    mask = theano.shared(np.random.sample(
        (batch_size, 1, TAG_SIZE, TAG_SIZE)))
    predicted = T.tensor4()
    loss_dict = loss_fn(mask, predicted, *args, **kwargs)
    outs = [loss_dict.loss]
    return theano.function([mask, predicted], outs), loss_dict


def compile_loss_fn(loss_fn, visualise=True, *args, **kwargs):
    mask = T.tensor4()
    predicted = T.tensor4()
    loss_dict = loss_fn(mask, predicted, *args, **kwargs)
    outs = [loss_dict.loss]
    if visualise:
        outs.extend(loss_dict.visual.values())
        outs.extend(loss_dict.shape.values())
        outs.extend(loss_dict.print.values())
    return theano.function([mask, predicted], outs), loss_dict
