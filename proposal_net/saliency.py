#! /usr/bin/env python3
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
from skimage.draw import circle
from skimage.draw._draw import circle_perimeter

from proposal_net import SaliencyNetwork, sample_locations
import matplotlib.pyplot as plt
import numpy as np


def rrccLoc(loc, shape=None):
    return circle(loc[0], loc[1], 5, shape=shape)

if __name__ == "__main__":
    net = SaliencyNetwork("saved_saliency_net.hdf5")
    sal, resized = net.saliency_file("/home/leon/uni/vision_swp/deeplocalizer_data/images/season_2015/cam1/groundtruth/Cam_1_20150821161518_254545.jpeg")
    plt.set_cmap('gray')
    plt.subplot(221)
    slice_x = slice(None)
    slice_y = slice(None)
    plt.imshow(resized[0, 0, slice_x, slice_y])
    locs = sample_locations(sal[0, 0], 2000)
    locations = np.zeros(resized.shape[-2:])
    for x, y in locs:
        rr, cc = rrccLoc((x, y), locations.shape)
        locations[rr, cc] = 10
    plt.subplot(222)
    cmap = plt.get_cmap('coolwarm')
    print(sal.shape)
    print(sal[0, 0, slice_x, slice_y].shape)
    plt.imshow(sal[0, 0, slice_x, slice_y], cmap=cmap)
    plt.colorbar()
    plt.subplot(223)
    plt.imshow(locations[slice_x, slice_y], cmap=cmap)
    plt.show()


