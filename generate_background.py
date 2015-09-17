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
import random
import math
import numpy as np

from skimage import draw
from skimage import filters
from deepdecoder import TAG_SIZE


class BackgroundGenerator(object):
    def __init__(self):
        self.background_range = (0x35, 0x40)
        self.n_circles_range = (25, 60)
        self.circle_radius_range = (3, 8)
        self.head_color_range = (0x12, 0x1A)
        self.circle_color_range = (-0x10, +0x10)
        self.blur_sigma_range = (1.4, 1.9)
    def drawHead(self, arr, z_angle):

        hx = int(20*math.sin(z_angle) + 30)
        hy = int(20*math.cos(z_angle) + 30)
        head_c = (hx, hy)
        head_r = (15, 25)
        rr, cc = draw.ellipse(head_c[0], head_c[1], head_r[0], head_r[1], shape=arr.shape)
        arr[rr, cc] = np.maximum(arr[rr, cc] - random.uniform(*self.head_color_range), 0)

    def setBackground(self, arr):
        arr[:] = random.uniform(*self.background_range)

    def drawCircles(self, arr):
        def gen_circles():
            for i in range(int(random.uniform(*self.n_circles_range))):
                c = 30
                cx, cy = c, c
                while math.sqrt((cx-c)**2 + (cy-c)**2) < 22:
                    cx = random.uniform(0, TAG_SIZE)
                    cy = random.uniform(0, TAG_SIZE)
                r = random.uniform(*self.circle_radius_range)
                yield draw.circle(cx, cy, r, shape=arr.shape)
        for rr, cc in gen_circles():
            arr[rr, cc] = np.maximum(arr[rr, cc] + random.uniform(*self.circle_color_range), 0)

    def blur(self, arr):
        sigma = random.uniform(*self.blur_sigma_range)
        filtered = filters.gaussian_filter(arr, sigma=sigma) * 255
        return filtered.astype(np.uint8)

    def draw(self, arr, z_angle):
        self.setBackground(arr)
        self.drawCircles(arr)
        self.drawHead(arr, z_angle)
        # return self.blur(arr)

