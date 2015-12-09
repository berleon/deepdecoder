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
import numpy as np
import skimage.io
from skimage import draw
from beesgrid.generate_grids import batches, GridGenerator
from generate_background import BackgroundGenerator


def test_generate_background():
    rows, cols = 25, 25
    size = 64
    big_image = np.zeros((rows * size + rows - 1, cols * size + cols - 1),
                         dtype=np.uint8)

    grid_gen = GridGenerator()
    next_batch = lambda: next(batches(rows*cols, generator=grid_gen, with_gird_params=True))
    bg_gen = BackgroundGenerator()

    grids, labels, grid_params = next_batch()
    print(grids.shape)
    for r in range(rows):
        off_x = r*size + r
        for c in range(cols):
            i = r*rows + c
            off_y = c*size + c
            arr = np.zeros((size, size), dtype=np.uint8)
            z_angle = grid_params[i, 0]
            bg_gen.draw(arr, z_angle)
            defined = grids[i, 0] != 0
            arr[defined] = grids[i, 0, defined]
            big_image[off_x:off_x+size, off_y:off_y + size] = bg_gen.blur(arr)


    skimage.io.imsave('generated.png', big_image)
    skimage.io.imsave('generated_contrast.png', np.minimum(big_image*1.5, 255).astype(np.uint8))

