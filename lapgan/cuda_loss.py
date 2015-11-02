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
from __future__ import print_function
from __future__ import absolute_import
import os
import math
import pycuda.driver as pycu
import pycuda.gpuarray as gpuarray
from pycuda.compiler import SourceModule
import pycuda.autoinit

from deepdecoder.generate_grids import MASK, MASK_BLACK, MASK_WHITE

import numpy as np
import theano
import theano.tensor as T
from theano.misc.pycuda_utils import to_gpuarray, to_cudandarray
import theano.sandbox.cuda as thcu
from theano.sandbox.cuda import CudaNdarrayType


mask_split_file = os.path.join(os.path.dirname(__file__), 'cuda/mask_split.cu')
with open(mask_split_file) as f:
    mask_split_kernel_code = f.read()


def shape_ok(shp):
    return shp[2] == shp[3] and shp[2] % 2 == 0 and shp[1] == 1


def contiguouse(x):
    return thcu.basic_ops.gpu_contiguous(
        thcu.basic_ops.as_cuda_ndarray_variable(x))


def pycuda_zeros(arr, shape):
    if arr is None or arr.shape != shape:
        arr = gpuarray.zeros(shape, dtype=np.float32)
    else:
        if type(arr) != gpuarray.GPUArray:
            arr = to_gpuarray(arr)
    pycu.memset_d32(arr.gpudata, 0, arr.size)
    return arr


class SplitMaskGrad(theano.Op):
    __props__ = ()

    def __init__(self, connected=None):
        super().__init__()
        if connected is None:
            self.connected = ["sum", "pow"]
        else:
            self.connected = connected
        if "sum" in self.connected and "pow" in self.connected:
            self.function_name = "image_mask_split_grad_sum_pow"
        elif "sum" in self.connected:
            self.function_name = "image_mask_split_grad_sum"
        elif "pow" in self.connected:
            self.function_name = "image_mask_split_grad_pow"
        else:
            raise ValueError("At least sum or pow gradient must be provided")

    def make_node(self, mask_idx, image, og_sum, og_pow):
        mask_idx = contiguouse(mask_idx)
        image = contiguouse(image)
        inputs = [mask_idx, image]
        if str(og_sum) == "<DisconnectedType>" and \
            str(og_pow) == "<DisconnectedType>":
            raise ValueError("At least sum or pow gradient must be provided")

        if str(og_sum) != "<DisconnectedType>":
            og_sum = contiguouse(og_sum)
            inputs.append(og_sum)
        if str(og_pow) != "<DisconnectedType>":
            og_pow = contiguouse(og_pow)
            inputs.append(og_pow)

        output_type = CudaNdarrayType(broadcastable=(False,)*4)
        return theano.Apply(self, inputs,
                            [output_type()])

    def make_thunk(self, node, storage_map, _, _2):
        inputs = [storage_map[v] for v in node.inputs]
        outputs = [storage_map[v] for v in node.outputs]
        mod = SourceModule(mask_split_kernel_code, no_extern_c=1)
        image_mask_split_grad = mod.get_function(self.function_name)

        def thunk():
            for i in inputs:
                print("{}: {}".format(i, i[0].shape))
            grad = outputs[0][0]
            mask_idx = inputs[0][0]
            batch_size = mask_idx.shape[0]
            assert shape_ok(mask_idx.shape)
            s = mask_idx.shape[3]
            sh = min(32, s)
            g = math.ceil(s / 32)
            mask_idx = to_gpuarray(mask_idx, copyif=True)

            image = inputs[1][0]
            assert shape_ok(image.shape)
            image = to_gpuarray(image, copyif=True)

            grad_shape = (batch_size, 1, s, s)
            grad = pycuda_zeros(grad, grad_shape)
            grid = (batch_size, g, g)
            block = (sh, sh, 1)
            if "sum" in self.connected and "pow" in self.connected:
                og_sum = to_gpuarray(inputs[2][0], copyif=True)
                og_pow = to_gpuarray(inputs[3][0], copyif=True)
                image_mask_split_grad(
                    mask_idx, image, og_sum, og_pow,
                    np.int32(batch_size), np.int32(s), grad,
                    block=block, grid=grid)
            elif "sum" in self.connected:
                og_sum = to_gpuarray(inputs[2][0], copyif=True)
                image_mask_split_grad(
                    mask_idx, image, og_sum,
                    np.int32(batch_size), np.int32(s), grad,
                    block=block, grid=grid)
            elif "pow" in self.connected:
                og_pow = to_gpuarray(inputs[2][0], copyif=True)
                image_mask_split_grad(
                    mask_idx, image, og_pow,
                    np.int32(batch_size), np.int32(s), grad,
                    block=block, grid=grid)
            outputs[0][0] = to_cudandarray(grad)

        return thunk


class SplitMask(theano.Op):
    __props__ = ()

    def make_node(self, mask_idx, image):
        mask_idx = contiguouse(mask_idx)
        image = contiguouse(image)
        assert mask_idx.dtype == "float32"
        assert image.dtype == "float32"
        output_type = CudaNdarrayType(broadcastable=(False,)*5)
        return theano.Apply(self, [mask_idx, image],
                            [output_type(), output_type(), output_type()])

    def make_thunk(self, node, storage_map, _, _2):
        inputs = [storage_map[v] for v in node.inputs]
        outputs = [storage_map[v] for v in node.outputs]
        mod = SourceModule(mask_split_kernel_code, no_extern_c=1)
        image_mask_split = mod.get_function("image_mask_split")
        self._sdata = None

        def thunk():
            mask_idx = inputs[0][0]
            image = inputs[1][0]
            batch_size = mask_idx.shape[0]
            assert shape_ok(mask_idx.shape)
            assert shape_ok(image.shape)
            mask_idx = to_gpuarray(mask_idx)
            image = to_gpuarray(image)
            s = mask_idx.shape[3]
            sh = s // 2
            sdata_shape = (3*len(MASK), batch_size, 1, sh, sh)
            self._sdata = pycuda_zeros(self._sdata, sdata_shape)
            grid = (batch_size, 1, 1)
            block = (sh, sh, 1)
            image_mask_split(mask_idx, image, np.int32(batch_size),
                             np.int32(s), self._sdata,
                             block=block, grid=grid)
            sdata_as_theano= to_cudandarray(self._sdata)
            m = len(MASK)
            outputs[0][0] = sdata_as_theano[:m]
            outputs[1][0] = sdata_as_theano[m:2*m]
            outputs[2][0] = sdata_as_theano[2*m:]

        return thunk

    def grad(self, inputs, output_gradients):
        grad_ins = inputs + output_gradients[:2]
        connected = []
        if str(output_gradients[0]) != "<DisconnectedType>":
            connected.append("sum")
        if str(output_gradients[1]) != "<DisconnectedType>":
            connected.append("pow")
        print(connected)
        return [T.zeros_like(inputs[0]), SplitMaskGrad(connected)(*grad_ins)]


cuda_split_mask = SplitMask()


def theano_split_mask(mask_idx, image):
    shp = image.shape
    count = T.zeros((len(MASK), shp[0]))
    splitted_list = []
    for i, (name, enum_val) in enumerate(MASK.items()):
        idx = T.eq(mask_idx, enum_val)
        count = T.set_subtensor(count[i], idx.sum(axis=(1, 2, 3)))
        tmp_image = T.zeros_like(image)
        tmp_image = T.set_subtensor(tmp_image[idx.nonzero()],
                                    image[idx.nonzero()])
        splitted_list.append(tmp_image)
    splitted = T.stack(splitted_list)
    return splitted, splitted**2, count.reshape((len(MASK), shp[0], 1, 1, 1), ndim=5)



def theano_mask_loss(mask_image, image):
    axis = [1, 2, 3]

    def get_subtensor_var(image, mean, idx):
        mean = T.patternbroadcast(mean.reshape((-1, 1, 1, 1)), [False, True, True, True])
        tmp_image = T.zeros_like(image)
        tmp_image = T.set_subtensor(tmp_image[idx.nonzero()], image[idx.nonzero()])
        return get_subtensor_mean((tmp_image - mean)**2, idx)

    def get_subtensor_sum(image, idx):
        tmp_image = T.zeros_like(image)
        tmp_image = T.set_subtensor(tmp_image[idx.nonzero()], image[idx.nonzero()])
        return T.sum(tmp_image, axis)

    def get_subtensor_mean(image, idx):
        return get_subtensor_sum(image, idx) / T.sum(idx, axis)

    white_mean = get_subtensor_mean(image, mask_image > MASK["IGNORE"])
    black_mean = get_subtensor_mean(image, mask_image < MASK["IGNORE"])
    min_distance = 0.25 * T.ones_like(black_mean)
    distance = T.minimum(white_mean - black_mean, min_distance)
    loss = (distance - min_distance)**2
    cell_loss = T.zeros_like(loss)

    def cell_loss_fn(mask_color, color_mean):
        cell_idx = T.eq(mask_image, MASK[mask_color])
        cell_mean = get_subtensor_mean(image, cell_idx)
        cell_weight = T.sum(cell_idx, axis)
        return T.switch(T.isnan(cell_mean),
                         T.zeros_like(black_mean),
                         cell_weight * (
                             (color_mean - cell_mean)**2 +
                             4*get_subtensor_var(image, color_mean, cell_idx)
                         ))
    for black_parts in MASK_BLACK:
        cell_loss += cell_loss_fn(black_parts, black_mean)
    for white_parts in MASK_WHITE:
        cell_loss += cell_loss_fn(white_parts, white_mean)

    cell_loss /= T.sum(T.neq(mask_image, MASK["IGNORE"]))
    loss += cell_loss
    return 50*T.mean(loss), 50*loss


def cuda_mask_loss(mask, image):
    raw_sum, raw_pow, raw_count = cuda_split_mask(mask, image)
    mask_sum = T.sum(raw_sum, axis=[2, 3, 4])
    mask_pow = T.sum(raw_pow, axis=[2, 3, 4])
    mask_count = T.sum(raw_count, axis=[2, 3, 4])

    mean = T.switch(T.eq(mask_count, 0),
                    T.zeros_like(mask_sum),
                    mask_sum / mask_count)
    var = T.switch(T.eq(mask_count, 0),
                   T.zeros_like(mask_sum),
                   mask_pow / mask_count - mean**2)





