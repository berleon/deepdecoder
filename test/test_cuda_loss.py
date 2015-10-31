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
from deepdecoder.pydeepdecoder import GridGenerator, MaskGridArtist
import theano
import theano.tests.unittest_tools
import theano.tensor as T
from lapgan.cuda_loss import cuda_split_mask
import deepdecoder.generate_grids as gen_grids
from deepdecoder.generate_grids import MASK, MASK_BLACK, MASK_WHITE
import numpy as np
from timeit import Timer


def masks(batch_size, scales=[1.]):
    batch_size += 64 - (batch_size % 64)
    generator = GridGenerator()
    artist = MaskGridArtist()
    for masks in gen_grids.batches(batch_size, generator, artist=artist,
                                   scales=scales):
        yield masks[0].astype(np.float32)


def test_cuda_mask_split_equal():
    for s in [1., 0.5, 0.25]:
        mask_idx = next(masks(1, scales=[s]))
        image = np.random.sample((64, 1, 64*s, 64*s)).astype(np.float32)
        th_mask = T.tensor4()
        th_img = T.tensor4()
        split_fn = theano.function([th_mask, th_img],
                                   cuda_split_mask(th_mask, th_img))
        mask_sum, mask_pow, mask_count = split_fn(mask_idx, image)
        assert mask_sum.shape == (len(MASK), 64, 1, int(32*s), int(32*s))
        np_split = np.zeros((len(MASK),) + mask_idx.shape)
        for i, k in enumerate(MASK_BLACK + ["IGNORE"] + MASK_WHITE):
            idx = mask_idx == MASK[k]
            np_split[i, idx] = image[idx]

        th_sum = np.asarray(mask_sum).sum(axis=(2, 3, 4))
        np_sum = np_split.sum(axis=(2, 3, 4))
        np.testing.assert_allclose(th_sum, np_sum, rtol=1e-5)

        np.testing.assert_allclose(np.asarray(mask_pow).sum(axis=(2, 3, 4)),
                                   (np_split**2).sum(axis=(2, 3, 4)), rtol=1e-5)


def test_cuda_mask_split_verify_grad():
    mask_idx = next(masks(1, scales=[0.25]))

    def fun_sum(image):
        th_sum, _, _ = cuda_split_mask(theano.shared(mask_idx), image)
        return th_sum.mean()

    def fun_sum_pow(image):
        th_sum, th_pow, _ = cuda_split_mask(theano.shared(mask_idx), image)
        return (th_sum + th_pow).mean()

    img_val = np.ones_like(mask_idx, dtype=np.float32)
    theano.tests.unittest_tools.verify_grad(fun_sum, [img_val], n_tests=1)

    theano.tests.unittest_tools.verify_grad(fun_sum_pow, [img_val], n_tests=1)


def test_cuda_mask_split_call_grad():
    th_mask = T.tensor4()
    th_img = T.tensor4()
    th_sum, _, _ = cuda_split_mask(th_mask, th_img)
    grad_fn = theano.function([th_mask, th_img],
                              T.grad(th_sum.mean(), th_img))
    mask_idx = next(masks(1))
    image = np.random.sample((mask_idx.shape)).astype(np.float32)
    grad_fn(mask_idx, image)
    t = Timer(lambda: grad_fn(mask_idx, image))
    n = 400
    print(t.timeit(number=n) / n)

