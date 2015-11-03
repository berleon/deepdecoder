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
from functools import partial
from deepdecoder.pydeepdecoder import GridGenerator, MaskGridArtist
import theano
import theano.tests.unittest_tools
import theano.tensor as T
from lapgan.cuda_loss import cuda_split_mask, theano_split_mask, mask_loss
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
    np.set_printoptions(threshold=np.nan)
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
        for i, mask_enum in enumerate(MASK.values()):
            idx = mask_idx == mask_enum
            np_split[i, idx] = image[idx]

        th_sum = np.asarray(mask_sum).sum(axis=(2, 3, 4))
        np_sum = np_split.sum(axis=(2, 3, 4))
        np.testing.assert_allclose(th_sum, np_sum, rtol=1e-5)
        np.testing.assert_allclose(np.asarray(mask_pow).sum(axis=(2, 3, 4)),
                                   (np_split**2).sum(axis=(2, 3, 4)), rtol=1e-5)


def test_cuda_mask_split_verify_grad():
    mask_idx = next(masks(1, scales=[0.25]))
    axis = [2, 3, 4]

    def fun_sum(fn, mask, image):
        th_sum, _, count = fn(mask, image)
        return (th_sum.sum(axis) / count.sum(axis)).sum()

    def fun_sum_pow(fn, mask, image):
        th_sum, th_pow, count = fn(mask, image)
        return ((th_sum + th_pow).sum(axis) / count.sum(axis)).sum()

    def grad_fn(loss):
        image = T.tensor4()
        mask = T.tensor4()
        return theano.function([mask, image], T.grad(loss(mask, image), image))

    cuda_grad_fn = grad_fn(partial(fun_sum, cuda_split_mask))
    theano_grad_fn = grad_fn(partial(fun_sum, theano_split_mask))

    img_val = np.random.random(mask_idx.shape).astype(np.float32)
    cuda_grad = cuda_grad_fn(mask_idx, img_val)
    theano_grad = theano_grad_fn(mask_idx, img_val)
    np.testing.assert_allclose(cuda_grad, theano_grad, rtol=1e-5)

    cuda_grad_fn = grad_fn(partial(fun_sum_pow, cuda_split_mask))
    theano_grad_fn = grad_fn(partial(fun_sum_pow, theano_split_mask))
    cuda_grad = cuda_grad_fn(mask_idx, img_val)
    theano_grad = theano_grad_fn(mask_idx, img_val)
    np.testing.assert_allclose(cuda_grad, theano_grad, rtol=1e-5)


def test_cuda_theano_mask_split_sum_equal():
    mask_idx = theano.shared(next(masks(1, scales=[0.5])))
    image = theano.shared(np.random.sample((64, 1, 32, 32)).astype(np.float32))
    axis = [2, 3, 4]
    cuda_split = [o.sum(axis) for o in cuda_split_mask(mask_idx, image)]
    theano_split = [o.sum(axis)for o in theano_split_mask(mask_idx, image)]

    for c, t in zip(cuda_split, theano_split):
        np.testing.assert_allclose(c.eval(), t.eval(), rtol=1e-5)


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


def test_theano_split_mask():
    th_mask = T.tensor4()
    th_img = T.tensor4()
    th_sum, _, _ = theano_split_mask(th_mask, th_img)
    sum_fn = theano.function([th_mask, th_img], th_sum)
    mask_idx = next(masks(1))
    image = np.random.sample((mask_idx.shape)).astype(np.float32)
    sum_fn(mask_idx, image)
    t = Timer(lambda: sum_fn(mask_idx, image))
    n = 10
    print(t.timeit(number=n) / n)


def test_mask_loss():

    th_mask, th_img = T.tensor4(), T.tensor4()
    cuda_mask_loss = theano.function([th_mask, th_img],
                                     mask_loss(th_mask, th_img, impl='cuda'))

    theano_mask_loss = theano.function([th_mask, th_img],
                                       mask_loss(th_mask, th_img, impl='theano'))
    mask_idx = next(masks(1))
    image_ok = np.zeros_like(mask_idx)
    image_ok[mask_idx > MASK["IGNORE"]] = 1

    assert (cuda_mask_loss(mask_idx, image_ok)[1] == 0).all()
    assert (theano_mask_loss(mask_idx, image_ok)[1] == 0).all()

    t = Timer(lambda: cuda_mask_loss(mask_idx, image_ok))
    n = 10
    print("cuda implementation: {}".format(t.timeit(number=n) / n))

    t = Timer(lambda: theano_mask_loss(mask_idx, image_ok))
    print("theano implementation: {}".format(t.timeit(number=n) / n))

