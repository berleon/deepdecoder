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

import itertools
from functools import partial
from timeit import Timer

import matplotlib.pyplot as plt
import numpy as np
import theano
import theano.tensor as T
import theano.tests.unittest_tools
from beesgrid import MASK, generate_grids
from beesgrid import GridGenerator, MaskGridArtist
from keras.layers.core import Dense, Reshape
from keras.models import Sequential
from keras.optimizers import Adam


from deepdecoder.mask_loss import cuda_split_mask, theano_split_mask, \
    mask_loss, median, mask_loss_sobel


def masks(batch_size, scales=[1.]):
    batch_size += 64 - (batch_size % 64)
    generator = GridGenerator()
    artist = MaskGridArtist()
    for masks in generate_grids(batch_size, generator, artist=artist,
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
        assert mask_sum.shape == (len(MASK), 64, 1, int(64*s), int(64*s))
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
    for s in [1., 0.25]:
        mask_idx = next(masks(1, scales=[s]))
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
    mask_idx = theano.shared(next(masks(1, scales=[1.0])))
    image = theano.shared(np.ones((64, 1, 64, 64)).astype(np.float32))
    axis = [2, 3, 4]
    cuda_split = [o.sum(axis) for o in cuda_split_mask(mask_idx, image)]
    theano_split = [o.sum(axis) for o in theano_split_mask(mask_idx, image)]

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
                                     mask_loss(th_mask, th_img,
                                               impl='cuda')['loss'])

    theano_mask_loss = theano.function([th_mask, th_img],
                                       mask_loss(th_mask, th_img,
                                                 impl='theano')['loss'])
    mask_idx = next(masks(1))
    image_ok = np.zeros_like(mask_idx)
    image_ok[mask_idx > MASK["IGNORE"]] = 1

    assert (cuda_mask_loss(mask_idx, image_ok) == 0).all()
    assert (theano_mask_loss(mask_idx, image_ok) == 0).all()

    t = Timer(lambda: cuda_mask_loss(mask_idx, image_ok))
    n = 10
    print("cuda implementation: {}".format(t.timeit(number=n) / n))

    t = Timer(lambda: theano_mask_loss(mask_idx, image_ok))
    print("theano implementation: {}".format(t.timeit(number=n) / n))


def test_mask_loss_sobel():
    th_mask, th_img = T.tensor4(), T.tensor4()
    ml = mask_loss_sobel(th_mask, th_img)
    mask_loss = theano.function([th_mask, th_img],
                                [ml.loss] + list(ml.sobel_mask) +
                                list(ml.sobel_img))

    mask_idx = next(masks(1))
    image_ok = 0.5 * np.ones_like(mask_idx)
    image_ok[mask_idx > MASK["IGNORE"]] = 1
    image_ok[mask_idx < MASK["BACKGROUND_RING"]] = 0

    print()
    loss, sobel_mask_x, sobel_mask_y, sobel_img_x, sobel_img_y = \
        mask_loss(mask_idx, image_ok)
    plt.set_cmap('gray')
    plt.subplot(221)
    plt.imshow(sobel_mask_x[0, 0])
    plt.subplot(222)
    plt.imshow(sobel_mask_y[0, 0])
    plt.colorbar()
    plt.subplot(223)
    plt.imshow(sobel_img_x[0, 0])
    plt.subplot(224)
    plt.imshow(sobel_img_y[0, 0])
    plt.colorbar()
    plt.savefig("mask_loss_sobel.png")
    print()
    print("mask_loss: {}".format(mask_loss(mask_idx, image_ok)))
    assert loss == 0


def test_mask_loss_network():
    model = Sequential()
    model.add(Dense(16*16, input_dim=16*16))
    model.add(Reshape((1, 16, 16)))
    net_out = model.get_output()

    net_in = model.get_input()
    th_mask = T.tensor4()
    loss = mask_loss(th_mask, net_out)['loss']
    updates = Adam().get_updates(model.params, model.constraints, loss)
    train_fn = theano.function([th_mask, net_in], [loss], updates=updates)

    nb_batches = 32
    mask_idx = next(masks(64*nb_batches, scales=[0.25]))
    z = np.random.uniform(low=-1, high=1, size=mask_idx.shape).reshape((-1, 16*16)).astype(np.float32)
    first_loss = 0

    epochs = 30
    nb_batches = 10
    for i, mask_idx in enumerate(itertools.islice(masks(64*nb_batches, scales=[0.25]), epochs)):
        z = np.random.uniform(low=-1, high=1, size=mask_idx.shape
                              ).reshape((-1, 16*16)).astype(np.float32)
        loss = train_fn(mask_idx, z)
        # print(loss)
        if i == 0:
            first_loss = loss

    assert first_loss > loss


def test_mask_median():
    img_shape = (29, 64, 1, 64, 64)
    pixels = img_shape[-1]*img_shape[-2]
    counts_shape = (29, 64)

    def np_median(image_split, counts):
        reshaped = np.reshape(image_split, image_split.shape[:3] + (-1,))
        img_sort = np.sort(reshaped)
        idxs = (pixels + counts) // 2
        print(idxs)
        return img_sort.take(idxs)

    img = np.random.sample(img_shape).astype(np.float32)
    counts = (0.8*np.random.sample(counts_shape) * pixels).astype(np.int32)

    np_result = np_median(img, counts)
    theano_result = median(theano.shared(img), theano.shared(counts)).eval()
    assert np_result.shape == (29, 64)
    assert theano_result.shape == (29, 64)
    np.testing.assert_allclose(np_result, theano_result, rtol=1e-5)


def test_mask_loss_median():
    th_mask, th_img = T.tensor4(), T.tensor4()

    cuda_out = mask_loss_median(th_mask, th_img, impl='cuda')
    cuda_mask_loss = theano.function([th_mask, th_img],
                                     [cuda_out['loss'],
                                      cuda_out['median_black'],
                                      cuda_out['loss_per_sample'],
                                      cuda_out['black_white_loss']])

    theano_mask_loss = theano.function([th_mask, th_img],
                                       mask_loss_median(th_mask, th_img,
                                                 impl='theano')['loss'])
    mask_idx = next(masks(1))
    image_ok = np.zeros_like(mask_idx)
    image_ok[mask_idx > MASK["IGNORE"]] = 1

    outs = cuda_mask_loss(mask_idx, image_ok)
    for s in outs[1:]:
        print(s.shape)
    assert (cuda_mask_loss(mask_idx, image_ok)[0] == 0).all()
    assert (theano_mask_loss(mask_idx, image_ok) == 0).all()

    t = Timer(lambda: cuda_mask_loss(mask_idx, image_ok))
    n = 10
    print("cuda implementation: {}".format(t.timeit(number=n) / n))

    t = Timer(lambda: theano_mask_loss(mask_idx, image_ok))
    print("theano implementation: {}".format(t.timeit(number=n) / n))
