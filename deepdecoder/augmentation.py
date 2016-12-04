
from skimage.transform import rescale
from skimage.filters import gaussian
import numpy as np
import cfg


def get_gauss(mu, size, cov):
    w = size
    lin = np.linspace(0, w, num=w)
    x, y = np.meshgrid(lin, lin)
    xy_mu = np.stack([x - mu[0], y - mu[1]], axis=-1)
    cov_i = np.linalg.inv(cov)
    return np.exp(-np.sum(xy_mu * np.dot(xy_mu, cov_i), axis=-1))


def random_gauss(mu, size, cov_scale=1):
    A = np.random.uniform(0, 1, (2, 2))
    cov = 0.5*(A.T + A)
    cov += 2*np.eye(2)
    return get_gauss(mu, size, cov_scale*cov)


def get_augmentations(name):
    augmentations = []
    previous = False
    if name == 'tag3d':
        augmentations.append(BlurAugmentation)
        augmentations.append(LightingAugmentation)
        previous = True

    if name == 'tag3d_lighten' or previous:
        augmentations.append(BackgroundAugmentation)
        previous = True

    if name == 'fake_without_noise' or previous:
        augmentations.append(SpotlightAugmentation)
        augmentations.append(NoiseAugmentation)

    return augmentations


def needed_datanames(name):
    needed = set(name)
    previous = False
    if name == 'tag3d':
        previous = True

    if name == 'tag3d_lighten' or previous:
        needed.add('tag3d_segmented')
        previous = True

    if name == 'fake_without_noise' or previous:
        needed.add('tag3d_segmented')
        needed.add('tag3d_depth_map')
    return list(needed)


def stack_augmentations(name, config):
    augmentation_classes = get_augmentations(name)
    batch_selections = {
        BackgroundAugmentation: lambda b: (b['tag3d_segmented'], ),
        SpotlightAugmentation: lambda b: (b['tag3d_segmented'], b['tag3d_depth_map']),
    }
    augs = []
    for cls in augmentation_classes:
        augs.append(cls(config))

    def wrapper(batch):
        xs = np.clip(batch[name], -1, 1)
        for aug in augs:
            if type(aug) in batch_selections:
                additional_args = batch_selections[type(aug)](batch)
            else:
                additional_args = []
            xs_aug = aug(xs, *additional_args)
            assert xs_aug.shape == xs.shape
            xs = xs_aug
        return np.clip(xs, -1, 1)

    return wrapper


config = cfg.Config()


@config('spotlights')
class SpotlightAugmentation():
    def __init__(self, nb_spots_prob=[0.7, 0.1, 0.1, 0.07, 0.03],
                 spot_cov_scale_low=1,
                 spot_cov_scale_high=2,
                 intensity_scale_low=0.5,
                 intensity_scale_high=3,
                 ):
        self.nb_spots_prob = nb_spots_prob
        self.spot_cov_scale = lambda: np.random.uniform(spot_cov_scale_low, spot_cov_scale_high)
        self.intensity_scale = lambda: np.random.uniform(intensity_scale_low, intensity_scale_high)

    def __call__(self, xs, tag3d_segmented, tag3d_depth_map):
        nb_spotlights = np.random.choice(len(self.nb_spots_prob), size=len(xs),
                                         p=self.nb_spots_prob)
        x_spots = np.copy(xs)
        for i in range(len(xs)):
            istag = (1 - tag3d_segmented[i, 0])
            depth_map = rescale(tag3d_depth_map[i, 0], 4)
            depth_map = depth_map * istag
            spots_pos = []
            while len(spots_pos) < int(nb_spotlights[i]):
                x, y = np.random.uniform(0, xs.shape[-1], 2).astype(np.int)
                if istag[x, y] < 0.8:
                    continue
                if depth_map[x, y] > np.random.uniform():
                    spots_pos.append((y, x))

            for pos in spots_pos:
                spotlight = random_gauss(pos, xs.shape[-1], cov_scale=np.random.uniform(1, 4))
                x_spots[i] += np.random.uniform(0.5, 3)*spotlight
        return x_spots


def random_backgrond(weights=[2, 8, 4, 3, 0.2, 0.2, 0.5], start_level=1, end_level=None):
    img = np.random.normal(0, 1, (start_level, start_level)) * weights[0]
    for i, w in enumerate(weights[1:], start_level):
        r = np.random.normal(0, 1, (2**i, 2**i))
        img = rescale(img, 2) + w*r

    if end_level and end_level != i:
        img = rescale(img, 2**(end_level - i))

    img = (img - img.min())
    img /= img.max()
    return img


@config('background')
class BackgroundAugmentation:
    def __init__(self, pyramid_weights=[2, 4, 4, 3],
                 intensity_scale_low=0.7,
                 intensity_scale_high=1.2,
                 intensity_shift_mean=-0.5,
                 intensity_shift_std=0.2,
                 segmentation_blur_low=1.5,
                 segmentation_blur_high=3):

        self.pyramid_weights = pyramid_weights
        self.intensity_scale = lambda: np.random.uniform(intensity_scale_low, intensity_scale_high)
        self.intensity_shift = lambda: np.random.normal(intensity_shift_mean, intensity_shift_std)
        self.segmentation_blur = lambda: np.random.uniform(segmentation_blur_low, segmentation_blur_high)

    def __call__(self, xs, tag3d_segmented):
        xs_bg = np.copy(xs)
        for i in range(len(xs)):
            x = xs[i, 0]
            istag = 1 - tag3d_segmented[i, 0]
            istag = gaussian(istag, self.segmentation_blur())
            bg = random_backgrond(self.pyramid_weights, end_level=6)
            bg *= self.intensity_scale()
            bg += self.intensity_shift()
            xs_bg[i] = x*istag + bg*(1-istag)
        return np.clip(xs_bg, -1, 1)


@config('noise')
class NoiseAugmentation:
    def __init__(self, noise_low=0.01, noise_high=0.08):
        self.noise_low = noise_low
        self.noise_high = noise_high

    def __call__(self, xs):
        xs_bg = np.copy(xs)
        for i in range(len(xs)):
            x = xs_bg[i, 0]
            h, w = x.shape
            noise = np.random.uniform(self.noise_low, self.noise_high)
            noise = np.clip(noise, 1e-5, np.inf)
            x += rescale(np.random.normal(0, noise, (h//2, w//2)), 2)
            x += np.random.normal(0, noise, x.shape)

        return np.clip(xs_bg, -1, 1)


@config('lighting')
class LightingAugmentation:
    def __init__(self, scale=0.85, weights=[8, 4, 2, 0, 0, 0, 0]):
        self.weights = weights
        self.scale = scale

    def __call__(self, xs):
        xs_ligh = np.copy(xs)
        for i in range(len(xs)):
            x_ligh = xs_ligh[i, 0]
            x = xs[i, 0]
            b = random_backgrond(self.weights, end_level=6)
            w = random_backgrond(self.weights, end_level=6)
            t = random_backgrond(self.weights, end_level=6)
            s = self.scale
            x_ligh[x < 0] *= s*b[x < 0] + (1 - s)
            x_ligh[x > 0] *= s*w[x > 0] + (1 - s)
            x_ligh += 2*s*(t - 0.5)
        return np.clip(xs_ligh, -1, 1)


@config('blur')
class BlurAugmentation:
    def __init__(self, low=1, high=3):
        self.low = low
        self.high = high

    def __call__(self, xs):
        xs_gaus = np.copy(xs)
        for i in range(len(xs)):
            xs_gaus[i] = gaussian(xs[i], np.random.uniform(self.low, self.high))
        return xs_gaus
