
from deepdecoder.augmentation import config, stack_augmentations
from deepdecoder.data import DistributionHDF5Dataset
from diktya.numpy import tile, zip_tile, image_save
import numpy as np


def test_config(outdir):
    assert 'spotlights' in config

    config.dump(str(outdir.join('augmentation.yml')))


def test_stack_augmentations(outdir):
    fname = "/home/leon/uni/bachelor/deepdecoder/test/data/00350.hdf5"
    dset = DistributionHDF5Dataset(fname)
    batch = next(dset.iter(10**2))

    for i, name in enumerate(['tag3d', 'tag3d_lighten', 'fake_without_noise', 'fake']):
        augment = stack_augmentations(name, config)
        xs = augment(batch)
        out = np.clip(zip_tile(batch['fake'], batch[name], xs), -1, 1)
        image_save(str(outdir.join('{}_{}_augmented.png'.format(i, name))), out)
