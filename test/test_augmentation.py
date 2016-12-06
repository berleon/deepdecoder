
import pytest
from deepdecoder.augmentation import config, stack_augmentations
from deepdecoder.data import DistributionHDF5Dataset
from diktya.numpy import tile, zip_tile, image_save
import numpy as np


def test_config(outdir):
    assert 'spotlights' in config

    config.dump(str(outdir.join('augmentation.yml')))


def test_stack_augmentations(outdir, datadir):
    fname = "/home/leon/uni/bachelor/deepdecoder/test/data/00350.hdf5"
    myconfig = config.load(str(datadir.join('augmentation.yml')))

    dset = DistributionHDF5Dataset(fname)
    batch = next(dset.iter(15**2))

    image_save(str(outdir.join('real.png')), tile(batch['real']))
    image_save(str(outdir.join('fake.png')), np.clip(tile(batch['fake']), -1, 1))
    # for i, name in enumerate(['tag3d']):
    for i, name in enumerate(['tag3d', 'tag3d_lighten', 'fake_without_noise', 'fake']):
        augment = stack_augmentations(name, myconfig)
        xs = augment(batch)
        image_save(str(outdir.join('{}_{}_aug_tiled.png'.format(i, name))), tile(xs))


if __name__ == "__main__":
    pytest.main(__file__)
