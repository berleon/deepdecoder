#! /usr/bin/env python

from pipeline import Pipeline
from pipeline.pipeline import get_auto_config
from pipeline.objects import Filename, LocalizerPositions, Image
from pipeline.stages.processing import Localizer
from bb_binary import parse_image_fname
from deepdecoder.data import HDF5Dataset
import numpy as np
import click
import progressbar
import os


@click.command()
@click.option('--out', type=click.Path())
@click.option('--force', is_flag=True)
@click.argument('image_path_file', type=click.File('r'))
def run(image_path_file, out, force):
    cache = {}

    if force and os.path.exists(out):
        print("Removing {}.".format(out))
        os.remove(out)

    def add_to_cache(**kwargs):
        for name, arr in kwargs.items():
            if name not in cache:
                cache[name] = [arr]
            else:
                cache[name].append(arr)

    def flush_cache():
        cache_concat = {n: np.concatenate(arrs) for n, arrs in cache.items()}
        nb_samples = len(next(iter(cache_concat.values())))
        permutation = np.random.permutation(nb_samples)
        cache_shuffled = {n: arrs[permutation]
                          for n, arrs in cache_concat.items()}
        dset.append(**cache_shuffled)
        cache.clear()

    roi_size = 96
    image_fnames = [n.rstrip('\n') for n in image_path_file.readlines()]
    pipeline = Pipeline([Filename],
                        [Image, LocalizerPositions],
                        **get_auto_config())
    dset = HDF5Dataset(out)
    bar = progressbar.ProgressBar(max_value=len(image_fnames))
    for i, image_fname in enumerate(bar(image_fnames)):
        try:
            results = pipeline([image_fname])
            rois, mask = Localizer.extract_rois(results[LocalizerPositions],
                                                results[Image], roi_size)
        except Exception as e:
            print(e)
            continue

        nb_detections = np.sum(mask)
        camIdx, dt = parse_image_fname(image_fname)
        season = np.array([dt.year] * nb_detections, dtype=np.uint16)
        timestamp = np.array([dt.timestamp()] * nb_detections, dtype=np.float64)
        add_to_cache(rois=rois, season=season, timestamp=timestamp)
        if i % 50 == 0 and i != 0:
            flush_cache()
    flush_cache()

if __name__ == "__main__":
    run()
