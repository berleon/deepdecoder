#! /usr/bin/env python3
# Copyright 2016 Leon Sixt
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


import matplotlib
matplotlib.use('Agg')  # noqa

import os
import argparse
from pipeline.io import raw_frames_generator
from bb_binary import parse_video_fname, get_fname
from diktya.numpy import image_save
import numpy as np
import joblib


def save_first_frame(video_fname, output_dir, force):
    camIdx, start, _ = parse_video_fname(video_fname)
    outname = os.path.join(output_dir, get_fname(camIdx, start) + ".png")
    if os.path.exists(outname) and not force:
        return
    gen = raw_frames_generator(video_fname, format='guess_on_ext')
    frame = next(gen)
    assert frame.dtype == np.uint8
    image_save(outname, frame)


def run(pathfile, n_jobs, force, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    with open(pathfile) as f:
        fnames = [l.rstrip('\n') for l in f.readlines()]

    joblib.Parallel(n_jobs, verbose=5)(
        joblib.delayed(save_first_frame)(fname, output_dir, force) for fname in fnames)


def main():
    parser = argparse.ArgumentParser(
        description='Extracts the first frame of many different videos')
    parser.add_argument('-n', '--n-jobs', type=int, default=2, help='number of parallel worker')
    parser.add_argument('-o', '--output-dir', type=str, help='output dir name')
    parser.add_argument('-f', '--force', action='store_true',
                        help='override existing images')
    parser.add_argument('pathfile', type=str,
                        help='File with a list of files.')
    args = parser.parse_args()
    run(args.pathfile, args.n_jobs, args.force, args.output_dir)

if __name__ == "__main__":
    main()
