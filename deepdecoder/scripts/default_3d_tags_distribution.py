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

import argparse
import os
from diktya.distributions import to_radians, Uniform, Constant, TruncNormal, \
    examplary_tag_distribution, DistributionCollection


def default_tag_distribution():
    tag_dist_params = examplary_tag_distribution()
    angle = to_radians(65)
    tag_dist_params["x_rotation"] = TruncNormal(-angle, angle, 0, angle / 2)
    tag_dist_params["y_rotation"] = TruncNormal(-angle, angle, 0, angle / 2)
    tag_dist_params["radius"] = TruncNormal(22, 26, 24.0, 1.3)
    tag_dist_params["center"] = (Uniform(-16, 16), 2)

    tag_dist_params["bulge_factor"] = Uniform(0.4, 0.8)
    tag_dist_params["focal_length"] = Uniform(2, 4)
    tag_dist_params["inner_ring_radius"] = Uniform(0.42, 0.48)
    tag_dist_params["middle_ring_radius"] = Constant(0.8)
    tag_dist_params["outer_ring_radius"] = Constant(1.)
    return DistributionCollection(tag_dist_params)


def main():
    parser = argparse.ArgumentParser(
        description='Generate images and depth maps from the 3d object model of the tag')
    parser.add_argument('output', type=str, help='output file name')
    args = parser.parse_args()
    assert not os.path.exists(args.output)

    dist = default_tag_distribution()
    with open(args.output, 'w+') as f:
        f.write(dist.to_json())

if __name__ == "__main__":
    main()
