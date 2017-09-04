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

from distutils.core import setup


setup(
    name='deepdecoder',
    version='0.0.1',
    description='models for my bacheleor thesis',
    author='Leon Sixt',
    author_email='github@leon-sixt.de',
    entry_points={
        'console_scripts': [
            'bb_extract_hd_images = deepdecoder.scripts.extract_hd_images:main',
            'bb_build_real_tag_dataset = deepdecoder.scripts.build_real_tag_dataset:run',
            'bb_generate_3d_tags = deepdecoder.scripts.generate_3d_tags:main',
            'bb_default_3d_tags_distribution = ' +
                'deepdecoder.scripts.default_3d_tags_distribution:main',
            'bb_train_tag3d_network = deepdecoder.scripts.train_tag3d_network:main',
            'bb_train_rendergan = deepdecoder.scripts.train_rendergan:main',
            'bb_sample_from_rendergan = deepdecoder.scripts.sample_from_rendergan:main',
            'bb_train_decoder = deepdecoder.scripts.train_decoder:main',
            'bb_train_multiple_decoders = deepdecoder.scripts.train_multiple_decoders:main',
            'bb_evaluate_decoder = deepdecoder.scripts.evaluate_decoder:main',
            'bb_make = deepdecoder.scripts.make:main',
            'shuffle_hdf5 = deepdecoder.scripts.shuffle_hdf5:main',
        ]
    },
    scripts=[
        'scripts/bb_select_videos',
    ],
    packages=[
        'deepdecoder',
        'deepdecoder.scripts',
    ],
    package_data={
        'deepdecoder.scripts': ['Makefile']
    }
)
