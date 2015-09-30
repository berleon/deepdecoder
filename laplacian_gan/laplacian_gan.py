#! /usr/bin/env python3
from deepdecoder.generate_grids import BlackWhiteArtist

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
import os
import os.path

from deepdecoder import NUM_CELLS
import sys
from scipy.misc import imsave
import numpy as np
PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

from beras.layers.attention import RotationTransformer
import GeneratedGridTrainer
from GeneratedGridTrainer import NetworkArgparser


def get_model():
    rot_model = Sequential()
    rot_model.add(Convolution2D(12, 1, 5, 5, activation='relu'))
    rot_model.add(MaxPooling2D((3, 3)))
    rot_model.add(Dropout(0.5))
    rot_model.add(Flatten())
    rot_model.add(Dense(3888, 256))
    rot_model.add(Dropout(0.5))
    rot_model.add(Dense(256, 1, activation='relu'))

    model = Sequential()
    model.add(RotationTransformer(rot_model))
    model.add(Convolution2D(16, 1, 3, 3, border_mode='full'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(poolsize=(2, 2)))
    model.add(Dropout(0.5))
    model.add(Convolution2D(32, 16, 2, 2))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(poolsize=(2, 2)))
    model.add(Dropout(0.5))
    model.add(Convolution2D(48, 32, 2, 2))
    model.add(MaxPooling2D(poolsize=(2, 2)))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Convolution2D(48, 48, 2, 2, border_mode='same'))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(2352, 2048))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(2048, 512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(512, 128))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(128, NUM_CELLS))
    model.add(Activation('sigmoid'))
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model


def train(args):
    model = get_model()
    trainer = GeneratedGridTrainer.GeneratedGridTrainer()
    trainer.artist = BlackWhiteArtist()
    trainer.fit_data_gen()
    trainer.save_iter = 30
    trainer.train(model, "weights_new")


def test(args):
    trainer = GeneratedGridTrainer.GeneratedGridTrainer()
    trainer.fit_data_gen()
    trainer.set_gt_path_file("/home/leon/uni/vision_swp/deeplocalizer_data/gt_preprocessed/gt_files.txt")
    real_images, _ = next(trainer.real_batches())
    fake_images, _ = next(trainer.batches())
    os.makedirs("real_images", exist_ok=True)
    for i in range(len(real_images)):
        imsave("real_images/{}.jpeg".format(i), real_images[i, 0])
    model = get_model()
    model.load_weights("weights_first/0990_all_degrees_model.hdf5")
    trainer.minibatches_per_epoche = 1
    trainer.minibatch_size = 8
    trainer.real_test(model, n_epochs=1000)
    trainer.test(model, n_epochs=1)


argparse = NetworkArgparser(train, test)
argparse.parse_args()
