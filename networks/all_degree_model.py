#! /usr/bin/env python3

import os
import os.path
import sys

from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.models import Sequential

from deepdecoder import NUM_CELLS, GeneratedGridTrainer

PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))


def get_model():
    model = Sequential()

    model.add(Convolution2D(16, 1, 3, 3, border_mode='full'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(poolsize=(2, 2)))
    model.add(Convolution2D(32, 16, 3, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(poolsize=(2, 2)))
    model.add(Dropout(0.4))

    model.add(Flatten())
    model.add(Dense(6272, 2048))
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

model = get_model()
model.load_weights("weights/0650_all_degrees_model.hdf5")

trainer = GeneratedGridTrainer.GeneratedGridTrainer()
trainer.set_gt_path_file("/home/leon/uni/bachelor/beesgrid_models/gt_files.txt")
trainer.fit_data_gen()
trainer.real_test(model, n_epochs=12)

