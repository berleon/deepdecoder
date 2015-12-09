#! /usr/bin/env python3

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import os
import os.path

import deepdecoder.generate_grids as gen_grids
from deepdecoder import NUM_CELLS
import matplotlib.pyplot as plt
import math


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

datagen = ImageDataGenerator(featurewise_center=True, featurewise_std_normalization=False)
grids, _ = next(gen_grids.batches(batchsize=2048, generator=gen))
grids = grids.astype(np.float32) / 255
datagen.fit(grids)

n_minibatch = 128
n_batch = n_minibatch * 12
weight_dir = "weights/all_degrees"
os.makedirs(weight_dir)

def train(model, weight_dir):
    for i, (raw_grids, raw_labels) in enumerate(gen_grids.batches(batchsize=n_batch, generator=gen)):
        print(i)
        if i % 10 == 0:
            file = os.path.abspath("{}/{:0>4}_all_degrees_model.hdf5".format(weight_dir, i))
            print("saving weights to: " + file)
            model.save_weights(file)
        grids, labels = next(datagen.flow(raw_grids.astype(np.float32)/255, raw_labels, batch_size=n_batch))
        model.fit(grids, labels, nb_epoch=1, batch_size=n_minibatch, verbose=1)


def test(model):
    for i, (raw_grids, raw_labels) in enumerate(gen_grids.batches(batchsize=n_minibatch, generator=gen)):
        grids, labels = next(datagen.flow(raw_grids.astype(np.float32)/255, raw_labels, batch_size=n_minibatch))
        prediction = model.predict(grids, batch_size=n_minibatch, verbose=0)
        bit_prediction = prediction > 0.5
        n_right = 0.
        n_total = labels.shape[0]
        for i in range(n_total):
            if (bit_prediction[i] == labels[i]).all():
                n_right += 1

        print(n_right / n_total)


model = get_model()
train(model, weight_dir)
