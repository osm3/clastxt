# -*- coding: utf-8 -*-
"""
Created on Tue Jul 18 10:27:36 2017

@author: osm
"""
from functools import reduce
import numpy
import gc
from keras.models import Sequential
from keras.layers import Dense, Flatten, Activation
from keras.layers import Dropout
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.utils import np_utils
from keras.optimizers import SGD, Adam
from keras.utils import plot_model
import pandas as pd
import csv
import json
import gc
import re
from keras import initializers
model = Sequential()
# Первый сверточный слой
model.add(Conv2D(10, (2, 2), padding='same',
                        input_shape=(10, 10), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(2, activation='softmax'))
adam = Adam()
model.compile(loss='categorical_crossentropy',
              optimizer=adam,
              metrics=['accuracy'])
block_size = 2
def get_data():
    while True:
        x=numpy.zeros((block_size,10,10,1))
        y=numpy.zeros((block_size,1))
        print("before yield\n")
        yield x,np_utils.to_categorical(y, 3)
        print("after yield\n")
model.fit_generator(get_data(), steps_per_epoch = 1, verbose = 1, epochs = 3, max_queue_size = 1)
