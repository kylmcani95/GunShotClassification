from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras.layers import Dense, Conv1D
from keras.layers import Dropout, MaxPooling1D, Flatten
from keras.models import Sequential 

import numpy as np

np.set_printoptions(precision=3)

data_train = np.load("train_data.npy")
data_train_wn = np.load("train_data_wn.npy")
data_train_delay = np.load("train_data_delay.npy")
print(data_train.shape)
print(data_train_wn.shape)

data_train = np.concatenate((data_train, data_train_wn, data_train_delay))


label_train = np.genfromtxt('train_labels.csv', delimiter = ',', skip_header=1)
label_train = np.concatenate((label_train, label_train, label_train))

print(data_train.shape)

data_train = np.reshape(data_train, [-1, 44100, 1])


label_train = label_train[:,1]

print(data_train.shape)
print(label_train.shape)

input_shape = (44100, 1)

model = Sequential()

# create model
model = Sequential()
model.add(Conv1D(filters=32, kernel_size=3, strides=1, input_shape=input_shape, padding='same'))
model.add(Conv1D(filters=64, kernel_size=3, strides=1, padding='same'))
model.add(Conv1D(filters=128, kernel_size=3, strides=1, padding='same'))
model.add(MaxPooling1D(pool_size=2, strides=2))
model.add(Dropout(rate=.25))
model.add(Flatten())
model.add(Dense(16, activation='relu'))
model.add(Dropout(rate=.4))
model.add(Dense(1, activation='sigmoid'))

# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.summary()

model.fit(data_train, label_train,
          batch_size=32, epochs=15,verbose=1,
          validation_split=.2)

model.save('comp_model_50Ep.h5')