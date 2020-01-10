from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras.layers import Dense, Conv2D
from keras.layers import Dropout, MaxPooling2D, Flatten
from keras.models import Sequential 

import numpy as np
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array

np.set_printoptions(precision=3)

data_spec = np.empty(shape=(3999,98,334,3), dtype=np.ndarray)

for index in range(2):
    im = load_img('specs/'+str(index)+'.png')
    print(type(im))
    data_img = img_to_array(im)
    print(type(data_img))
    print(data_img.shape)
    data_spec[index] = data_img

label_train = np.genfromtxt('train_labels.csv', delimiter = ',', skip_header=1)

label_train = label_train[:,1]

input_shape = (98, 334, 3)

model = Sequential()

# create model
model = Sequential()
model.add(Conv2D(filters=32, kernel_size=(3,3), strides=(1,1), input_shape=input_shape, padding='same'))
model.add(Conv2D(filters=64, kernel_size=3, strides=1, padding='same'))
model.add(Conv2D(filters=128, kernel_size=3, strides=1, padding='same'))
model.add(MaxPooling2D(pool_size=2, strides=2))
model.add(Dropout(rate=.25))
model.add(Flatten())
model.add(Dense(16, activation='relu'))
model.add(Dropout(rate=.4))
model.add(Dense(1, activation='sigmoid'))

# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.summary()

model.fit(data_spec, label_train,
          batch_size=32, epochs=15,verbose=1,
          validation_split=.2)

model.save('comp_model_spec.h5')