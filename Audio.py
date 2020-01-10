from scipy import signal
import matplotlib.pyplot as plt
import numpy as np
import time

data_train = np.load("train_data.npy")


data_roll = np.roll(data_train[12], 24000)

for index in range(len(data_train)):
    data_train[index] = np.roll(data_train[index], 24000)


index = 12

plt.figure(1)
plt.title("test")
plt.plot(data_train[index])
plt.show()

plt.figure(2)
plt.title("test")
plt.plot(data_roll)
plt.show()

import sounddevice as sd

sd.play(data_train[index])
sd.play(data_roll)

np.save('train_data_delay.npy', data_train)