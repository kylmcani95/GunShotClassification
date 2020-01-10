from scipy import signal
import matplotlib.pyplot as plt
import numpy as np

data_train = np.load("train_data.npy")


wn = np.random.randn(len(data_train[0]))

for index in range(len(data_train)):
    data_train[index] = data_train[index] + .010*wn


index = 12

plt.figure(1)
plt.title("test")
plt.plot(data_train[index])
plt.show()

import sounddevice as sd

sd.play(data_train[index])


np.save('data_train_wn.npy', data_train)