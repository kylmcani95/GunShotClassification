
import matplotlib.pyplot as plt
import numpy as np


data_train = np.load("train_data.npy")
fs = 10e3


x = data_train[11]




for index in range(3997):
    plt.subplot(211)
    plt.specgram(x,Fs=fs)
    plt.axis('off')
    plt.savefig('specs/'+str(index)+'.png', bbox_inches = 'tight', transparent=True, pad_inches=0.0 )
    

#np.save('data_train_spec.npy', data_spec)