from keras.models import load_model

import numpy as np


np.set_printoptions(edgeitems=100)
#3997 entries
data_test = np.load("test_data.npy")
data_test = np.reshape(data_test, [-1, 44100, 1])
#data_test = data_test.astype('float32')

count = data_test.shape[0]
data_submit = (count, 2)
data_submit = np.zeros(data_submit)

train_model = load_model('comp_model_kk50.h5')

predictions = train_model.predict(data_test, batch_size=128, verbose=1)


for index in range(count):
    data_submit[index][0] = index
    data_submit[index][1] = predictions[index]
    
print(data_submit)

np.savetxt('comp_kk.csv', data_submit, delimiter=',')