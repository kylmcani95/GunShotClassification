from keras.layers import Dense, Input
from keras.layers import Conv1D, Flatten, Lambda
from keras.layers import Reshape, Dropout, MaxPooling1D
from keras.models import Model
import matplotlib.pyplot as plt

from keras.losses import  binary_crossentropy
#from keras.utils import plot_model
from keras import backend as K

import numpy as np
import os

data_train = np.load("train_data.npy")
data_test = np.load("test_data.npy")
label_train = np.genfromtxt('train_labels.csv', delimiter=',', skip_header=1)


print(data_train.shape)
print(data_test.shape)

data_train = np.reshape(data_train, [-1, 44100, 1])
data_test = np.reshape(data_test, [-1, 44100, 1])
data_train = data_train.astype('float32') / 255
data_test = data_test.astype('float32') / 255
label_train= label_train[:,1]

print(data_test.shape)
# network parameters
input_shape = (44100, 1)
latent_dim = 30
epochs = 5
kernel_size = 1
filters = 16


#sampling for mean and log variance from Keras
def sampling(args):
    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon

def plot_results(models,
                 data,
                 batch_size=128,
                 model_name="vae_mnist"):
    """Plots labels and MNIST digits as a function of the 2D latent vector

    # Arguments
        models (tuple): encoder and decoder models
        data (tuple): test data and label
        batch_size (int): prediction batch size
        model_name (string): which model is using this function
    """

    encoder, decoder = models
    x_test, y_test = data
    os.makedirs(model_name, exist_ok=True)

    filename = os.path.join(model_name, "vae_mean.png")
    # display a 2D plot of the digit classes in the latent space
    z_mean, _, _ = encoder.predict(x_test,
                                   batch_size=batch_size)
    plt.figure(figsize=(12, 10))
    plt.scatter(z_mean[:, 0], z_mean[:, 1], c=y_test)
    plt.colorbar()
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.savefig(filename)
    plt.show()

    

#dense layers seem to be the only thing that works
#using Conv1D/Conv2D wouldnt start training
inputs = Input(shape=(input_shape))
shape = K.int_shape(inputs)
print("TEST TEST TEST")
print(shape)
print(shape)
input_layers = inputs
input_layers = Conv1D(filters=filters, kernel_size=kernel_size)(input_layers)
input_layers = Flatten()(input_layers)
input_layers = Dense(16, activation='relu')(input_layers)
z_mean = Dense(latent_dim, name='z_mean')(input_layers)
z_log_var = Dense(latent_dim, name='z_log_var')(input_layers)

z = Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])

# instantiate encoder model
encoded = Model(inputs, [z_mean, z_log_var, z], name='encoder')
encoded.summary()

#decoded
latent_inputs = Input(shape=(latent_dim,))
output_layers = Dense(shape[1] * shape[2], activation='relu')(latent_inputs)
output_layers = Reshape((shape[1], shape[2]))(output_layers)
output_layers = Conv1D(filters=filters*2, kernel_size=kernel_size)(output_layers)
output_final = Conv1D(filters=1, kernel_size=kernel_size, activation='sigmoid')(output_layers)
decoded = Model(latent_inputs, output_final)
decoded.summary()



# instantiate VAE model
outputs = decoded(encoded(inputs)[2])
vae = Model(inputs, outputs, name='vae')

kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
kl_loss = K.sum(kl_loss, axis=-1)
kl_loss *= -0.5
vae_loss = K.mean(binary_crossentropy(K.flatten(inputs), K.flatten(outputs)) + kl_loss)
vae.add_loss(vae_loss)
vae.compile(optimizer='rmsprop')
vae.summary()

vae.fit(data_train,
                epochs=epochs,
                batch_size=100,
                validation_data(data_test, None))
vae.save_weights('vae_mlp_mnist.h5')
vae.save('vae_fixed.h5')

models = (encoded, decoded)
data = (data_train, label_train)
plot_results(models,
                 data,
                 batch_size=100,
                 model_name="vae_mlp")
