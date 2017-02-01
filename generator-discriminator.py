import numpy as np
import numpy.random
import matplotlib.pyplot as plt

import os, sys, socket
gpuid = 1 

from collections import defaultdict
from tqdm import tqdm

import keras
from keras.datasets import mnist
from keras.models import Model, Sequential
from keras.layers import Dense, Dropout, Activation, Input, Convolution2D, Flatten, merge
from keras.layers.normalization import BatchNormalization 
from keras.utils import np_utils
from keras.optimizers import SGD, Adam

# Load data.
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()
print("Original X shape", X_train.shape)
print("Original Y shape", Y_train.shape)

# Reshape data.
X_train = X_train.reshape(60000, 784)
X_test  = X_test.reshape(10000, 784)
X_train = X_train.astype('float32') 
X_test  = X_test.astype('float32')
X_train /= 255 # Original data is uint8 (0-255). Scale it to range [0,1].
X_test  /= 255
print("Training X matrix shape", X_train.shape)
print("Testing X matrix shape", X_test.shape)
    
# Represent the targets as one-hot vectors: e.g. 2 -> [0, 0, 1, 0, 0, 0, 0, 0, 0].
nb_classes = 10
Y_train = np_utils.to_categorical(Y_train, nb_classes)
Y_test  = np_utils.to_categorical(Y_test, nb_classes)
print("Training Y matrix shape", Y_train.shape)
print("Testing Y matrix shape", Y_test.shape)

# Model
def define_generator(ndense=1, nhid=100, lr=1e-3, act='relu', mom=0.9, dropout=0.0, input_shape=20):
    # Define 'generator' that predicts sig vs bg from features.
    input_shape = input_shape # Number of inputs to generator (random numbers)
    output_shape= 784
    ndense      = 2
    nhid        = 100
    act         = 'relu'
    input       = Input(shape=(input_shape,), name='input')
    x           = input
    for i in range(ndense):
        x = Dense(output_dim=nhid, activation=act, init='glorot_normal')(x)
        x = BatchNormalization()(x)
    output    = Dense(output_dim=output_shape, activation='sigmoid', init='glorot_normal', name='output')(x)
    generator = Model(input, output)
    opt       = Adam(lr=lr, beta_1=mom) 
    generator.compile(loss='binary_crossentropy', optimizer=opt)
    return generator

def define_discriminator(ndense=1, nhid=100, act='relu', lr=1e-3, mom=0.9, decay=0.0, dropout=0.0):
    # Define 'discriminator' that predicts whether input is generated vs. real.
    input_shape = 784 
    output_shape= 1 
    ndense      = ndense
    nhid        = nhid
    act         = act
    input       = Input(shape=(input_shape,))
    x           = input
    for i in range(ndense):
        x = Dense(output_dim=nhid, activation=act, init='glorot_normal')(x)
    output = Dense(output_dim=output_shape, activation='sigmoid', init='glorot_normal')(x)
    discriminator = Model(input, output)
    #opt       = Adam(lr=lr, beta_1=mom) 
    opt = SGD(lr=lr,momentum=mom,decay=decay)
    discriminator.compile(loss='binary_crossentropy', optimizer=opt)
    return discriminator

def plot_loss(history, filename):
    plt.figure(1)
    plt.clf()
    for k in history:
        plt.plot(history[k], label=k)
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('')
    #plt.show()
    plt.savefig(filename)
    
def plot_gen(generator, input_shape, filename):
    noisebatch = np.random.uniform(-1, 1, size=[10, input_shape])
    generated  = generator.predict(noisebatch)
    plt.figure(2, figsize=(14,5))
    plt.clf()
    for i in range(10):
        plt.subplot(2,5, i+1)
        plt.imshow(generated[i,:].reshape(28,28), cmap='gray', interpolation='none', vmin=0.0, vmax=1.0)
        plt.xticks([])
        plt.yticks([]) 
    #plt.show()
    plt.savefig(filename)
    
num_epochs  = 10000  *3
plt_frq     = 500
batchsize   = 100
input_shape = 10 # Number of random inputs to generator.
lr          = .01
momgen      = 0.
momdisc     = 0.
clipnorm    = 1.
decay     =   1e-5

generator     = define_generator(ndense=4, nhid=100, act='tanh', input_shape=input_shape)
discriminator = define_discriminator(ndense=2, nhid=20, act='tanh', lr=lr*10, mom=momdisc,decay=decay) #, clipnorm=clipnorm)
discriminator.trainable = False
gan           = Sequential(layers=[generator, discriminator])
gan.compile(loss='binary_crossentropy', optimizer=SGD(lr=lr, momentum=momgen, decay=decay, clipnorm=clipnorm))
      
history = defaultdict(list)
for epoch in tqdm(range(num_epochs)):  

    # Generate images.
    noisebatch = np.random.uniform(-1, 1, size=[batchsize,input_shape])
    generated  = generator.predict(noisebatch)

    # Update discriminator.
    databatch  = X_train[np.random.randint(0,X_train.shape[0], size=batchsize), :]    
    X          = np.concatenate([databatch, generated])
    T          = np.concatenate([np.ones((databatch.shape[0],)), np.zeros((generated.shape[0],))]) # Real:1, fake:0
    discriminator.trainable = True
    d_loss  = discriminator.train_on_batch(X,T)
    history["d_loss"].append(d_loss)
    
    # Update generator.
    noisebatch = np.random.uniform(-1, 1, size=[batchsize,input_shape])
    T          = np.ones((noisebatch.shape[0],)) # Try to generate data that looks real.
    discriminator.trainable = False
    g_loss = gan.train_on_batch(noisebatch, T)
    history["g_loss"].append(g_loss)
    
    # Updates plots
    if epoch % plt_frq == plt_frq - 1:
        plot_loss(history,'loss'+str(epoch))
        plot_gen(generator, input_shape,'gen'+str(epoch))
