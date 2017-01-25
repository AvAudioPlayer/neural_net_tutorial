import numpy as np
import matplotlib.pyplot as plt

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Input
from keras.utils import np_utils
from keras.regularizers import l2

# Load data.
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()
print("Original X shape", X_train.shape)
print("Original Y shape", Y_train.shape)

# Reshape data.
X_train = X_train.reshape(60000, 784)
X_train = X_train.astype('float32') 
X_train /= 255 # Original data is uint8 (0-255). Scale it to range [0,1].
print("Training X matrix shape", X_train.shape)
    
print("Y matrix original shape", Y_train.shape)
indices_to_be_deleted = []
for i in reversed(range(len(Y_train))):
	if Y_train[i] > 1:
		indices_to_be_deleted.append(i)
Y_train = np.delete(Y_train, indices_to_be_deleted, 0)
X_train = np.delete(X_train, indices_to_be_deleted, 0)

# Simple neural network with 1 output neuron.
model = Sequential()
model.add(Dense(1, input_shape=(784,), activation='sigmoid', init='zero')) # Use softmax layer for multi-class problems.

model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

history = model.fit(X_train, Y_train, batch_size=128, nb_epoch=15, verbose=1)

# Plot loss trajectory throughout training.
plt.figure(1)
plt.plot(history.history['loss'], label='train')
plt.xlabel('Training Epoch')
plt.ylabel('Training Error')

plt.savefig('binary.png')
