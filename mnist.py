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
X_test  = X_test.reshape(10000, 784)
X_train = X_train.astype('float32') 
X_test  = X_test.astype('float32')
#X_train /= 255 # Original data is uint8 (0-255). Scale it to range [0,1].
#X_test  /= 255
print("Training X matrix shape", X_train.shape)
print("Testing X matrix shape", X_test.shape)
    
# Represent the targets as one-hot vectors: e.g. 2 -> [0, 0, 1, 0, 0, 0, 0, 0, 0].
nb_classes = 10
Y_train = np_utils.to_categorical(Y_train, nb_classes)
Y_test  = np_utils.to_categorical(Y_test, nb_classes)
print("Training Y matrix shape", Y_train.shape)
print("Testing Y matrix shape", Y_test.shape)

# Plot examples of the data.
#plt.figure(1, figsize=(14,3))
#for i in range(10):
#    plt.subplot(1,10,i+1)
#    plt.imshow(X_train[i].reshape(28,28), cmap='gray', interpolation='nearest')
#    plt.xticks([])
#    plt.yticks([])

# Simple fully-connected neural network with 2 hidden layers.
model = Sequential()
model.add(Dense(256, activation='relu', input_shape=(784,), W_regularizer=l2(0.001))) # Use input_shape=(28,28) for unflattened data.
model.add(Dense(256, activation='relu'))
#model.add(Dropout(0.2)) # Including dropout layer helps avoid overfitting.
model.add(Dense(10, activation='softmax')) # Use softmax layer for multi-class problems.

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

history = model.fit(X_train, Y_train, batch_size=128, nb_epoch=15, verbose=1,
                    validation_data=(X_test, Y_test))

# Plot loss trajectory throughout training.
#plt.figure(1, figsize=(14,5))
plt.figure(1)
#plt.subplot(1,2,1)
plt.plot(history.history['loss'], label='train')
#plt.plot(history.history['val_loss'], label='valid')
plt.xlabel('Training Epoch')
plt.ylabel('Training Error')
#plt.legend()

'''plt.subplot(1,2,2)
plt.plot(history.history['acc'], label='train')
plt.plot(history.history['val_acc'], label='valid')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()'''
plt.savefig('relu_unnorm.png')

# Note: when calling evaluate, dropout is automatically turned off.
# score = model.evaluate(X_test, Y_test, show_accuracy=True, verbose=0)
# print('Test cross-entropy loss: %0.5f' % score[0])
# print('Test accuracy: %0.2f' % score[1])

