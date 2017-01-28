import numpy as np
import numpy.random
import matplotlib.pyplot as plt

import os, sys, socket
gpuid = 0 
#os.environ['THEANO_FLAGS'] = "mode=FAST_RUN,device=gpu%d,floatX=float32,force_device=True,base_compiledir=~/.theano/%s_gpu%d" % (gpuid, socket.gethostname(), gpuid)

import theano
import keras
from keras.datasets import mnist
from keras.models import Model
from keras.layers import Dense, Dropout, Activation, Input, Convolution2D, Flatten, merge
from keras.utils import np_utils
from keras.optimizers import SGD, Adam

# The following function is used to serve up both train and validation data.
def data_generator(X, Y, batchsize):
    nb_classes = 10
    N = X.shape[0]
    while True:
        indices1 = np.random.randint(low=0, high=N, size=(batchsize,))
        indices2 = np.random.randint(low=0, high=N, size=(batchsize,))
        
        X1 = X[indices1,...].astype('float32') / 255.0
        X2 = X[indices2,...].astype('float32') / 255.0
        Y1 = Y[indices1]
        Y2 = Y[indices2]
        T  = (Y1 + Y2) # Sum of values.
        
        X1 = np.expand_dims(X1, axis=1) # For conv with theano, shape=(batchsize, channels, row, col).
        X2 = np.expand_dims(X2, axis=1) # We are just adding a dummy dimension saying that there is one channel.
        
        Y1 = np_utils.to_categorical(Y1, nb_classes)
        Y2 = np_utils.to_categorical(Y2, nb_classes)
        T  = np_utils.to_categorical(T, 19) # 19 possible values.
        
        yield {'input1':X1 , 'input2':X2},  {'out':T, 'aux1':Y1, 'aux2':Y2}
        
# Load data.
(X_train, Y_train), (X_valid, Y_valid) = mnist.load_data() # Shape = (N,28,28), (N,)
# Create generators.
batchsize  = 200 
data_train = data_generator(X_train, Y_train, batchsize)
data_valid = data_generator(X_valid, Y_valid, batchsize)

# Plot examples of the data.
inputs, outputs = data_train.next()
Xbatch = inputs['input1']
plt.figure(1, figsize=(12,3))
for i in range(10):
    plt.subplot(2,10,i+1)
    plt.imshow(inputs['input1'][i].reshape(28,28), cmap='gray', interpolation='nearest')
    plt.xticks([])
    plt.yticks([])
    plt.title('Target=%d' % np.argmax(outputs['out'][i]))
    plt.subplot(2,10,10+i+1)
    plt.imshow(inputs['input2'][i].reshape(28,28), cmap='gray', interpolation='nearest')
    plt.xticks([])
    plt.yticks([])
plt.savefig('a')
plt.close()
    
# Input data.
input1   = Input(shape=(1, 28,28), dtype='float32', name='input1') # Argument 'name' must match name in dictionary.
input2   = Input(shape=(1, 28,28), dtype='float32', name='input2')

# Layer one: Single convolutional layer applied to each input independently, with "shared weights."
nb_filter      = 32     # Number of convolutional kernels.
nb_row, nb_col = 7,7   # Convolution kernel size.
subsample      = (3,3) # Step size for convolution kernels.
conv  = Convolution2D(nb_filter, nb_row, nb_col, activation='relu', border_mode='same', subsample=subsample)
x1    = conv(input1) # Layer object conv transforms data.
x2    = conv(input2)

# Flatten data: transform from (28,28) to (784,)
x1 = Flatten()(x1)
x2 = Flatten()(x2)

# Layer two: Single fully-connected layer applied to each input independently, with shared weights.
layer = Dense(256, activation='relu')
x1 = layer(x1)
x2 = layer(x2)

# Auxiliary softmax layers try to classify digits. 
# The output of these layers aren't fed into the next layer.
layer = Dense(10) # These weights are shared.
aux1  = Activation(activation='softmax', name='aux1')(layer(x1)) # Output layers must be named.
aux2  = Activation(activation='softmax', name='aux2')(layer(x2))

# Merge hidden representations.
x = merge([x1, x2], mode='concat')

# More dense layers then output.
x   = Dense(512, activation='relu')(x)
out = Dense(19, activation='softmax', name='out')(x) # Output layers must be named.

# Create model object that puts it all together.
model = Model(input=[input1, input2], output=[out, aux1, aux2])
model.summary()

optimizer = Adam(lr=0.001, beta_1=0.99, beta_2=0.999, epsilon=1e-08) # Optimization hyperparameters.

model.compile(optimizer=optimizer,
              loss={'out':'categorical_crossentropy',
                    'aux1':'categorical_crossentropy',
                    'aux2':'categorical_crossentropy'},
              #loss_weights={'out': 1.0, 'aux1': 1.0, 'aux2':1.0}, # These can be tuned.
              loss_weights={'out': 1.0, 'aux1': 0.0, 'aux2':0.0}, # These can be tuned.
              metrics=['accuracy'])
              
# Callbacks can be used to stop early, decrease learning rate, checkpoint the model, etc.
#from keras.callbacks import EarlyStopping
#stopping  = keras.callbacks.EarlyStopping(monitor='val_loss', patience=0, verbose=0, mode='auto')
callbacks = []#[stopping]

# The fit_generator function loads data batches on the fly, instead of transfering entire data set to the gpu.
history   = model.fit_generator(generator=data_train, samples_per_epoch=50000, 
                              nb_epoch=5, verbose=1,
                              callbacks=callbacks, 
                              validation_data=data_valid, nb_val_samples=10000)

# Plot loss trajectory throughout training.
plt.figure(1, figsize=(10,5))
plt.plot(history.history['out_acc'], label='primary')
plt.plot(history.history['aux1_acc'], label='aux1')
plt.plot(history.history['aux2_acc'], label='aux2')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig('b')
plt.close()

# Plot examples of the data.
inputs, outputs = data_valid.next()
plt.figure(1, figsize=(12,3))
i = 0
while i < 10:
    inputs, outputs = data_valid.next()
    predictions = model.predict(inputs)
    j = 0
    while j < 100:
        yhat   = predictions[0][j].argmax(axis=-1) # predictions is list: [out, aux1, aux2]
        y      = np.argmax(outputs['out'][j])
        input1 = inputs['input1'][j].reshape(28,28)
        input2 = inputs['input2'][j].reshape(28,28)
        j += 1
        if yhat == y:
            continue
        
        plt.subplot(2,10,i+1)
        plt.imshow(input1, cmap='gray', interpolation='nearest')
        plt.xticks([])
        plt.yticks([])
        plt.title('Pred=%d' % yhat)
        plt.title('$y=%d,\hat{y}=%d$' % (y,yhat))
        plt.subplot(2,10,10+i+1)
        plt.imshow(input2, cmap='gray', interpolation='nearest')
        plt.xticks([])
        plt.yticks([])
        i += 1
        if not i<10:
            break
plt.savefig('c')
plt.close()
