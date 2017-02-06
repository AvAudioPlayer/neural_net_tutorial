
from keras.datasets import mnist
import numpy as np
(x_train, _), (x_test, _) = mnist.load_data()

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

import cPickle as pickle
encoded_imgs = pickle.load( open( "1176_encoded_imgs.pkl", "rb" ) )
decoded_imgs = pickle.load( open( "1176_decoded_imgs.pkl", "rb" ) )
# use Matplotlib (don't ask)
import matplotlib.pyplot as plt

n = 4  # how many digits we will display
plt.figure(figsize=(8, 4))
for i in range(n):
    # display original
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(x_test[10*i+1].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(decoded_imgs[10*i+1].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()
plt.savefig('1176_encoding_dim')
