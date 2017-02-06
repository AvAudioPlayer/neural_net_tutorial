from keras.layers import Input, Dense
from keras.models import Model

# this is the size of our encoded representations
encoding_dim = 1176  # 32 floats -> compression of factor 24.5, assuming the input is 784 floats

# this is our input placeholder
input_img = Input(shape=(784,))
# "encoded" is the encoded representation of the input
encoded = Dense(encoding_dim, activation='relu')(input_img)
# "decoded" is the lossy reconstruction of the input
decoded = Dense(784, activation='sigmoid')(encoded)

# this model maps an input to its reconstruction
autoencoder = Model(input=input_img, output=decoded)

# this model maps an input to its encoded representation
encoder = Model(input=input_img, output=encoded)

# create a placeholder for an encoded (32-dimensional) input
encoded_input = Input(shape=(encoding_dim,))
# retrieve the last layer of the autoencoder model
decoder_layer = autoencoder.layers[-1]
# create the decoder model
decoder = Model(input=encoded_input, output=decoder_layer(encoded_input))
print decoder.get_weights()

import numpy as np
np.savetxt(str(encoding_dim)+'_encoder_edges_before_training', encoder.get_weights()[0])
np.savetxt(str(encoding_dim)+'_encoder_nodes_before_training', encoder.get_weights()[1])
np.savetxt(str(encoding_dim)+'_decoder_0_before_training', decoder.get_weights()[0])
np.savetxt(str(encoding_dim)+'_decoder_1_before_training', decoder.get_weights()[1])

autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

from keras.datasets import mnist
(x_train, _), (x_test, _) = mnist.load_data()

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
print x_train.shape
print x_test.shape

autoencoder.fit(x_train, x_train,
                nb_epoch=50,
                batch_size=256,
                shuffle=True,
                validation_data=(x_test, x_test))
                
np.savetxt(str(encoding_dim)+'_encoder_edges_after_training', encoder.get_weights()[0])
np.savetxt(str(encoding_dim)+'_encoder_nodes_after_training', encoder.get_weights()[1])
np.savetxt(str(encoding_dim)+'_decoder_0_after_training', decoder.get_weights()[0])
np.savetxt(str(encoding_dim)+'_decoder_1_after_training', decoder.get_weights()[1])

encoder.save_weights(str(encoding_dim)+'_encoder_weights')
decoder.save_weights(str(encoding_dim)+'_decoder_weights')
# encode and decode some digits
# note that we take them from the *test* set
encoded_imgs = encoder.predict(x_test)
decoded_imgs = decoder.predict(encoded_imgs)

import cPickle as pickle
pickle.dump( encoded_imgs, open( str(encoding_dim)+"_encoded_imgs.pkl", "wb" ) )
pickle.dump( decoded_imgs, open( str(encoding_dim)+"_decoded_imgs.pkl", "wb" ) )
#pickle.dump( encoder, open( "1176_encoder.pkl", "wb" ) )
#pickle.dump( decoder, open( "1176_decoder.pkl", "wb" ) )
#pickle.dump( autoencoder, open( "1176_autoencoder.pkl", "wb" ) )
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
plt.savefig(str(encoding_dim)+'_encoding_dim')

