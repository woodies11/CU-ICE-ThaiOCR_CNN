import numpy
import theano
import keras

print('ENVIRONMENT INFOMATION:')
print('Numpy: ' + numpy.__version__)
print('Theano: ' + theano.__version__)
print('Keras: ' + keras.__version__)


# download and import mnist using keras
from keras.datasets import mnist
import matplotlib.pyplot as plt

# load the MNIST dataset 
# (will download if not cached, require connection)
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# plot 4 images as gray scale
# each X_train[] is a 2 dimensional array 
# containing gray scale image data
plt.subplot(221)
plt.imshow(X_train[0], cmap=plt.get_cmap('gray'))
plt.subplot(222)
plt.imshow(X_train[1], cmap=plt.get_cmap('gray'))
plt.subplot(223)
plt.imshow(X_train[2], cmap=plt.get_cmap('gray'))
plt.subplot(224)
plt.imshow(X_train[3], cmap=plt.get_cmap('gray'))
print('displaying dataset samples...')
plt.show()