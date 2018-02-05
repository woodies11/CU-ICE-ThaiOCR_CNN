import numpy
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.utils import np_utils

# fix random result
_random_seed = 7
numpy.random.seed(_random_seed)

# load dataset
# X_train = array of images where each image is a 2-dimensional array
# containing gray scale pixel

# X is the features
# y is the label or the answer
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# ndarray.shape return [X, Y, Z, ...] etc dimension of the array
# we want to know the num of pixels of each image
# the first dimension is the collection of images
# we skip the first dimension and do Width x Height
num_pixels = X_train.shape[1] * X_train.shape[2]
num_train_samples = X_train.shape[0]
num_test_samples = X_test.shape[0]

# we want to reduce the dimension of the array by one
# so we reshape it by making each image a 1D array instead
# this help with the performance while still preserving the information.
# we also want to convert each pixel to 32 bits precision
X_train = X_train.reshape(num_train_samples, num_pixels).astype('float32')
X_test = X_test.reshape(num_test_samples, num_pixels).astype('float32')

# normalize dataset to percentage (0-1) instead of 0-255
X_train = X_train/255
X_test = X_test/255

# ONE HOT ENCODING
# because we are doing classification problem where
# each category is given the same weight, we need to
# convert our set to a binary matrix
# otherwise, the order or caterical index will confuse our ML
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_classes = y_test.shape[1]


def baseline_model():
	# create a model simple

	# TODO: RESEARCH
	# - softmax ACTIVATION FUNCTION
	# - caterical_crossentropy LOSS FUNCTION

	model = Sequential()
	# add a retifier layer
	model.add(Dense(num_pixels, input_dim=num_pixels, kernel_initializer='normal', activation='relu'))
	# add a hidden layer
	# softmax activation function turn outputs into probability-like value
	model.add(Dense(num_classes, kernel_initializer='normal', activation='softmax'))

	# compile model 
	# loss function: categorical_crossentropy
	# using ADAM gradient descent algorithm to learn the weights
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model


# build the model
model = baseline_model()

# fit model
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=200, verbose=2)

# evaluate the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("scores: " + str(scores))
print("Baseline Error: {:.2f}%".format(100-scores[1]*100))


# ======== TEST PREDICTION ==========

from scipy import misc
import matplotlib.pyplot as plt
import os

paths = os.listdir('samples')

for img_path in paths:

	# ignore system files
	if(img_path.startswith('.')):
		continue

	img = 255 - misc.imread('samples/'+img_path, flatten=True, mode='I')

	plt.imshow(img, cmap=plt.get_cmap('gray'))

	img_num_pixels = img.shape[0] * img.shape[1]
	img = img.reshape(1, img_num_pixels).astype('float32')
	img = img/255

	print(img.shape)

	pred = model.predict_classes(img)

	pred_proba = model.predict_proba(img)
	pred_proba = "%.2f%%" % (pred_proba[0][pred]*100)

	print(pred[0], "with probability", pred_proba)
	plt.show();