# Larger CNN for the MNIST Dataset
import numpy
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
K.set_image_dim_ordering('th')
from keras.models import model_from_json
import cnn_model


# ------------------------------------------------------------------------
# Prepare Data
# ------------------------------------------------------------------------
# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)
# load data
(X_train, y_train), (X_test, y_test) = mnist.load_data()
# reshape to be [samples][pixels][width][height]
X_train = X_train.reshape(X_train.shape[0], 1, 28, 28).astype('float32')
X_test = X_test.reshape(X_test.shape[0], 1, 28, 28).astype('float32')
# normalize inputs from 0-255 to 0-1
X_train = X_train / 255
X_test = X_test / 255
# one hot encode outputs
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

# ------------------------------------------------------------------------


load = False

if load:
	model = cnn_model.load_model_from_json('model.json', 'model.h5')
	print('model loaded from disk')
else:
	print('fitting model')
	model = cnn_model.create_model(
		X_train, 
		y_train, 
		X_test, 
		y_test, 
		epochs=1, 
		batch_size=200, 
		save=True
	)

# evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("Large CNN Error: %.2f%%" % (100-scores[1]*100))

# ======== TEST PREDICTION ==========

from scipy import misc
import matplotlib.pyplot as plt
import os
import imageutil

paths = os.listdir('samples')

correct_count = 0
test_data_count = 0

for img_path in paths:

	# ignore system files
	if(img_path.startswith('.')):
		continue	

	test_data_count += 1

	img = imageutil.readimageinput('samples/'+img_path, True, True, 0.1)

	ans = img_path.split('-')[0]

	pred = model.predict_classes(img)

	pred_proba = model.predict_proba(img)
	pred_proba = "%.2f%%" % (pred_proba[0][pred]*100)

	is_correct = (str(pred[0]) == str(ans))

	if is_correct:
		correct_count += 1

	print(pred[0], "with probability", pred_proba, 'which is', is_correct)

	plt.show();

print('{}/{} correct ({})'.format(correct_count, test_data_count, correct_count/test_data_count))

