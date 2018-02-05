import traindata
import cnnmodel
import math
import numpy
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from keras.utils import np_utils

# percentage of data to keep as test set
test_size = 0.20


# fix random
numpy_seed = 7
sklearn_seed = 42
numpy.random.seed(numpy_seed)

# load our image data
X_set, _y_set = traindata.load_image_data(10)

# convert our label to one-hot-encoded matrix
encoder = LabelBinarizer()
encoder.fit(_y_set)
classes = encoder.classes_
y_set = encoder.transform(_y_set)

# partition data into test set and train set
X_train, X_test, y_train, y_test = train_test_split(
	X_set, 
	y_set, 
	test_size=test_size,
	random_state=sklearn_seed
)

img_width = X_set.shape[1]
img_height = X_set.shape[2]
print("Images are of {}x{} size.".format(img_width, img_height))

# reshape to be [samples][layer (black)][x][y]
X_train = X_train.reshape(X_train.shape[0], 1, img_width, img_height).astype('float32')
X_test = X_test.reshape(X_test.shape[0], 1, img_width, img_height).astype('float32')
# normalize inputs from 0-255 to 0-1
X_train = X_train / 255
X_test = X_test / 255


# ------------------------------------------------------------------------
load = False

if load:
	model = cnnmodel.load_model_from_json('model.json', 'model.h5')
	print('model loaded from disk')
else:
	print('fitting model')
	model = cnnmodel.create_model(
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
print("CNN Error: %.2f%%" % (100-scores[1]*100))