import traindata
import cnnmodel
import math
import numpy
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from keras.utils import np_utils

# percentage of data to keep as test set
test_size = 0.20
# number of set to load, load fewer during code testing
# None for load all
num_of_set = None

# fix random
numpy_seed = 7
sklearn_seed = 42
numpy.random.seed(numpy_seed)

# load our image data
X_set, _y_set = traindata.load_image_data(num_of_set)

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
		epochs=2, 
		batch_size=50, 
		save=True
	)

# evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("CNN Error: %.2f%%" % (100-scores[1]*100))

# ======== TEST PREDICTION ==========

from scipy import misc
import matplotlib.pyplot as plt
import os
import imageutil

sample_path = 'th_samples'

paths = os.listdir(sample_path)

correct_count = 0
test_data_count = 0

for img_path in paths:

	# ignore system files
	if(img_path.startswith('.')):
		continue	

	test_data_count += 1

	img = imageutil.readimageinput(sample_path+'/'+img_path, True, False, 0.1)

	ans = img_path.split('-')[0]

	pred = model.predict_classes(img)

	pred_proba = model.predict_proba(img)
	pred_proba = "%.2f%%" % (pred_proba[0][pred]*100)

	pred_class = classes[pred[0]]

	is_correct = (str(pred_class) == str(ans))

	if is_correct:
		correct_count += 1

	print(pred_class, "with probability", pred_proba, 'which is', is_correct)

	plt.show();

print('{}/{} correct ({})'.format(correct_count, test_data_count, correct_count/test_data_count))

