import traindata
import cnnmodel
import math
import numpy
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from keras.utils import np_utils
from keras.callbacks import EarlyStopping
import sys
import argparse
import predictor

# TODO: export config so git won't kill us
load = True

# parse argument
parser = argparse.ArgumentParser(description='Whether to force recreate model.')
parser.add_argument(
	'-n', 
	'--new-model', 
	action='store_true', 
	help='use this argument to force model recreation.'
)
parser.add_argument(
	'-m', 
	'--model',
	default='model', 
	help='specify model to use (both JSON and h5 files must be of the given name)'
)
args = parser.parse_args()

load = not args.new_model
model_name = args.model

# percentage of data to keep as test set
test_size = 0.20
# number of set to load, load fewer during code testing
# None for load all
num_of_set = None

# fix random
numpy_seed = 7
sklearn_seed = 42
numpy.random.seed(numpy_seed)

classes = [chr(i) for i in range(ord('ก'), ord('ฮ')+1)]

#batch_size to train
batch_size = 50
# number of epochs to train
nb_epoch = 50


# ------------------------------------------------------------------------


if load:
	model = cnnmodel.load_model_from_json(model_name+'.json', model_name+'.h5')
	print('MODEL LOADED FROM DISK.')

else:
	# load our image data
	print("START BUILDING MODEL...")
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
	print("\tImages are of {}x{} size.".format(img_width, img_height))

	# reshape to be [samples][layer (black)][x][y]
	X_train = X_train.reshape(X_train.shape[0], 1, img_width, img_height).astype('float32')
	X_test = X_test.reshape(X_test.shape[0], 1, img_width, img_height).astype('float32')
	# normalize inputs from 0-255 to 0-1
	X_train = X_train / 255
	X_test = X_test / 255
	print('\t\tfitting model')
	early_stopping_monitor = EarlyStopping(patience=30)
	model,hist = cnnmodel.create_model(
		X_train, 
		y_train, 
		X_test, 
		y_test, 
		epochs=nb_epoch, 
		batch_size=batch_size, 
		callback_fn = [early_stopping_monitor],
		save=True
	)

	# visualizing losses and accuracy
	train_loss=hist.history['loss']
	val_loss=hist.history['val_loss']
	train_acc=hist.history['acc']
	val_acc=hist.history['val_acc']
	xc=range(nb_epoch)

	plt.figure(1,figsize=(7,5))
	plt.plot(xc,train_loss)
	plt.plot(xc,val_loss)
	plt.xlabel('num of Epochs')
	plt.ylabel('loss')
	plt.title('train_loss vs val_loss')
	plt.grid(True)
	plt.legend(['train','val'])
	print (plt.style.available) # use bmh, classic,ggplot for big pictures
	plt.style.use(['classic'])

	plt.figure(2,figsize=(7,5))
	plt.plot(xc,train_acc)
	plt.plot(xc,val_acc)
	plt.xlabel('num of Epochs')
	plt.ylabel('accuracy')
	plt.title('train_acc vs val_acc')
	plt.grid(True)
	plt.legend(['train','val'],loc=4)
	#print plt.style.available # use bmh, classic,ggplot for big pictures
	plt.style.use(['classic'])
	plt.show()

	# evaluation of the model
	scores = model.evaluate(X_test, y_test, verbose=0)
	print("\t\tCNN Error: %.2f%%" % (100-scores[1]*100))



# ======== TEST PREDICTION ==========

# Use the model to predict all known samples and evaluate.
predictor.evaluate(model)


