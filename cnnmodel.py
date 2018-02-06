import numpy
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


def save_model_to_json(model, name='model'):
	# serialize model to JSON
	model_json = model.to_json()
	with open(name+'.json', 'w') as json_file:
		json_file.write(model_json)

	# serialize weights to HDF5
	model.save_weights(name+'.h5')
	print('model saved')


def load_model_from_json(json_file='model.json', weights_file='model.h5'):
	## TODO: allow handling of exception if file not found
	
	# load json and create model
	json_file = open('model.json', 'r')
	loaded_model_json = json_file.read()
	json_file.close()
	model = model_from_json(loaded_model_json)
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

	# load weights into new model
	model.load_weights("model.h5")

	return model


def create_model(X_train, y_train, X_test, y_test, epochs, batch_size, save=True):

	def init_model():

		num_classes = y_test.shape[1]
		img_width = X_train.shape[2]
		img_height = X_train.shape[3]

		# create model
		model = Sequential()
		model.add(Conv2D(30, (5, 5), input_shape=(1, img_width, img_height), activation='relu'))
		model.add(MaxPooling2D(pool_size=(2, 2)))
		model.add(Conv2D(15, (3, 3), activation='relu'))
		model.add(MaxPooling2D(pool_size=(2, 2)))
		model.add(Dropout(0.2))
		model.add(Flatten())
		model.add(Dense(128, activation='relu'))
		model.add(Dense(50, activation='relu'))
		model.add(Dense(num_classes, activation='softmax'))
		# Compile model
		model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
		return model

	model = init_model()

	model.fit(
		X_train, 
		y_train, 
		validation_data=(X_test, y_test), 
		epochs=epochs, 
		batch_size=batch_size
	)

	if save:
		save_model_to_json(model)

	return model