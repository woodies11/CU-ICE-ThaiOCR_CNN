from .experiment import Experiment
import glob, os
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

class MNISTNET(Experiment):

    EXPERIMENT_NAME = "SIMPLE MNISTNET"
    EXPERIMENT_DESCRIPTION = """
    Models trained using a simple network originally intended for MNIST.
    """

    def _model_from_json(self, json, **kwargs):
        return model_from_json(json)

    def predict(self, model, test_sample, classes, **kwargs):
        pred = model.predict_classes(test_sample)
        pred_class = classes[pred[0]]
        return pred_class

    def _createmodel(self, X_train, y_train, X_test, y_test, batch_size, epochs, **kwargs):
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
        return self._compilemodel(model, **kwargs)

    def _compilemodel(self, model, **kwargs):
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model