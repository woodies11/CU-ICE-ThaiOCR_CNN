from .experiment import Experiment
import glob, os
import numpy
import keras
import keras_resnet.models
from keras.utils import np_utils
from keras import backend as K
K.set_image_dim_ordering('th')
from keras.models import model_from_json

class RESNET18(Experiment):

    EXPERIMENT_NAME = "RESNET18"
    EXPERIMENT_DESCRIPTION = """
    Model trained based on the ResNet18 library.
    """

    def _model_from_json(self, json, **kwargs):
        return model_from_json(json, custom_objects=keras_resnet.custom_objects)

    def predict(self, model, test_sample, classes, **kwargs):
        y_prob = model.predict(test_sample)
        pred = y_prob.argmax(axis=-1)
        pred_class = classes[pred[0]]
        return pred_class

    def _createmodel(self, X_train, y_train, X_test, y_test, batch_size, epochs, **kwargs):
        num_classes = y_test.shape[1]
        img_width = X_train.shape[2]
        img_height = X_train.shape[3]
        shape, classes = (1, img_width, img_height), num_classes

        x = keras.layers.Input(shape)
        model = keras_resnet.models.ResNet18(x, classes=classes)
        model.compile("adam", "categorical_crossentropy", ["accuracy"])
        return model

    def _compilemodel(self, model, **kwargs):
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model