from .experiment import Experiment
import glob, os
import numpy
import keras
import keras_resnet.models
from keras.utils import np_utils
from keras import backend as K
K.set_image_dim_ordering('th')
from keras.models import model_from_json
from collections import defaultdict
import operator

class SUPERMODEL(Experiment):

    EXPERIMENT_NAME = "SUPERMODEL"
    EXPERIMENT_DESCRIPTION = """
    Model trained based on the ResNet18 library.
    """

    MODEL_PAHTS = [
        ["./experiments/models/RESNET50-b100/resnet50-b100_b100-model.json","./experiments/models/RESNET50-b100/resnet50-b100-b100-e07-va0.97.h5"],
        ["./experiments/models/RESNET50-b100/resnet50-b100_b100-model.json","./experiments/models/RESNET50-b100/resnet50-b100-b100-e08-va0.97.h5"],
        ["./experiments/models/RESNET50-b100/resnet50-b100_b100-model.json","./experiments/models/RESNET50-b100/resnet50-b100-b100-e28-va0.97.h5"],
        ["./experiments/models/RESNET50-b100/resnet50-b100_b100-model.json","./experiments/models/RESNET50-b100/resnet50-b100-b100-e22-va0.97.h5"],
        ["./experiments/models/RESNET50-b100/resnet50-b100_b100-model.json","./experiments/models/RESNET50-b100/resnet50-b100-b100-e12-va0.97.h5"],
        ["./experiments/models/RESNET18-b200/resnet18-b200_b200-model.json","./experiments/models/RESNET18-b200/resnet18-b200-b200-e10-va0.98.h5"],
    ]

    def _model_from_json(self, json, **kwargs):
        return model_from_json(json, custom_objects=keras_resnet.custom_objects)

    def predict(self, model, test_sample, classes, **kwargs):

        # dict with default value of 0 (never KeyError)
        predicted_class = defaultdict(int)

        # collect the prediction of all models
        for m in model:
            y_prob = m.predict(test_sample)
            pred = y_prob.argmax(axis=-1)
            pred_class = classes[pred[0]]
            predicted_class[pred_class] += 1

        pred_class = max(predicted_class.items(), key=operator.itemgetter(1))[0]

        return pred_class

    def _createmodel(self, X_train, y_train, X_test, y_test, batch_size, epochs, **kwargs):
        raise NotImplementedError("this should only be use for statistic")

    def _compilemodel(self, model, **kwargs):
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model

    def gen_statistic(self, test_samples, batch_size, **kwargs):
        models = [self.loadmodel(m[0], m[1]) for m in SUPERMODEL.MODEL_PAHTS]

        # evaluate model
        classes_acc, overall_acc, correct_count, test_data_count = self.evaluate(models, test_samples, **kwargs)
        xlabel = '{}/{} correct ({})'.format(correct_count, test_data_count, overall_acc)

        # generate and save plot
        self._generate_bar_char_img(classes_acc, "SUPERMODEL", xlabel=xlabel)