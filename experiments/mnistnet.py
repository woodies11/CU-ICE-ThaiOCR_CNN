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

def listdir_nohidden(path):
	directories = glob.glob(path+'/**/*', recursive=True)
	return [file for file in directories if os.path.isfile(file)]

class MNISTNET(Experiment):

    EXPERIMENT_NAME = "SIMPLE MNISTNET"
    EXPERIMENT_DESCRIPTION = """
    Models trained using a simple network originally intended for MNIST.
    """

    classes = [chr(i) for i in range(ord('ก'), ord('ฮ')+1)]
    classes_dict = {chr(i):[0,0] for i in range(ord('ก'), ord('ฮ')+1)}

    def _model_from_json(self, json, **kwargs):
        return model_from_json(json)

    def predict(self, model, test_sample, **kwargs):
        pred = model.predict_classes(test_sample)
        pred_class = MNISTNET.classes[pred[0]]
        return pred_class

    def evaluate(self, model, test_samples, **kwargs):

        # { character : [count, right] }
        classes = MNISTNET.classes
        classes_dict = MNISTNET.classes_dict

        correct_count = 0
        test_data_count = 0

        for class_key in test_samples:
            samples = test_samples[class_key]
            test_data_count += len(samples)
            for img in samples:
                pred = model.predict_classes(img)
                
                pred_proba = model.predict_proba(img)
                pred_proba = "%.2f%%" % (pred_proba[0][pred]*100)

                pred_class = classes[pred[0]]

                is_correct = str(pred_class) == str(class_key)

                classes_dict[class_key][0] += 1
                if is_correct:
                    correct_count += 1
                    classes_dict[class_key][1] += 1

        # -- end outer for --

        # prevent divide by zero so test can continue
        if test_data_count == 0:
            test_data_count = -1

        self.general_logger.info('{}/{} correct ({})'.format(correct_count, test_data_count, correct_count/test_data_count))
        return {k:(classes_dict[k][1]/classes_dict[k][0] if classes_dict[k][0] > 0 else 0) for k in classes_dict}

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
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model

    def _compilemodel(self, model, **kwargs):
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model