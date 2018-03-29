import logging, os, glob
from keras.models import model_from_json
import matplotlib.pyplot as plt
import numpy as np
import re
from pathlib import Path
from keras.callbacks import ModelCheckpoint
from sklearn.metrics import confusion_matrix
import itertools

def setup_logger(name, log_file, level=logging.DEBUG, format='%(levelname)-7s|%(module)s|%(asctime)s: %(message)s'):
    """Function setup as many loggers as you want"""

    # create a file handler to write to file
    formatter = logging.Formatter(format)
    handler = logging.FileHandler(log_file)        
    handler.setFormatter(formatter)

    # and also print to console
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)

    # clear previously register handler first
    logger.handler = []
    logger.addHandler(handler)
    logger.addHandler(console_handler)

    return logger

def makedirifnotexist(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def listdir_nohidden(path):
	directories = glob.glob(path+'/**/*', recursive=True)
	return [file for file in directories if os.path.isfile(file)]


#################################################################################################
# Experiment Class start here:
#################################################################################################

class Experiment(object):

    ################################################
    # General description.
    # Override and set these for yourself.
    # Will be used for output files naming.

    EXPERIMENT_NAME = "GENERIC EXPERIMENT"
    EXPERIMENT_DESCRIPTION = """
    A base class template for an experiment.
    """

    ################################################

    CLASSES = [chr(i) for i in range(ord('ก'), ord('ฮ')+1) if not i == ord('ฤ') or not i == ord('ฦ')]

    def __init__(self, nameprefix="", namesuffix=""):
        self.INSTANCE_NAME = nameprefix + self.EXPERIMENT_NAME + namesuffix

        # dafault directory to save models
        self.MODEL_DIRECTORY = "./experiments/models/{}/".format(self.INSTANCE_NAME)
        self.RESULT_STATISTIC_DIRECTORY = "./experiments/results/{}/".format(self.INSTANCE_NAME)

        self.BASE_NAME_FORMAT = "{}_b{}-model"
        self.NAME_FORMAT = self.BASE_NAME_FORMAT.format("{}", "{}_e{}")

        makedirifnotexist(self.MODEL_DIRECTORY)
        makedirifnotexist(self.RESULT_STATISTIC_DIRECTORY)
        makedirifnotexist("./experiments/logs/")

    # ========================================================================================================
    # These MUST BE OVERRIDDEN and IMPLEMENTED
    # ========================================================================================================

    def _model_from_json(self, json, **kwargs):
        """
        Call keras.models.model_from_json(json, custom_objects)
        with appropiate custom_objects.

        IMPORTANT:
        ----------
        The json object will be supplied by run(),
        please only load AND compile the model then return,
        no need to load weight.
        """
        raise NotImplementedError

    def predict(self, model, test_sample, classes, **kwargs):
        """
        Should return the prediction in whatever format evaluate() will use.
        """
        raise NotImplementedError

    # These are chained call by the internal methods.
    # You SHOULD OVERRIDE these to implement your own
    # function but SHOULD NOT CALL THEM YOURSELF.
    # Let the internal methods call them.

    def _compilemodel(self, model, **kwargs):
        raise NotImplementedError

    def _createmodel(self, X_train, y_train, X_test, y_test, batch_size, epochs, **kwargs):
        raise NotImplementedError

    #########################################################################################################

    # ========================================================================================================
    # These are handy default implementation. You can override these if needed.
    # ========================================================================================================

    def _fitmodel(self, model, X_train, y_train, X_test, y_test, batch_size, epochs, **kwargs):
        name = self.INSTANCE_NAME.replace(' ', '_').lower()
        formatted_name="{}-b{}-e{}-va{}.h5".format(name, batch_size, "{epoch:02d}", "{val_acc:.2f}")
        filepath = self.MODEL_DIRECTORY + formatted_name
        checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=False, mode='max')
        
        hist = model.fit(
            X_train, 
            y_train, 
            validation_data=(X_test, y_test), 
            epochs=epochs, 
            batch_size=batch_size,
            callbacks=[checkpoint]
	    )

    def _model_name_from_parameters(self, batch_size, **kwargs):
        """
        Use to create a name for saving model.
        Note that this name should be able to uniquely
        identify each set of parameters used to generate
        the model. 
        
        E.g. ResNet18_b100_e10-model for ResNet18 batch_size: 100, epochs: 10

        DEFAULT:
        -------- 
        return the experiment name in lower case + "_b<batch_size>-model"
        """
        name = self.INSTANCE_NAME.replace(' ', '_').lower()
        return self.BASE_NAME_FORMAT.format(name, batch_size)

    def savemodel(self, model, name, directory=None, **kwargs):
        """
        Save model as <name>.json and <name>.h5 to directory.

        model - the model to be saved
        name - will be used as file name
        directory - is the path to folder to save the model to

        Generally, model are save in exactly the same manner
        so there shouldn't be a need to override this.
        """

        if directory is None:
            directory = self.MODEL_DIRECTORY

        # make sure directory path ended with the last / 
        if not directory.endswith('/'):
            directory = directory + '/'
            
        # serialize model to JSON
        model_json = model.to_json()
        with open(directory + name +'.json', 'w') as json_file:
            json_file.write(model_json)

        # log
        self.general_logger.info("MODEL: {} SAVED TO {}".format(name, directory))

    def loadmodel(self, json_file, weights_file, **kwargs):
        """
        Load model from json and h5 file.
        This method DOES NOT construct and compile model by itself.
        It will load the JSON file and pass it to _model_from_json()
        in order to load the model. This is done because each 
        """

        # add in file extension if not already added
        if not json_file.endswith('.json'):
            json_file += '.json'

        if not weights_file.endswith('.h5'):
            weights_file += '.h5'

        # load json
        json_file = open(json_file, 'r')
        loaded_model_json = json_file.read()
        json_file.close()

        # use experiment specific method to construct
        # the model from JSON
        model = self._model_from_json(loaded_model_json, **kwargs)

        model = self._compilemodel(model, **kwargs)

        # load weights into new model
        model.load_weights(weights_file)

        # return the ready-to-use model
        return model

    def _generate_bar_char_img(self, classes_acc, name, directory=None, title=None, xlabel=""):

        if directory is None:
            directory = self.RESULT_STATISTIC_DIRECTORY

        # make sure directory path ended with the last / 
        if not directory.endswith('/'):
            directory = directory + '/'

        if title is None:
            title = name
        
        save_path = directory + name
        # . will mess up savefig
        # However, the very first one is a legit . for ./PATH so we need to keep it
        # by ignoring the first, now -, character and add the . back on
        save_path = "." + save_path.replace(".", "-")[1:]

        # Pop the two character we ignored out if exist.
        classes_acc.pop('ฤ', None)
        classes_acc.pop('ฦ', None)

        fig = plt.figure()
        ax = fig.gca()
        d = classes_acc
        X = np.arange(len(d))
        C = [
            'g' if classes_acc[k] >= 0.7 else 
            ('y' if classes_acc[k] >= 0.5 else 'r') 
            for k in classes_acc
        ]
        plt.title(title)
        plt.xlabel(xlabel)
        plt.bar(X, d.values(), color=C, align='center', width=0.5)
        plt.axhline(0.7, color='g', linestyle='dashed', linewidth=1)
        plt.axhline(0.5, color='y', linestyle='dashed', linewidth=1)
        plt.xticks(X, d.keys(), fontname='Tahoma')
        plt.ylim(0, 1.1)

        # save figure to the save_path
        plt.savefig(save_path)
        plt.close()
        self.general_logger.info("STATISTIC: Bar chart for {} saved to {}".format(name, save_path))

    def _generate_confusion_matrix(self, all_class, all_label, all_pred, name):
        cm = confusion_matrix(all_label, all_pred)
        cm_plot_labels = all_class
        self.plot_confusion_matrix(cm, cm_plot_labels, name)

    def plot_confusion_matrix(self, cm, classes, name,
                          directory=None,
                          title=None,
                          showVal = False,
                          normalize=False):
        """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        """
        if directory is None:
            directory = self.RESULT_STATISTIC_DIRECTORY

        # make sure directory path ended with the last / 
        if not directory.endswith('/'):
            directory = directory + '/'

        if title is None:
            title = name

        save_path = directory + name
        # . will mess up savefig
        # However, the very first one is a legit . for ./PATH so we need to keep it
        # by ignoring the first, now -, character and add the . back on
        save_path = "." + save_path.replace(".", "-")[1:] + "_cm"
        if showVal:
            plt.figure(dpi=200)
            cmap = plt.cm.Blues
        else:
            plt.figure(dpi=150)
            cmap = plt.cm.Greys
        
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, fontname='Tahoma', fontsize=7)
        plt.yticks(tick_marks, classes, fontname='Tahoma', fontsize=7)
        log_text = "Confusion matrix:"
        if showVal:
            if normalize:
                cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
                log_text += " Normalized"
            else:
                log_text += " without normalization"

        # print(cm)

        thresh = cm.max() / 2.
        if showVal:
            # warning, showing value in such dense matrix will not look so nice
            # try change font size, canvas size, dpi
            for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
                val_text = cm[i, j]
                if val_text==0 :
                    val_text = ""
                plt.text(j, i, 
                         val_text,
                         horizontalalignment="center",
                         color="white" if cm[i, j] > thresh else "black",
                         fontname='Tahoma',
                         fontsize=3)
        else:
            for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
                plt.text(j, i, 
                         "",
                         horizontalalignment="center",
                         color="white" if cm[i, j] > thresh else "black")

        # plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')

        # save figure to the save_path
        plt.savefig(save_path)
        plt.close()
        self.general_logger.info(log_text + " for {} saved to {}".format(name, save_path))

    # ========================================================================================================
    # These SHOULD NOT BE OVERRIDDEN
    # ========================================================================================================
    # These are internal method called by run().
    # They are for implementing preprocess such as
    # logging, etc.
    # YOU SHOULD NOT OVERRIDE THESE!
    
    def __internal_createmodel(self, dataset, batch_size, epochs, **kwargs):
        """
        note that dataset are in from of (X_train, y_train, X_test, y_test)
        we expand it into X_train, y_train, X_test, y_test for our function
        for easier use
        """
        return self._createmodel(*dataset, batch_size, epochs, **kwargs)

    def __internal_fitmodel(self, model, dataset, batch_size, epochs, **kwargs):
        """
        note that dataset are in from of (X_train, y_train, X_test, y_test)
        we expand it into X_train, y_train, X_test, y_test for our function
        for easier use
        """
        return self._fitmodel(model, *dataset, batch_size, epochs, **kwargs)

    def run(self, dataset, batch_size, epochs, **kwargs):
        """
        Run the experiment using the given parameters.

        The parameters (kwargs) for this iteration will be given by the
        testbench in the form of a keyword-dictionary.

        dataset must be in the form of (X_train, y_train, X_test, y_test)
        """

        self.result_logger = setup_logger(self.INSTANCE_NAME+"-result-run", "./experiments/logs/results.txt")
        self.general_logger = setup_logger(self.INSTANCE_NAME+"-general-run", "./experiments/logs/experiments.txt")

        model_name = self._model_name_from_parameters(batch_size, **kwargs)

        model = self.__internal_createmodel(dataset, batch_size, epochs, **kwargs)

        # save this model (h5 files are saved using fitting checkpoints)
        self.savemodel(model, model_name, **kwargs)

        # start fitting model
        self.__internal_fitmodel(model, dataset, batch_size, epochs, **kwargs)
        
        

    def gen_statistic(self, test_samples, batch_size, **kwargs):

        self.result_logger = setup_logger(self.INSTANCE_NAME+"-result-stat", "./experiments/logs/results.txt")
        self.general_logger = setup_logger(self.INSTANCE_NAME+"-general-stat", "./experiments/logs/experiments.txt")

        self.generate_all_bar_chars(test_samples, batch_size, **kwargs)

    def generate_all_bar_chars(self, test_samples, batch_size, **kwargs):

        directory = self.MODEL_DIRECTORY
        # make sure directory path ended with the last / 
        if not directory.endswith('/'):
            directory = directory + '/'

        model_paths = listdir_nohidden(directory)
        json_file = directory + self._model_name_from_parameters(batch_size, **kwargs)

        for model_weight in model_paths:
            if not model_weight.endswith('.h5'):
                continue
            model_name = model_weight.split('/')[-1].split('\\')[-1][:-3]

            model = self.loadmodel(json_file, model_weight, **kwargs)

            # evaluate model
            classes_acc, overall_acc, correct_count, test_data_count, all_class, all_label, all_pred = self.evaluate(model, test_samples, **kwargs)
            xlabel = '{}/{} correct ({})'.format(correct_count, test_data_count, overall_acc)

            # generate and save plot
            self._generate_bar_char_img(classes_acc, model_name, xlabel=xlabel)
            self._generate_confusion_matrix(all_class, all_label, all_pred, model_name)

    def __internal_predict(self, model, test_sample, **kwargs):
        classes = Experiment.CLASSES
        return self.predict(model, test_sample, classes, **kwargs)

    def evaluate(self, model, test_samples, **kwargs):

        # { character : [count, right] }
        classes_dict = {c:[0,0] for c in self.CLASSES}

        correct_count = 0
        test_data_count = 0
        all_class = []
        all_label = []
        all_pred = []

        for class_key in test_samples:
            samples = test_samples[class_key]
            sample_size = len(samples)
            test_data_count += sample_size
            if sample_size != 0:
                all_class.append(class_key)
            for img in samples:
    
                pred_class = self.__internal_predict(model, img)

                all_label.append(class_key)
                all_pred.append(pred_class)

                is_correct = str(pred_class) == str(class_key)

                classes_dict[class_key][0] += 1
                if is_correct:
                    correct_count += 1
                    classes_dict[class_key][1] += 1

        # -- end outer for --

        # prevent divide by zero so test can continue
        if test_data_count == 0:
            test_data_count = -1

        classes_acc = {k:(classes_dict[k][1]/classes_dict[k][0] if classes_dict[k][0] > 0 else 0) for k in classes_dict}
        overall_acc = correct_count/test_data_count
        self.general_logger.info('{}/{} correct ({})'.format(correct_count, test_data_count, overall_acc))
        
        return classes_acc, overall_acc, correct_count, test_data_count, all_class, all_label, all_pred