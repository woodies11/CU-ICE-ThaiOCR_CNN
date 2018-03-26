import logging, os, glob
from keras.models import model_from_json
import matplotlib.pyplot as plt
import numpy as np
import re
from pathlib import Path
from keras.callbacks import ModelCheckpoint

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

    def __init__(self, nameprefix=""):
        self.INSTANCE_NAME = nameprefix + self.EXPERIMENT_NAME

        # dafault directory to save models
        self.MODEL_DIRECTORY = "./experiments/models/{}/".format(self.INSTANCE_NAME)
        self.RESULT_STATISTIC_DIRECTORY = "./experiments/results/{}/".format(self.INSTANCE_NAME)

        self.BASE_NAME_FORMAT = "{}_b{}-model"
        self.NAME_FORMAT = self.BASE_NAME_FORMAT.format("{}", "{}_e{}")

        makedirifnotexist(self.MODEL_DIRECTORY)
        makedirifnotexist(self.RESULT_STATISTIC_DIRECTORY)
        makedirifnotexist("./experiments/logs/")

        self.result_logger = setup_logger(self.INSTANCE_NAME+"-result", "./experiments/logs/results.txt")
        self.general_logger = setup_logger(self.INSTANCE_NAME+"-general", "./experiments/logs/experiments.txt")

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

    def predict(self, model, test_sample, **kwargs):
        """
        Should return the prediction in whatever format evaluate() will use.
        """
        raise NotImplementedError

    def evaluate(self, model, test_samples, **kwargs):
        """

        test_samples should be in format:
        [
            'ก': [ImgObject],
            'ข': [ImgObject],
            ...
        ]

        Should return classes_acc in the format:
        [
            'ก': 0.90,
            'ข': 0.75,
            ...
        ]

        where the key is the class and the value is the accuracy from 100%
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

    def _fitmodel(self, model, X_train, y_train, X_test, y_test, batch_size, epochs, callbacks=[], **kwargs):
        name = self.INSTANCE_NAME.replace(' ', '_').lower()
        formatted_name="{}-b{}-e{}-va{}.h5".format(name, batch_size, "{epoch:02d}", "{val_acc:.2f}")
        filepath = self.MODEL_DIRECTORY + formatted_name
        checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=False, mode='max')
        callbacks.append(checkpoint)
        
        hist = model.fit(
            X_train, 
            y_train, 
            validation_data=(X_test, y_test), 
            epochs=epochs, 
            batch_size=batch_size,
            callbacks=callbacks
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

    def _generate_bar_char_img(self, classes_acc, name, directory=None, title=None):

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

        fig = plt.figure()
        ax = fig.gca()
        d = classes_acc
        X = np.arange(len(d))
        C = [
            'g' if classes_acc[k] >= 0.7 else 
            ('y' if classes_acc[k] >= 0.5 else 'r') 
            for k in classes_acc
        ]
        plt.bar(X, d.values(), color=C, align='center', width=0.5)
        plt.axhline(0.7, color='g', linestyle='dashed', linewidth=1)
        plt.axhline(0.5, color='y', linestyle='dashed', linewidth=1)
        plt.xticks(X, d.keys(), fontname='Tahoma')
        plt.ylim(0, 1.1)
        plt.title = title

        # save figure to the save_path
        plt.savefig(save_path)
        plt.close()
        self.general_logger.info("BAR CHART for {} saved to {}.".format(name, save_path))

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

    def run(self, dataset, test_samples, batch_size, epochs, **kwargs):
        """
        Run the experiment using the given parameters.

        The parameters (kwargs) for this iteration will be given by the
        testbench in the form of a keyword-dictionary.

        dataset must be in the form of (X_train, y_train, X_test, y_test)
        """

        model_name = self._model_name_from_parameters(batch_size, **kwargs)

        model = self.__internal_createmodel(dataset, batch_size, epochs, **kwargs)

        # start fitting model
        self.__internal_fitmodel(model, dataset, batch_size, epochs, **kwargs)
        
        # save this model (h5 files are already saved using fitting checkpoints)
        self.savemodel(model, model_name, **kwargs)

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
            classes_acc = self.evaluate(model, test_samples, **kwargs)

            # generate and save plot
            self._generate_bar_char_img(classes_acc, model_name)