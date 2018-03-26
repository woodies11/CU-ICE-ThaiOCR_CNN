import logging, os, glob
from keras.models import model_from_json
import matplotlib.pyplot as plt
import numpy as np
import re
from pathlib import Path

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

        self.NAME_FORMAT = "{}_b{}_e{}-model"

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

    def _fitmodel(self, model, X_train, y_train, X_test, y_test, batch_size, epochs, **kwargs):
        raise NotImplementedError


    #########################################################################################################

    # ========================================================================================================
    # These are handy default implementation. You can override these if needed.
    # ========================================================================================================
    
    def _model_name_from_parameters(self, batch_size, epochs, **kwargs):
        """
        Use to create a name for saving model.
        Note that this name should be able to uniquely
        identify each set of parameters used to generate
        the model. 
        
        E.g. ResNet18_b100_e10-model for ResNet18 batch_size: 100, epochs: 10

        DEFAULT:
        -------- 
        return the experiment name in lower case + "_b<batch_size>_e<epochs>-model"
        """
        name = self.INSTANCE_NAME.replace(' ', '_').lower()
        return self.NAME_FORMAT.format(name, batch_size, epochs)

    def __try_load_for_continuation(self, batch_size, epochs, **kwargs):
        """
        This method try to load the most recent model, if exist,
        that can be continued from for this parameters.

        For example, if kwargs = [batch_size: 100, epochs: 10]
        and we already have a saved model from [batch_size: 100, epochs: 7]
        then that model will be loaded and return along with new parameters
        [batch_size: 100, epochs: 3], where epochs = 3 because 10 - 7 = 3.
        That is, we need 3 more epochs to reach overall epochs of 10.

        The default implementation will only for default name format ending with
        "_b<batch_size>_e<epochs>-model"

        -----
        The method must return

        (model, batch_size, epochs, new_kwargs)

        model - the loaded model,
        batch_size, epochs - new batch_size and epochs
        new_kwargs - kwargs needed for continuation

        """

        existing_models = listdir_nohidden(self.MODEL_DIRECTORY)

        # make a regex that extract all epochs given the number of
        # batch size
        regex = self.NAME_FORMAT.format("", batch_size, "([0-9]*)")
        matcher = re.compile(regex)

        available_epochs = [
            int(matcher.search(x).group(1)) 
            for x in existing_models 
            if matcher.search(x) is not None
        ]

        self.general_logger.debug(available_epochs)

        if len(available_epochs) <= 0:
            self.general_logger.info("NO CONTINUABLE MODEL AVAILABLE.")
            return (None, batch_size, epochs, kwargs)

        _the_epochs = max(available_epochs)
        self.general_logger.debug("MAX EPOCHS: {}".format(_the_epochs))
        if _the_epochs > epochs:
            available_epochs = available_epochs.sort()
            _the_epochs = -1
            for ep in available_epochs:
                if ep >= epochs:
                    break
                if ep > _the_epochs:
                    _the_epochs = ep
            
            if _the_epochs == -1:
                return (None, batch_size, epochs, kwargs)

        model_name_to_load = self._model_name_from_parameters(batch_size, _the_epochs, **kwargs)
        loaded_model = self.loadmodel_from_name(model_name_to_load)

        self.general_logger.info("Load {} from disk to continue up to {}".format(model_name_to_load, epochs))

        return (loaded_model, batch_size, epochs - _the_epochs, kwargs)

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
        with open(directory + name+'.json', 'w') as json_file:
            json_file.write(model_json)

        # serialize weights to HDF5
        model.save_weights(directory + name+'.h5')

        # log
        self.general_logger.info("MODEL: {} SAVED TO {}.".format(name, directory))
    
    def loadmodel_from_name(self, name, directory=None, **kwargs):
        """
        Utility method to call loadmodel() using only name and directory.
        """

        if directory is None:
            directory = self.MODEL_DIRECTORY

        # make sure directory path ended with the last / 
        if not directory.endswith('/'):
            directory = directory + '/'

        base_model_dir = directory + name
        json_file = base_model_dir+'.json'
        weights_file = base_model_dir+'.h5'

        return self.loadmodel(json_file, weights_file, *kwargs)

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

    def run(self, dataset, test_samples, batch_size, epochs, allow_continuation=True, force_recreate=False, **kwargs):
        """
        Run the experiment using the given parameters.

        The parameters (kwargs) for this iteration will be given by the
        testbench in the form of a keyword-dictionary.

        dataset must be in the form of (X_train, y_train, X_test, y_test)
        """

        # You need to make sure that model name is generated
        # before loading for continuation otherwise the name
        # will be using an altered epochs!
        model_name = self._model_name_from_parameters(batch_size, epochs, **kwargs)

        if not force_recreate:
            directory = self.MODEL_DIRECTORY
            # make sure directory path ended with the last / 
            if not directory.endswith('/'):
                directory = directory + '/'

            # if model already exist, skip
            my_file = Path(directory+model_name)
            if my_file.exists():
                self.general_logger.info("MODEL ALREADY EXIST, SKIPPING...")
                return

            model = None

        if allow_continuation and not force_recreate:
            
            #try to load the any existing model that can be continue from
            (model, new_batch_size, new_epochs, new_kwargs) = self.__try_load_for_continuation(batch_size, epochs, **kwargs)

        if model is not None:
            # If able to load model, set kwargs to the new_kwargs for continuation.
            # Refer to comments in the __try_load_for_continuation() for explanation.
            kwargs = new_kwargs
            batch_size = new_batch_size
            epochs = new_epochs
        else:
            # create the model from scratch
            self.general_logger.info("LOAD FOR CONTINUATION FAILED. CREATING FROM SCRATCH...")
            model = self.__internal_createmodel(dataset, batch_size, epochs, **kwargs)

        # (continue) fitting model
        self.__internal_fitmodel(model, dataset, batch_size, epochs, **kwargs)
        
        # save this model
        self.savemodel(model, model_name, **kwargs)

        # evaluate model
        classes_acc = self.evaluate(model, test_samples, **kwargs)
        self.result_logger.info("CLASSES ACC for model {}: {}".format(model_name, classes_acc))

        # generate and save plot
        self._generate_bar_char_img(classes_acc, model_name)