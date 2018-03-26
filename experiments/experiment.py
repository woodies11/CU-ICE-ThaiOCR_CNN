import logging
from keras.models import model_from_json
import matplotlib.pyplot as plt

def setup_logger(name, log_file, level=logging.INFO, format='%(levelname)-7s|%(module)s|%(asctime)s: %(message)s'):
    """Function setup as many loggers as you want"""

    formatter = logging.Formatter(format)
    handler = logging.FileHandler(log_file)        
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)

    # also print to console
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger

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

        self.result_logger = setup_logger(self.INSTANCE_NAME, "./experiments/logs/results.txt")
        self.general_logger = setup_logger(self.INSTANCE_NAME, "./experiments/logs/experiments.txt")

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

    def _model_name_from_parameters(self, batch_size, epochs, **kwargs):
        """
        Use to create a name for saving model.
        Note that this name should be able to uniquely
        identify each set of parameters used to generate
        the model. 
        
        E.g. ResNet18_b100_e10 for ResNet18 batch_size: 100, epochs: 10
        """
        raise NotImplementedError

    def __try_load_for_continuation(self, batch_size, epochs, **kwargs):
        """
        This method try to load the most recent model, if exist,
        that can be continued from for this parameters.

        For example, if kwargs = [batch_size: 100, epochs: 10]
        and we already have a saved model from [batch_size: 100, epochs: 7]
        then that model will be loaded and return along with new parameters
        [batch_size: 100, epochs: 3], where epochs = 3 because 10 - 7 = 3.
        That is, we need 3 more epochs to reach overall epochs of 10.

        Since each network may need different parameter, the default implemnetation
        return None so the model will always be created from scratch.

        -----
        The method must return

        (model, batch_size, epochs, new_kwargs)

        model - the loaded model,
        batch_size, epochs - new batch_size and epochs
        new_kwargs - kwargs needed for continuation

        """
        return (None, batch_size, epochs, kwargs)

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

    def _createmodel(self, X_train, y_train, X_test, y_test, batch_size, epochs, **kwargs):
        raise NotImplementedError

    def _fitmodel(self, model, X_train, y_train, X_test, y_test, batch_size, epochs, **kwargs):
        raise NotImplementedError


    #########################################################################################################

    # ========================================================================================================
    # These are handy default implementation. You can override these if needed.
    # ========================================================================================================
    
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

    def run(self, dataset, test_samples, batch_size, epochs, allow_continuation=True, **kwargs):
        """
        Run the experiment using the given parameters.

        The parameters (kwargs) for this iteration will be given by the
        testbench in the form of a keyword-dictionary.

        dataset must be in the form of (X_train, y_train, X_test, y_test)
        """

        model_name = self._model_name_from_parameters(batch_size, epochs, **kwargs)

        model = None

        if allow_continuation:
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
            model = self.__internal_createmodel(dataset, batch_size, epochs, **kwargs)

        # (continue) fitting model
        self.__internal_fitmodel(model, dataset, batch_size, epochs, **kwargs)
        
        # save this model
        self.savemodel(model, model_name, **kwargs)

        # evaluate model
        classes_acc = self.evaluate(model, test_samples, *kwargs)
        self.result_logger.info("CLASSES ACC for model {}: {}".format(model_name, classes_acc))

        # generate and save plot
        self._generate_bar_char_img(classes_acc, model_name)


