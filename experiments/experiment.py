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

class Experiment(object):

    EXPERIMENT_NAME = "GENERIC EXPERIMENT"
    EXPERIMENT_DESCRIPTION = """
    A base class template for an experiment.
    """

    # dafault directory to save models
    MODEL_DIRECTORY = "./experiments/models/{}/".format(EXPERIMENT_NAME)
    RESULT_STATISTIC_DIRECTORY = "./experiments/results/{}/".format(EXPERIMENT_NAME)

    result_logger = setup_logger(EXPERIMENT_NAME, "./experiments/logs/results.txt")
    general_logger = setup_logger(EXPERIMENT_NAME, "./experiments/logs/experiments.txt")

    @staticmethod
    def run(dataset, test_samples, allow_continuation=True, **kwargs):
        """
        Run the experiment using the given parameters.

        The parameters (kwargs) for this iteration will be given by the
        testbench in the form of a keyword-dictionary.

        dataset must be in the form of (X_train, y_train, X_test, y_test)
        """

        model_name = _model_name_from_parameters(**kwargs)

        model = None

        if allow_continuation:
            #try to load the any existing model that can be continue from
            (model, new_kwargs) = Experiment.__try_load_for_continuation(**kwargs)

        if model is not None:
            # If able to load model, set kwargs to the new_kwargs for continuation.
            # Refer to comments in the __try_load_for_continuation() for explanation.
            kwargs = new_kwargs
        else:
            # create the model from scratch
            model = Experiment.__internal_createmodel(dataset, **kwargs)

        # (continue) fitting model
        Experiment.__internal_fitmodel(dataset, model, **kwargs)
        
        # save this model
        Experiment.savemodel(model, model_name, kwargs=**kwargs)

        # evaluate model
        classes_acc = Experiment.evaluate(model, test_samples, *kwargs)
        result_logger.info("CLASSES ACC for model {}: {}".format(model_name, classes_acc))

        # generate and save plot
        Experiment._generate_bar_char_img(classes_acc, model_name)

    # ====================================================
        
    @staticmethod
    def _model_from_json(json, **kwargs):
        raise NotImplementedError

    # ====================================================
    
    @staticmethod
    def savemodel(model, name, directory=Experiment.MODEL_DIRECTORY, **kwargs):
        """
        Save model as <name>.json and <name>.h5 to directory.

        model - the model to be saved
        name - will be used as file name
        directory - is the path to folder to save the model to

        Generally, model are save in exactly the same manner
        so there shouldn't be a need to override this.
        """

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
        Experiment.general_logger.info("MODEL: {} SAVED TO {}.".format(name, directory))
    
    @staticmethod
    def loadmodel_from_name(name, directory=Experiment.MODEL_DIRECTORY, **kwargs):
        """
        Utility method to call loadmodel() using only name and directory.
        """

        # make sure directory path ended with the last / 
        if not directory.endswith('/'):
            directory = directory + '/'

        base_model_dir = directory + name
        json_file = base_model_dir+'.json'
        weights_file = base_model_dir+'.h5'

        return Experiment.loadmodel(json_file, weights_file, *kwargs)

    @staticmethod
    def loadmodel(json_file, weights_file, **kwargs):
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
        model = Experiment._model_from_json(loaded_model_json, **kwargs)

        # load weights into new model
        model.load_weights(weights_file)

        # return the ready-to-use model
        return model
    
    @staticmethod
    def predict(model, **kwargs):
        raise NotImplementedError

    @staticmethod
    def evaluate(model, test_samples, **kwargs):
        """
        Should return classes_acc in the format:
        [
            'ก': 0.90,
            'ข': 0.75,
            .
            .
            .
        ]

        where the key is the class and the value is the accuracy from 100%
        """
        raise NotImplementedError

    # ====================================================
    # These are internal method called by run().
    # They are for implementing preprocess such as
    # logging, etc.
    # YOU SHOULD NOT OVERRIDE THESE!
    
    @staticmethod
    def __internal_createmodel(dataset, **kwargs):
        # note that dataset are in from of (X_train, y_train, X_test, y_test)
        # we expand it into X_train, y_train, X_test, y_test for our function
        # for easier use
        return Experiment._createmodel(*dataset, **kwargs)

    @staticmethod
    def __internal_fitmodel(dataset, model, **kwargs):
        # note that dataset are in from of (X_train, y_train, X_test, y_test)
        # we expand it into X_train, y_train, X_test, y_test for our function
        # for easier use
        return Experiment._fitmodel(*dataset, model, **kwargs)

    # -----------------------------------

    @staticmethod
    def __try_load_for_continuation(**kwargs):
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

        (model, **new_kwargs)

        model - the loaded model,
        **new_kwargs - kwargs needed for continuation

        """
        return (None, **kwargs)

    # ====================================================
    # These are chained call by the internal methods.
    # You should override these to implement your own
    # function but SHOULD NOT CALL THEM YOURSELF.
    # Let the internal methods call them.

    @staticmethod
    def _createmodel(X_train, y_train, X_test, y_test, **kwargs):
        raise NotImplementedError

    @staticmethod
    def _fitmodel(X_train, y_train, X_test, y_test, model, **kwargs):
        raise NotImplementedError

    @staticmethod
    def _model_name_from_parameters(**kwargs):
        """
        Use to create a name for saving model.
        Note that this name should be able to uniquely
        identify each set of parameters used to generate
        the model. 
        
        E.g. ResNet18_b100_e10 for ResNet18 batch_size: 100, epochs: 10
        """
        raise NotImplementedError
    
    @staticmethod
    def _generate_bar_char_img(classes_acc, name, directory=Experiment.RESULT_STATISTIC_DIRECTORY, title=name):
        # make sure directory path ended with the last / 
        if not directory.endswith('/'):
            directory = directory + '/'
        
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
        Experiment.general_logger.info("BAR CHART for {} saved to {}.".format(name, save_path))


