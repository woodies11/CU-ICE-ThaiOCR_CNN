import logging
from keras.models import model_from_json

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

    result_logger = setup_logger(EXPERIMENT_NAME, "../logs/results.txt")
    general_logger = setup_logger(EXPERIMENT_NAME, "../logs/experiments.txt")

    @staticmethod
    def run(dataset, **kwargs):
        """
        Run the experiment using the given parameters.

        The parameters (kwargs) for this iteration will be given by the
        testbench in the form of a keyword-dictionary.
        """

        # run setup to do any preprocessing
        (_dataset, _kwargs) = Experiment.__internal_setup(dataset, **kwargs)

        # create the model
        model = Experiment.__internal_createmodel(_dataset, **_kwargs)
        
        # save this model
        Experiment.savemodel(model, **kwargs)

    # ====================================================
        
    @staticmethod
    def _model_from_json(json, **kwargs):
        raise NotImplementedError

    # ====================================================

    
    @staticmethod
    def savemodel(model, name, directory, **kwargs):
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
    def loadmodel(json_file, weights_file, **kwargs):
        """
        Load model from json and h5 file.
        This method DOES NOT construct and compile model by itself.
        It will load the JSON file and pass it to _model_from_json()
        in order to load the model. This is done because each 
        """

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
    def evaluate(self, **kwargs):
        raise NotImplementedError

    # ====================================================
    # These are internal method called by run().
    # They are for implementing preprocess such as
    # logging, etc.
    # YOU SHOULD NOT OVERRIDE THESE!
    
    @staticmethod
    def __internal_setup(dataset, **kwargs):
        return Experiment._setup(dataset, **kwargs)
    
    @staticmethod
    def __internal_createmodel(dataset, **kwargs):
        return Experiment._createmodel(dataset, **kwargs)

    # -----------------------------------

    # ====================================================
    # These are chained call by the internal methods.
    # You should override these to implement your own
    # function but SHOULD NOT CALL THEM YOURSELF.
    # Let the internal methods call them.

    @staticmethod
    def _setup(dataset, **kwargs):
        """
        This method is run before each experiment.
        Do your set up here.
        """
        return dataset, kwargs

    @staticmethod
    def _createmodel(dataset, **kwargs):
        raise NotImplementedError

    @staticmethod
    def _model_name_from_parameters(**kwargs):
        raise NotImplementedError
    


