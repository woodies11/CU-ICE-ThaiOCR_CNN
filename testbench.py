import logging, os
from experiments.mnistnet import MNISTNET
import traindata
import imageutil
import traceback
import numpy as np


experiments = [MNISTNET]

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

general_logger = setup_logger("general", "logs/general.txt", logging.DEBUG)
result_logger = setup_logger("result", "logs/result.txt", logging.INFO)


def unittestexperiment(experiment):
    pass


def run():

    # load dataset
    dataset = traindata.load_and_construct_dataset()

    # load test_sample array
    test_samples = traindata.load_test_samples()

    for exp in experiments:

        for ep in range(1, 10, 2):
            # Wrap everthing in try catch so our test
            # will always continue.
            # Check the log file for any errors.
            try:
                experiment = exp()
                general_logger.info("Starting experiment {}...".format(experiment.EXPERIMENT_NAME))

                # Start each experiment:
                experiment.run(dataset, test_samples, 100, ep)

            except Exception as e:
                # send to log that something went wrong
                general_logger.error("Oops! something went wrong: {}: {} \n {}".format(e.__class__, str(e), traceback.format_tb(e.__traceback__)))
                traceback.print_tb(e.__traceback__)



if __name__ == "__main__":
    run()