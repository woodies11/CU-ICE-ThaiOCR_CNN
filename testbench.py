import logging
from experiments import mnistnet

experiments = [mnistnet]

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

def run():
    for experiment in experiments:
        # Wrap everthing in try catch so our test
        # will always continue.
        # Check the log file for any errors.
        try:
            general_logger.info("Starting experiment {}...".format(experiment.__name__))

            
            try:
                # get experiment interation information
                iterations = experiment.iterations
            except AttributeError:
                # If no iterations is specified,
                # create one iteration with empty parameters.
                iterations = [[]] 

            # Variable iterations is in this scope after try/except.

            general_logger.info("Experiment iteration info: {}".format(iterations))

            # Start each experiment:
            

        except Exception as e:
            # send to log that something went wrong
            general_logger.error("Oops! something went wrong: {}: {}".format(e.__class__, str(e)))



if __name__ == "__main__":
    run()