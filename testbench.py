import logging, os
import traindata
import traceback
import numpy as np
import gc

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

def run(experiments, should_train_model=True, should_generate_statistic=True, forcecreate=False, debug_mode=False):

    general_logger = setup_logger("general", "logs/general.txt", logging.DEBUG)
    result_logger = setup_logger("result", "logs/result.txt", logging.INFO)

    def run_experiment(exp_class, dataset, batch_size, epochs):

        exp_name = "-b{}".format(batch_size)
        if debug_mode:
            exp_name += "-debug"

        # create a new Experiment object
        experiment = exp_class(namesuffix=exp_name)
        general_logger.info("Starting experiment {} | batch_size: {} epochs: {}".format(experiment.EXPERIMENT_NAME, batch_size, epochs))

        # Start the experiment
        experiment.run(dataset, batch_size, epochs)

    def gen_statistic(exp_class, test_samples, batch_size):

        exp_name = "-b{}".format(batch_size)
        if debug_mode:
            exp_name += "-debug"

        # create a new Experiment object
        experiment = exp_class(namesuffix=exp_name)
        general_logger.info("Start generating statistic for experiment {} | batch_size: {} epochs: {}".format(experiment.EXPERIMENT_NAME, batch_size, epochs))

        # Generate statistic
        experiment.gen_statistic(test_samples, batch_size)

    # If want to avoid loading both the dataset and the test_samples into memory at once.
    # Thus, we prefer to do two for loop and recreating the Experiments objects instead
    # so we can keep using the dataset once it is loaded in memory.

    if should_train_model:
        # load dataset
        dataset = traindata.load_and_construct_dataset(small_test_set=debug_mode, forcecreate=forcecreate)

        for exp in experiments:

            # Wrap everthing in try catch so our test
            # will always continue.
            # Check the log file for any errors.
            try:
                exp_class = exp["experiment"]
                batch_size = exp["batch_size"]
                epochs = exp["epochs"]

                if type(batch_size) is range:
                    for b in batch_size:
                        run_experiment(exp_class, dataset, b, epochs)
                else:
                    run_experiment(exp_class, dataset, batch_size, epochs)
            except Exception as e:
                # send to log that something went wrong
                general_logger.error("Oops! something went wrong: {}: {} \n {}".format(e.__class__, str(e), traceback.format_tb(e.__traceback__)))
                traceback.print_tb(e.__traceback__)

        # free up memory
        dataset = None
        gc.collect()


    if should_generate_statistic:
        # load test_sample array
        test_samples = traindata.load_test_samples()

        for exp in experiments:

            # Wrap everthing in try catch so our test
            # will always continue.
            # Check the log file for any errors.
            try:
                exp_class = exp["experiment"]
                batch_size = exp["batch_size"]
                epochs = exp["epochs"]

                if type(batch_size) is range:
                    for b in batch_size:
                        gen_statistic(exp_class, test_samples, b)
                else:
                    gen_statistic(exp_class, test_samples, batch_size)

            except Exception as e:
                # send to log that something went wrong
                general_logger.error("Oops! something went wrong: {}: {} \n {}".format(e.__class__, str(e), traceback.format_tb(e.__traceback__)))
                traceback.print_tb(e.__traceback__)
        
        # free up memory
        test_samples = None
        gc.collect()

if __name__ == "__main__":
    import argparse

    # parse argument
    parser = argparse.ArgumentParser(description='Whether to force recreate model.')
    parser.add_argument(
        '-f', 
        '--force_recreate', 
        action='store_true', 
        help='use this argument to force model recreation.'
    )

    parser.add_argument(
        '-t', 
        '--train',
        action='store_true', 
        help='Enable experiment models training (use -s to only generate statistic from existing model). If neither -t or -s is specified, both will be enabled.'
    )

    parser.add_argument(
        '-s', 
        '--statistic',
        action='store_true', 
        help='Enable statistical model generation from the generated experiment models (use -t to train model only). If neither -t or -s is specified, both will be enabled.'
    )

    parser.add_argument(
        '-d', 
        '--debug_mode',
        action='store_true', 
        help='Enable debug mode, a very small training set will be loaded instead of the full version.'
    )

    args = parser.parse_args()

    debug_mode = args.debug_mode
    forcecreate = args.force_recreate

    _train = args.train
    _stat = args.statistic

    # If neither -train or -statistic is specified, we default to run both.

    # train stat should_train_model
    #   0   0       1 - T'S' | (T  + S )
    #   0   1       0 - T'S  | (T  + S')
    #   1   0       1 - TS'  | (T' + S )
    #   1   1       1 - TS   | (T' + S')
    # SoP => T'S' + TS' + TS
    # PoS => T + S'
    should_train_model = _train or not _stat 

    # train stat should_generate_statistic
    #   0   0       1 - T'S' | (T  + S )
    #   0   1       1 - T'S  | (T  + S')
    #   1   0       0 - TS'  | (T' + S )
    #   1   1       1 - TS   | (T' + S')
    # SoP => T'S' + T'S + TS
    # PoS => T' + S
    should_generate_statistic = _stat or not _train

    import testbench_config
    experiments = testbench_config.experiments

    run(experiments, should_train_model, should_generate_statistic, forcecreate, debug_mode)