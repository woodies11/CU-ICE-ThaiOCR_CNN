import logging, os
from experiments.mnistnet import MNISTNET
import traindata
import imageutil

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

    # load common data for speed

    # load dataset
    dataset = traindata.load_image_data()

    # construct test_sample array
    classes = [chr(i) for i in range(ord('ก'), ord('ฮ')+1)]
    test_samples = {c:[] for c in classes}

    SAMPLE_PATH = 'th_samples'
    paths = os.listdir(SAMPLE_PATH)
    
    for character in paths:
        # ignore system files
        if(character.startswith('.')):
            continue	

        img_paths = os.listdir(SAMPLE_PATH+"/"+character)

        for img_name in img_paths:
            # ignore system files
            if(img_name.startswith('.')):
                continue	

            img = imageutil.readimageinput(SAMPLE_PATH+"/"+character+'/'+img_name, preview=preview, invert=False, size=(128,128))
            test_samples[character].append(img)


    for exp in experiments:
        # Wrap everthing in try catch so our test
        # will always continue.
        # Check the log file for any errors.
        try:
            experiment = exp()
            general_logger.info("Starting experiment {}...".format(experiment.__name__))

            # Start each experiment:
            experiment.run(dataset, test_samples, 100, 1)

        except Exception as e:
            # send to log that something went wrong
            general_logger.error("Oops! something went wrong: {}: {}".format(e.__class__, str(e)))



if __name__ == "__main__":
    run()