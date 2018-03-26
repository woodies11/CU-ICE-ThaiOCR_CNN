import logging, os
from experiments.mnistnet import MNISTNET
import traindata
import imageutil
import traceback
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer

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
    # percentage of data to keep as test set
    test_size = 0.20
    # number of set to load, load fewer during code testing
    # None for load all
    num_of_set = None

    # fix random
    np_seed = 7
    sklearn_seed = 42
    np.random.seed(np_seed)

    # load dataset
    X_set, _y_set = traindata.load_image_data()
    # convert our label to one-hot-encoded matrix
    encoder = LabelBinarizer()
    encoder.fit(_y_set)
    classes = encoder.classes_
    y_set = encoder.transform(_y_set)

    # partition data into test set and train set
    X_train, X_test, y_train, y_test = train_test_split(
        X_set, 
        y_set, 
        test_size=test_size,
        random_state=sklearn_seed
    )

    img_width = X_set.shape[1]
    img_height = X_set.shape[2]

    # reshape to be [samples][layer (black)][x][y]
    X_train = X_train.reshape(X_train.shape[0], 1, img_width, img_height).astype('float32')
    X_test = X_test.reshape(X_test.shape[0], 1, img_width, img_height).astype('float32')
    # normalize inputs from 0-255 to 0-1
    X_train = X_train / 255
    X_test = X_test / 255

    dataset = (X_train, y_train, X_test, y_test)

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

            img = imageutil.readimageinput(SAMPLE_PATH+"/"+character+'/'+img_name, preview=False, invert=False, size=(128,128))
            test_samples[character].append(img)


    for exp in experiments:
        # Wrap everthing in try catch so our test
        # will always continue.
        # Check the log file for any errors.
        try:
            experiment = exp()
            general_logger.info("Starting experiment {}...".format(experiment.EXPERIMENT_NAME))

            # Start each experiment:
            experiment.run(dataset, test_samples, 100, 1)

        except Exception as e:
            # send to log that something went wrong
            general_logger.error("Oops! something went wrong: {}: {} \n {}".format(e.__class__, str(e), traceback.format_tb(e.__traceback__)))
            traceback.print_tb(e.__traceback__)



if __name__ == "__main__":
    run()