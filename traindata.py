import numpy as np
import os
import glob
from PIL import Image
import h5py
import imageutil2 as imgutil
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer

DATA_FILE_NAME = "TRAIN_DATA_ARRAY"
SMALL_DATA_FILE_NAME = "SMALL_DATA_ARRAY"

TEST_SAMPLES_FILE_NAME = "TEST_SAMPLES_ARRAY"
TEST_SAMPLES_PATH = 'th_samples'

SAMPLE_SIZE = (56, 56)

def listdir_nohidden(path):
	directories = glob.glob(path+'/**/*', recursive=True)
	return [file for file in directories if os.path.isfile(file)]

def apply_image_preprocess(img, size=SAMPLE_SIZE, binarise=True):
    img = imgutil.trim(img)
    img = imgutil.padtosquare(img)
    if binarise:
        img = imgutil.binarise(img)
    img = imgutil.resize(img, size)
    img = imgutil.morph_open(img, (2, 2))
    return img

def reshape_image_for_keras(img):
    return imgutil.reshape_for_keras(img)

def load_img_as_numpy(img_path, reshape=False, preprocess=True):
    img = imgutil.readimage(img_path)
    if preprocess:
        img = apply_image_preprocess(img)
    if reshape:
        img = reshape_image_for_keras(img)
    return img

# ============= TEST SAMPLES ===============

def load_test_samples(forcecreate=False):
    # construct test_sample array
    classes = [chr(i) for i in range(ord('ก'), ord('ฮ')+1) if i != ord('ฦ') and i != ord('ฤ')]
    test_samples = {c:[] for c in classes}

    paths = os.listdir(TEST_SAMPLES_PATH)
    
    for character in paths:
        # ignore system files
        if(character.startswith('.')):
            continue    

        img_paths = os.listdir(TEST_SAMPLES_PATH+"/"+character)

        for img_name in img_paths:
            # ignore system files
            if(img_name.startswith('.')):
                continue    
            img_path = os.path.join(TEST_SAMPLES_PATH, character, img_name)
            img = load_img_as_numpy(img_path, reshape=True, preprocess=True)
            test_samples[character].append(img)

    return test_samples

# ============ TRAIN SAMPLE ===============

def load_trainset(max_set=None, test_size = 0.20, sklearn_seed = 42, np_seed = 7, small_test_set=False, forcecreate=False):
    
    test_set = SMALL_DATA_FILE_NAME if small_test_set else DATA_FILE_NAME
    max_set = 20 if small_test_set else None

    # fix random
    np.random.seed(np_seed)



def load_and_construct_dataset(max_set=None, test_size = 0.20, sklearn_seed = 42, np_seed = 7, small_test_set=False, forcecreate=False):

    test_set = SMALL_DATA_FILE_NAME if small_test_set else DATA_FILE_NAME
    max_set = 100 if small_test_set else None

    # fix random
    np.random.seed(np_seed)

    # load dataset
    X_set, _y_set = load_image_data(max_set=max_set, filename=test_set, forcecreate=forcecreate)
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

    # normalize inputs from 0-255 to 0-1
    X_train = X_train / 255
    X_test = X_test / 255

    dataset = (X_train, y_train, X_test, y_test)

    return dataset

def load_image_data(max_set=None, filename=DATA_FILE_NAME, forcecreate=False, size=SAMPLE_SIZE):

    print('Loading Numpy Array...')
    
    if not forcecreate:
        try:
            h5f = h5py.File(filename+'.h5','r')
            X_set = h5f['X_set'][:]
            y_set = h5f['y_set'][:]
            h5f.close()
            print("\tData loaded from h5 file.")

            print('\t{} data loaded '.format(X_set.shape[0]))
            return X_set, y_set
        except OSError:
            print("\tNo h5 file found. Recreating Numpy Array...")
        except KeyError:
            print("\tCorrupted h5 file. Recreating Numpy Array...")
    else:
        print("\tforcecreate is set to True. Recreating Numpy Array...")

    fonts_path = './fonts'
    fonts = listdir_nohidden(fonts_path)

    if max_set is not None:
        fonts = fonts[0:max_set]

    classes = [chr(i) for i in range(ord('ก'), ord('ฮ')+1) if i != ord('ฦ') and i != ord('ฤ')]

    X_set = np.array([
        imgutil.chars_from_font(classes, font, size) 
        for font in fonts
    ])

    y_set = np.array([y for _ in fonts for y in classes])

    img_width = X_set.shape[2]
    img_height = X_set.shape[3]

    X_set = X_set.reshape(X_set.shape[0]*X_set.shape[1], 1, img_width, img_height)

    return X_set, y_set

    try:
        print("\t\tSaving numpy array for future use.")
        h5f = h5py.File(filename+'.h5', 'w')
        h5f.create_dataset('X_set', data=X_set)
        dt = h5py.special_dtype(vlen=str)
        h5f.create_dataset('y_set', data=y_set, dtype=dt)
        h5f.close()
        print("\t\tNumpy array saved.")
    except TypeError as e:
        print("\t\tFAILED TO SAVE MODEL!!")
        print(e)
        print("\t\tCONTINUE WITHOUT SAVING.")

    print('\t\t{} data loaded '.format(X_set.shape[0]))

    return X_set, y_set

if __name__ == "__main__":
    load_image_data(10, forcecreate=True)