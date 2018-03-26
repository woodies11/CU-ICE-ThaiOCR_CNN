import numpy as np
import os
import glob
from PIL import Image
import h5py
import imageutil
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer

DATA_FILE_NAME = "TRAIN_DATA_ARRAY"

def load_test_samples():
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

    return test_samples

def load_and_construct_dataset(max_set=None, test_size = 0.20, sklearn_seed = 42, np_seed = 7):
	# fix random
	np.random.seed(np_seed)

	# load dataset
	X_set, _y_set = load_image_data(max_set)
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

	return dataset

def load_image_data(max_set=None):

    print('Generating Numpy Array...')

    def listdir_nohidden(path):
        # list files without hidden file
        return glob.glob(os.path.join(path, '*'))

    image_folder = 'image_from_font'
    set_paths = listdir_nohidden(image_folder)
    if max_set is not None:
        set_paths = set_paths[0:max_set+1]

    try:
        h5f = h5py.File(DATA_FILE_NAME+'.h5','r')
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
        
    # load images as grayscale
    X_set = np.array(
        [
            # preprocess can be done here
            np.array(Image.open(image).convert('L')) 
                for set_path in set_paths
                    for image in listdir_nohidden(set_path) 
        ]
    )
    y_set = np.array(
        [
            # '/' for mac directory, '\\' for windows directory 
            image.split('/')[-1].split('\\')[-1].split('.')[0].encode('utf-8') 
                for set_path in set_paths
                    for image in listdir_nohidden(set_path)
        ]
    )

    try:
        print("\t\tSaving numpy array for future use.")
        h5f = h5py.File(DATA_FILE_NAME+'.h5', 'w')
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