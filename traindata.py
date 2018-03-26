import numpy as np
import os
import glob
from PIL import Image
import h5py

DATA_FILE_NAME = "TRAIN_DATA_ARRAY"

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