import numpy as np
import os
import glob
from PIL import Image


def load_image_data(max_set=None):

	print('loading data into Numpy Array')

	def listdir_nohidden(path):
		# list files without hidden file
	    return glob.glob(os.path.join(path, '*'))

	image_folder = 'image_from_font'
	set_paths = listdir_nohidden(image_folder)
	if max_set is not None:
		set_paths = set_paths[0:max_set+1]

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
			image.split('/')[-1].split('.')[0]  
				for set_path in set_paths
					for image in listdir_nohidden(set_path)
		]
	)

	print('{} data loaded '.format(X_set.shape[0]))

	return X_set, y_set