import numpy
from scipy import ndimage
from scipy import misc
import matplotlib.pyplot as plt

def readimageinput(img_path, preview=False, invert=False, bin_threshold=0.5):
	# read image as black and white, Int mode (0-255)
	img = misc.imread(img_path, flatten=True, mode='I')

	# mnist dataset is white on black, this option invert the input
	# if needed to match our trained set
	if invert:
		img = 1 - img/255

	# binarisation
	img = numpy.where(img > bin_threshold, 1, 0)

	# if preview, the program will be block until the
	# plot window is closed
	if preview:
		plt.imshow(img, cmap=plt.get_cmap('gray'))

	# reshape the numpy array to the same format as our model
	img = img.reshape(1, 1, 28, 28).astype('float32')


	return img



