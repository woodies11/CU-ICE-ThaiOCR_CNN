import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2

def readimageinput(img_path, preview=False, invert=False, size=None):
	# read image as black and white, Int mode (0-255)
	img = Image.open(img_path).convert('L')

	# resize image if size is specified
	if size is not None:
		img = img.resize(size=size)

	# convert to numpy array
	img = np.array(img)

	# binarisation
	ret, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

	# mnist dataset is white on black, this option invert the input
	# if needed to match our trained set
	if invert:
		img = 1 - img/255

	# if preview, the program will be block until the
	# plot window is closed
	if preview:
		plt.imshow(img, cmap=plt.get_cmap('gray'))

	img_width = img.shape[0]
	img_height = img.shape[1]

	# reshape the numpy array to the same format as our model
	img = img.reshape(1, 1, img_width, img_height).astype('float32')


	return img
	
if __name__ == '__main__':
	readimageinput('th_samples/ก/im2_1.jpg', True, False, (128,128))
	plt.show()
