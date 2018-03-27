import numpy as np
from PIL import Image, ImageChops
import matplotlib.pyplot as plt
import cv2

def trim(im):
    bg = Image.new(im.mode, im.size, im.getpixel((0,0)))
    diff = ImageChops.difference(im, bg)
    diff = ImageChops.add(diff, diff, 2.0, -100)
    bbox = diff.getbbox()
    if bbox:
        return im.crop(bbox)

def readimageinput(img_path, preview=False, invert=False, size=None, as_Image=False):

	# read image as black and white
	img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

	# fall back option for Windows since
	# cv2 doesn't work with Unicode path
	if img is None:
		img = Image.open(img_path).convert('L')
		img = np.array(img)

	# binarisation
	ret, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)


	# Change into PILLOW image to do trim and padding
	img = Image.fromarray(img)
	_img = trim(img)

	if _img is None:
		print(img_path + "is empty...")
	else:
		img = _img

	# get image size and find the longer dimension
	# we will pad in the other dimension to make 
	# a perfect square
	x,y = img.size
	max_size = max(x,y)

	# add padding border
	# max_size = int(max_size*1)

	# create a blank square image
	sqcanvas = Image.new('L', (max_size, max_size), (255))

	# since paste() use top left corner as an anchor point,
	# we want to calculate the position which will center
	# our image
	img_x = int((max_size - x)/2)
	img_y = int((max_size - y)/2)
	sqcanvas.paste(img, (img_x, img_y))
	
	img = sqcanvas


	img = np.array(img)

	# mnist dataset is white on black, this option invert the input
	# if needed to match our trained set
	if invert:
		img = 1 - img/255

	# resize image if size is specified
	if size is not None:
		img = cv2.resize(img, size)

	kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
	img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)

	# if preview, the program will be block until the
	# plot window is closed
	if preview:
		plt.imshow(img, cmap=plt.get_cmap('gray'))

	if as_Image:
		return Image.fromarray(img)

	img_width = img.shape[0]
	img_height = img.shape[1]

	# reshape the numpy array to the same format as our model
	img = img.reshape(1, 1, img_width, img_height).astype('float32')


	return img
	
if __name__ == '__main__':
	readimageinput('th_samples/ค/3-164-A4-KHO KHWAI-43.png', True, False, (128,128))
	plt.show()
