import sys, os, glob
from PIL import Image
import numpy as np
import random

def listdir_nohidden(path):
	directories = glob.glob(path+'/**/*', recursive=True)
	return [file for file in directories if os.path.isfile(file)]

folder_path = sys.argv[1]
th_sample_path = 'th_samples'

file_paths = listdir_nohidden(folder_path)

# create destination folders if not already exist
for i in range(ord('ก'), ord('ฮ') + 1):
	char = chr(i)
	directory = th_sample_path + '/' + char 
	if not os.path.exists(directory):
		os.makedirs(directory)

for img_path in file_paths:

	img_name = img_path.split('/')[-1].split("\\")[-1]
	
	img_name = img_name.split('.')[0] + "-{:06d}".format(random.randint(0, 999999))

	# ignore Icon? files
	if "Icon" in img_name:
		continue

	# char = chr(ord('ก') + int(img_path.split('/')[-2]) - 1 )
	char = img_path.split('/')[-1].split("\\")[-1].split('.')[0]
	
	# open the image and convert to grayscale
	img = Image.open(img_path).convert('L')

	# convert to numpy array and invert the color
	# then convert back to PILLOW Image
	# these is probably a better solution but this
	# works for now
	img = np.array(img)
	# img = 255 - img
	img = Image.fromarray(img)

	# get image size and find the longer dimension
	# we will pad in the other dimension to make 
	# a perfect square
	x,y = img.size
	max_size = max(x,y)

	# add padding
	max_size = int(max_size*1.6)

	# create a blank square image
	sqcanvas = Image.new('L', (max_size, max_size), (255))

	# since paste() use top left corner as an anchor point,
	# we want to calculate the position which will center
	# our image
	img_x = int((max_size - x)/2)
	img_y = int((max_size - y)/2)
	sqcanvas.paste(img, (img_x, img_y))
	
	# finally, save the image to an appropriate path
	sqcanvas.save(th_sample_path+"/"+char+"/"+img_name+".jpg")

