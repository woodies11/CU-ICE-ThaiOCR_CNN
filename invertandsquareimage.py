import sys, os, glob
import cv2
from PIL import Image
import numpy as np

def listdir_nohidden(path):
	directories = glob.glob(path+'/**/*', recursive=True)
	return [file for file in directories if os.path.isfile(file)]

char = sys.argv[1]
folder_path = sys.argv[2]
th_sample_path = 'th_samples'

file_paths = listdir_nohidden(folder_path)

for img_path in file_paths:
	img = Image.open(img_path).convert('L')
	img = np.array(img)
	img = 255 - img
	img = Image.fromarray(img)
	x,y = img.size
	max_size = max(x,y)
	sqcanvas = Image.new('L', (max_size, max_size), (255))
	img_x = int((max_size - x)/2)
	img_y = int((max_size - y)/2)
	sqcanvas.paste(img, (img_x, img_y))


	img_name = img_path.split('/')[-1].split('.')[0]
	sqcanvas.save(th_sample_path+"/"+char+"/"+img_name+".jpg")

