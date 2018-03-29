import sys, os, glob, shutil

def listdir_nohidden(path):
	directories = glob.glob(path+'/**/*', recursive=True)
	return [file for file in directories if os.path.isfile(file)]

folder_path = 'char_seg_test/segmented_img' 

files_path = listdir_nohidden(folder_path)

for i in range(ord('ก'), ord('ฮ') + 1):
	char = chr(i)
	directory = folder_path + '/' + char 
	if not os.path.exists(directory):
		os.makedirs(directory)

for fp in files_path:
	filename = fp.split('/')[-1].split('\\')[-1] 
	foldername = fp.split('\\')[-2] #windows 
  	#foldername = fp.split('/')[-2] #unix 
	char = filename[0]
	dest = folder_path + '/' + char + '/' + foldername + '.png' 
	shutil.move(fp, dest)