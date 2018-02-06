from PIL import Image, ImageDraw, ImageFont
import os
import glob
from fontTools.ttLib import TTFont

def generate_image(char, path, font_path, image_width=128, image_height=128):

	# create a white width*height image
	img = Image.new('RGB', (img_width, img_height), (255, 255, 255))

	d = ImageDraw.Draw(img)
	fnt = ImageFont.truetype(font_path, size=min(img_width, img_height))

	# caculate where to place image so it is centered
	text_width, text_height = d.textsize(text=char, font=fnt)
	fnt_offset_x, fnt_offset_y = fnt.getoffset(char)
	img_x = (img_width - text_width - fnt_offset_x)/2
	img_y = (img_height - text_height - fnt_offset_y)/2

	# print
	# print('text-size: {}x{}'.format(text_width, text_height))
	# print('top-left: ({},{})'.format(img_x, img_y))

	# draw text on the image, starting at (img_x, img_y) top-left
	d.text((img_x, img_y), char, font=fnt, fill=(0,0,0))

	img.save(path, 'png')

def listdir_nohidden(path):
    return glob.glob(os.path.join(path, '*'))

def char_in_font(unicode_char, font):
    for cmap in font['cmap'].tables:
        if cmap.isUnicode():
            if ord(unicode_char) in cmap.cmap:
                return True
    return False

image_store_path = "image_from_font/"
img_width = 128
img_height = 128

# get all fonts
fonts = listdir_nohidden('fonts')

for font_path in fonts:
	# get the end (final of path) and remove extension
	font_name = font_path.split('/')[-1].split('.')[0]

	# skip the font if Thai is not supported
	# this avoid invalid character messing up
	# our CNN
	# PS This assume that if ก is not supported
	# then all is not supported and vice versa
	f = TTFont(font_path)
	supported = False
	try:
		supported = char_in_font('ก', f)
	except:
		supported = False
		print('{} corrupted | '.format(font_name), end='')

	if not supported:
		print('{} not supported'.format(font_name))
		continue

	# we want to store all characters generate from
	# this font in a folder with the font's name
	directory = image_store_path+font_name+'/'

	# make new directory if not exist, otherwise
	# assume already generated, skip
	if not os.path.exists(directory):
		os.makedirs(directory)
	else:
		continue
	for i in range(ord('ก'), ord('ฮ')):
		char = chr(i)
		out_path = directory + char + '.png'
		
		generate_image(char, out_path, font_path, img_width, img_height)