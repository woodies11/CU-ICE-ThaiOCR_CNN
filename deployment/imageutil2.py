import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageFont
from fontTools.ttLib import TTFont
import matplotlib.pyplot as plt
import cv2

def trim(img,tol=255*0.95):
    # img is image data
    # tol  is tolerance
    mask = img<tol
    return img[np.ix_(mask.any(1),mask.any(0))]

def readimage(img_path):
    # read image as black and white
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    # fall back option for Windows since
    # cv2 doesn't work with Unicode path
    if img is None:
        img = Image.open(img_path).convert('L')
        img = np.array(img)

    return img

def binarise(img):
    # binarisation
    _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    return img

def invertimage(img):
    return 255 - img

def padtosquare(img, padding_ratio=0):
    # get image size and find the longer dimension
    # we will pad in the other dimension to make 
    # a perfect square (since img is a matrix, it shape is row x col)
    h,w = img.shape

    max_dim = max(w,h)

    # add padding border
    max_dim = int(max_dim*(1+padding_ratio))

    x_space = max_dim - w
    y_space = max_dim - h

    # since int() convertion always round down, we
    # add 1 to left or top if the needed number of
    # pixels is in odd number
    x_is_even = (x_space % 2 == 0)
    y_is_even = (y_space % 2 == 0)

    left    = int(x_space/2) + (0 if x_is_even else 1)
    right   = int(x_space/2)
    top     = int(y_space/2) + (0 if y_is_even else 1)
    bottom  = int(y_space/2)
    
    img = cv2.copyMakeBorder(img, top, bottom, left, right, borderType=cv2.BORDER_CONSTANT, value=[255, 255, 255])
    return img

def resize(img, size):
    return cv2.resize(img, size)

def morph_open(img, kernel_size):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel_size)
    img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    return img

def reshape_for_keras(img):
    """ reshape the numpy array to the same format as our model """
    return img.reshape(1, 1, img.shape[0], img.shape[1]).astype('float32')

def char_img(char, size=(70,70)):
    w = size[0]
    h = size[1]
    # create a white width*height image
    img = Image.new(mode='L', size=size, color=255)
    d = ImageDraw.Draw(img)

    # we want to find the size in pt which fit exactly in our canvas

    # we start with an estimate (the number here came from trial and error)
    size_in_pt = int(min(w,h)*1.7)

    fnt = ImageFont.truetype("Norasi.ttf", size_in_pt)
    text_width, text_height = fnt.getsize(char)
    fnt_offset_x, fnt_offset_y = fnt.getoffset(char)

    # caculate where to place image so it is centered
    img_x = (w - text_width - fnt_offset_x)/2
    img_y = (h - text_height - fnt_offset_y)/2

    # draw text on the image, starting at (img_x, img_y) top-left
    d.text((img_x, img_y), char, font=fnt, fill=0)
    return img

def chars_from_font(chars, font_path, size):

    w = size[0]
    h = size[1]

    char_imgs = []

    try:
        for char in chars:

            # create a white width*height image
            img = Image.new(mode='L', size=size, color=255)
            d = ImageDraw.Draw(img)

            # we want to find the size in pt which fit exactly in our canvas

            # we start with an estimate (the number here came from trial and error)
            size_in_pt = int(min(w,h)*1.2)

            while True:
                # we scan up until it overflow

                _fnt = ImageFont.truetype(font_path, size=size_in_pt)

                text_width, text_height = _fnt.getsize(char)
                fnt_offset_x, fnt_offset_y = _fnt.getoffset(char)

                if text_width-fnt_offset_x > w or text_height-fnt_offset_y > h:
                    break

                # the rate here can be quite fast as we will do
                # a precious scan down later
                size_in_pt += min(int(max(w,h)*0.5), 100)

            while True:
                # now keep reducing the size slowly until we reach
                # a size that do not overflow
                _fnt = ImageFont.truetype(font_path, size=size_in_pt)

                text_width, text_height = _fnt.getsize(char)
                fnt_offset_x, fnt_offset_y = _fnt.getoffset(char)

                if text_width-fnt_offset_x <= w and text_height-fnt_offset_y <= h:
                    break
                size_in_pt -= 1

            fnt = ImageFont.truetype(font_path, size=size_in_pt)
            text_width, text_height = fnt.getsize(char)
            fnt_offset_x, fnt_offset_y = fnt.getoffset(char)

            # caculate where to place image so it is centered
            img_x = (w - text_width - fnt_offset_x)/2
            img_y = (h - text_height - fnt_offset_y)/2

            # draw text on the image, starting at (img_x, img_y) top-left
            d.text((img_x, img_y), char, font=fnt, fill=0)
            
            char_imgs.append(np.array(img))

    except OSError as e:
        print("problem openning {}".format(font_path))
        return None

    # convert to a numpy array then return
    return np.array(char_imgs)

def showimage(img):
    plt.imshow(img, cmap=plt.get_cmap('gray'))
    plt.show()
    plt.close()

if __name__ == '__main__':

    # for i in range(ord('ก'), ord('ฮ')):

    #     plt.subplot(5,10,i+1-ord('ก'))

    #     img = char_from_font(chr(i), 'fonts/4711_AtNoon_Traditional.ttf', size=(56, 56))
    #     # img = binarise(img)

    #     plt.imshow(img, cmap=plt.get_cmap('gray'))

    for i in range(1,9):
        plt.subplot(2,4,i)
        img = readimage('th_samples/ก/im12_3.jpg')
        # img = binarise(img)
        img = trim(img)
        img = padtosquare(img)
        img = resize(img, (56, 56))
        img = morph_open(img, (1<<2, 1<<2))

        plt.imshow(img, cmap=plt.get_cmap('gray'))
    plt.show()
    plt.close()