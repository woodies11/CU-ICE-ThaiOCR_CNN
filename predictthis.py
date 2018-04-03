from tkinter import Tk, Button
from tkinter.filedialog import askopenfilename
import imageutil2 as imgutil
import keras
import keras_resnet
from keras.models import model_from_json
from tkinter import messagebox
import matplotlib.pyplot as plt
from PIL import Image, ImageFont, ImageDraw, ImageEnhance

def _model_from_json(json, **kwargs):
        return model_from_json(json, custom_objects=keras_resnet.custom_objects)

def _compilemodel(model, **kwargs):
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
    
def loadmodel(json_file, weights_file, **kwargs):
        """
        Load model from json and h5 file.
        This method DOES NOT construct and compile model by itself.
        It will load the JSON file and pass it to _model_from_json()
        in order to load the model. This is done because each 
        """

        # add in file extension if not already added
        if not json_file.endswith('.json'):
            json_file += '.json'

        if not weights_file.endswith('.h5'):
            weights_file += '.h5'

        # load json
        json_file = open(json_file, 'r')
        loaded_model_json = json_file.read()
        json_file.close()

        # use experiment specific method to construct
        # the model from JSON
        model = _model_from_json(loaded_model_json, **kwargs)

        model = _compilemodel(model, **kwargs)

        # load weights into new model
        model.load_weights(weights_file)

        # return the ready-to-use model
        return model

def predict(model, test_sample, classes, **kwargs):
    y_prob = model.predict(test_sample)
    pred = y_prob.argmax(axis=-1)
    pred_class = classes[pred[0]]
    return pred_class

def do_prediction():

    inputimg = askopenfilename()
    img = imgutil.readimage(inputimg)

    plt.subplot(1,3,1)
    plt.imshow(img, cmap=plt.get_cmap('gray'))

    img = imgutil.binarise(img)
    img = imgutil.trim(img)
    img = imgutil.padtosquare(img)
    img = imgutil.resize(img, (70, 70))

    plt.subplot(1,3,2)
    plt.imshow(img, cmap=plt.get_cmap('gray'))

    img = imgutil.reshape_for_keras(img)
    pred_class = predict(model, img, CLASSES)

    plt.subplot(1,3,3)
    pred_class_preview = imgutil.char_img(pred_class)
    plt.imshow(pred_class_preview, cmap=plt.get_cmap('gray'))

    plt.show()
    plt.close()

CLASSES = [chr(i) for i in range(ord('ก'), ord('ฮ')+1)]

window = Tk()
button = Button(text="predict", command=do_prediction)
button.pack()

modelpath = askopenfilename()
weightpath = askopenfilename()

print("Loading model...")

model = loadmodel(modelpath, weightpath)

window.mainloop()