from PIL import Image
from keras.applications.mobilenet import preprocess_input
import numpy as np
import keras
from keras.models import load_model

def proses(file):
    #model_baru=load_model('myNewModel.h5') This is no longer works

    json_file = open('model.json', 'r')     # use this block instead, when creating the model use JSON SAVE as below
    loaded_model_json = json_file.read()    # and copy model.json and model.h5 to this folder (instead of myNewModel)
    json_file.close()
    model_baru = model_from_json(loaded_model_json)
    # load weights into new model
    model_baru.load_weights("model.h5")

    jenis = ['Parang','Mega Mendung','Kawung']

    image = Image.open(file.file)
    image = image.convert('RGB')
    image = image.resize((224,224))
    image = np.asarray(image)
    image = np.expand_dims(image,0)
    #image = image/255
    gambar = keras.applications.mobilenet.preprocess_input(image)
    
    p = model_baru.predict(gambar)
    kelas = p.argmax(axis = 1)[0]
    label = jenis[kelas]
    conf = p[0][kelas]
    return conf, label


# from tensorflow.keras.models import Sequential, model_from_json

# model_json = modelKu.to_json()
# with open("model.json", "w") as json_file:
#     json_file.write(model_json)
# modelKu.save_weights("model.h5")

# json_file = open('model.json', 'r')
# loaded_model_json = json_file.read()
# json_file.close()
# loaded_model = model_from_json(loaded_model_json)
# loaded_model.load_weights("model.h5")