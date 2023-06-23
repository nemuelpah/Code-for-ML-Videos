from PIL import Image
from keras.applications.mobilenet import preprocess_input
import numpy as np
import keras
from keras.models import load_model

def proses(file):
    model_baru=load_model('myNewModel.h5')
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