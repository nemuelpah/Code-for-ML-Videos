# -*- coding: utf-8 -*-
"""GunakanMobileNetV2_3.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/15vWVxAbqIJDOLvtb9gRWmqrRK3oLlVuG
"""

import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf
from PIL import Image
from numpy import asarray

# Loading base model dari pre-trained model MobileNet V2

MobileNetV2 = tf.keras.applications.MobileNetV2(input_shape=(160,160,3), include_top=True, weights='imagenet')

gbr = Image.open('test.jpg')
gbr = gbr.convert('RGB')
gbr = gbr.resize((160,160))
gbr = asarray(gbr)
plt.imshow(gbr)

plt.imshow(gbr)

gbr=np.expand_dims(gbr, axis=0)

inputs = tf.keras.Input(shape=(160, 160, 3))
x = tf.keras.layers.Rescaling(1./127.5, offset=-1)(inputs)
outputs = MobileNetV2(x)

modelKu = tf.keras.Model(inputs, outputs)

prediksi=modelKu(gbr)

p=np.round(prediksi)

kelas=np.where(p==1)

int(kelas[1])



















import json
import ast
import urllib
urllib.request.urlretrieve("https://gist.githubusercontent.com/yrevar/942d3a0ac09ec9e5eb3a/raw/238f720ff059c1f82f368259d1ca4ffa5dd8f9f5/imagenet1000_clsidx_to_labels.txt", "labels.json")
with open("labels.json") as f:
    labels = ast.literal_eval(f.read())

labels[int(kelas[1])]











prediksi=MobileNetV2Ku(gbr)

prediksi=np.round(prediksi)

np.where(prediksi==1)









import json
import ast
import urllib
urllib.request.urlretrieve("https://gist.githubusercontent.com/yrevar/942d3a0ac09ec9e5eb3a/raw/238f720ff059c1f82f368259d1ca4ffa5dd8f9f5/imagenet1000_clsidx_to_labels.txt", "labels.json")
with open("labels.json") as f:
    labels = ast.literal_eval(f.read())

gbr = Image.open('coba3.jpg')
gbr = gbr.convert('RGB')
gbr = gbr.resize((160,160))
gbr = asarray(gbr)

plt.imshow(gbr)

gbr.shape

gbr=np.expand_dims(gbr, axis=0)

gbr.shape

preprocess_input = tf.keras.layers.Rescaling(1./127.5, offset=-1)

inputs = tf.keras.Input(shape=(160, 160, 3))
x = preprocess_input(inputs)
outputs = base_model(x)

modelSaya = tf.keras.Model(inputs, outputs)

prediksi=modelSaya(gbr)


prediksi=np.round(prediksi)
kelas=np.where(prediksi==1)
labels[int(kelas[1])]

prediksi

labels
