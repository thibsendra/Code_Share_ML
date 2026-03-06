# -*- coding: utf-8 -*-

# Import
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from tensorflow.keras.models import load_model
 
# Charger le modèle sauvegardé 
name = 'TNN_BM_eddify_2' 
model = load_model("./models/"+name+".keras", compile=False, safe_mode=False)
model.summary() 

# Conversion en tensorflow lite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
with open('mon_modele.tflite', 'wb') as f:
    f.write(tflite_model)