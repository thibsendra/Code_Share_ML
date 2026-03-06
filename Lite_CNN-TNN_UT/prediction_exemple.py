# -*- coding: utf-8 -*-
#%%
# Import
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import numpy as np

from scipy.io import loadmat
from scipy.io import savemat
from scipy.signal import find_peaks

import prediction as pred


#%%

# Charger le modèle sauvegardé
interpreter = tf.lite.Interpreter(model_path='./models/mon_modele.tflite')
interpreter.allocate_tensors()

# Import des datasets
A_scans = loadmat("./datasets/Dataset_real_signal_mean_test_indus.mat") # Mettre votre dataset 
A_scans = np.array(A_scans['signal_test']) 
N = len(A_scans[0])
M = len(A_scans)
A_scans = A_scans.reshape(M,N,1)

# Données de l'acquisition
filtre = True   # Si les données sont déjà filtrés : True, sinon : False
F_ech = 100e6   # Fréquence d'échantillonnage
Cl=5932.07      # Vitesse longitudinale 
dist_min = 1e-3 # 2 * distance minimale
d_nom = 0e-3    # distance nominale du bloc, ne pas définir si non connue ou si l'épaisseur est fortement variable
F = 5e6         # Fréquence de la sonde

dist_pre = []
predict = np.zeros([M, 2000])

###### Estimation des distances et comparaison ######
for i in range(0,M):
    print("Iteration : ", i+1)
    d_min, predictions = pred.prediction(interpreter, A_scans[i], F_ech, F, Cl, dist_min, filtre, d_nom)
    dist_pre.append(d_min)
    print(d_min)
    predict[i][:] = predictions

###### Affichage des résultats ######

savemat('./Dist_pred.mat', {'dist': dist_pre}) 
savemat('./Deconv_pred.mat', {'pred': predict}) 
