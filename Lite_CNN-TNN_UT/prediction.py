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

import convert_A_scan as convert

def prediction(model, A_scan, F_ech, F, Cl, dist_min, filtre, d_nom = 0):
    # Input et reshape
    test = A_scan
    test = convert.convert_A_scan(test,F_ech, filtre)
    N = len(A_scan)
    test = test.reshape(1,N,1)
      
    # Si le signal est nul -> distance nulle
    if test[0,10] == 0:
      dist = 0
      predictions = np.zeros([1, 2000])
      return dist, predictions
    
    # Sinon
    input_details = model.get_input_details()
    output_details = model.get_output_details()
    
    test = np.array(test, dtype=input_details[0]['dtype'])
    
    # Paramètres
    pts_max = 1980
    d = np.round(dist_min*F_ech/Cl) # distance minimale entre les pics de déconvolution
    precision = np.round(F_ech/F) # précision de longueur d'onde si d_nom connue pour la méthode de calcul d'épaisseur
    diff_nom = round(d_nom*F_ech/Cl*2)
    
    # Prédictions
    model.set_tensor(input_details[0]['index'], test)
    
    # Exécuter l'inférence
    model.invoke()
    
    # Récupérer les résultats
    output_data = model.get_tensor(output_details[0]['index'])
      
    # Obtention de l'épaisseur (méthode moyenne si aucune distance nominale connue ou distance variable)
    predictions = output_data.reshape(2000)
    peaks,_ = find_peaks(predictions[1:pts_max], distance=d, prominence=0.5*max(predictions))
    indices = peaks
    
    if len(indices)<2:
      dist = 0
      return dist, predictions
      
    distances = np.diff(indices)
    moy = np.mean(distances)
    difference = np.abs(distances - moy)
    idx = np.argmin(difference)
    distance = distances[idx]

    
    # Algo : si d_nom connue
    if d_nom > 0:  
      if np.abs(diff_nom - distance) > precision:
        # Méthode 2 - max
        indices2 = sorted(peaks, key=lambda i: predictions[i], reverse=True)
        
        if len(indices2)<2:
          dist = 0
          return dist, predictions
          
        indices2 = indices2[:2]
        distance = np.abs(indices2[1]-indices2[0])
        
        if np.abs(diff_nom - distance) > precision:
          # Méthode 3 - previous
          distances = np.abs(peaks[:, None] - peaks)
          distances = distances[np.triu_indices(len(peaks), k=1)]
          difference = np.abs(distances - diff_nom)
          idx = np.argmin(difference)
          distance = distances[idx]
          
          if np.abs(diff_nom - distance) > precision:
            # Faux positif
            dist = 0

            return dist, predictions
    
    # On calcul le temps de vol puis la distance entre les 2 pics
    dist = abs(distance)*Cl*1/F_ech/2 
    return dist, predictions