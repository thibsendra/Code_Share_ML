# -*- coding: utf-8 -*-

import numpy as np
import scipy

def convert_A_scan(A_scan, F_ech, filtre):
    A_scan = np.array(A_scan)
    
    # Resampling de F_ech à 100 MHz
    size_init = len(A_scan)
    A_scan = A_scan.reshape(size_init)
    temp_init = np.linspace(0, (size_init-1)/F_ech,size_init)
    temp_cible = np.linspace(0, temp_init[-1],int(np.round(size_init/F_ech*100e6)))
    interp_func = scipy.interpolate.interp1d(temp_init, A_scan, kind='cubic', fill_value='extrapolate')
    A_scan = interp_func(temp_cible)
    
    # Padding ou limitation à 2000 points
    size_cible = len(A_scan)
    if size_cible >= 2000:
        A_scan = A_scan[0:2000]
    else:
        A_scan = np.append(A_scan,np.zeros([2000-size_cible, ]))
    
    # Filtrage si filtre = False
    low = 2e6 / (100e6/2)
    high = 10e6 / (100e6/2)
    b, a = scipy.signal.iirfilter(N=1, Wn = [low, high], btype='band',ftype='butter')
    if filtre == False:
      A_scan = scipy.signal.lfilter(b,a,A_scan)
    
    # Fenêtre tukeywin
    wind = scipy.signal.windows.tukey(size_cible,alpha=0.1)
    wind[0:round(size_cible/2)]=1
    A_scan[0:size_cible] =  A_scan[0:size_cible]*wind
    
    # Normalisation
    A_scan = A_scan / max(np.abs(A_scan))
    
    return A_scan