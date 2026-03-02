# -*- coding: utf-8 -*-
"""
Created on Mon Feb  2 14:29:22 2026

@author: arthu
"""

from scipy.signal import upfirdn
import numpy as np


L = 4 #oversampling factor (Tbit/Tsampling)
ak = np.array([0,1,1,0,1,0])
t = np.linspace(0, L*(len(ak)))

x=2*ak-1

s_bb = upfirdn([1]*L, x, up=L)


fc = 100
car = np.sin(2*np.pi*fc*t)
