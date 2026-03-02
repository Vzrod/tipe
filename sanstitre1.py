# -*- coding: utf-8 -*-
"""
Created on Sun Feb 22 16:22:15 2026

@author: arthu
"""

import numpy as np
def bpsk_simulation(bits, EbN0_dB):
    """
    bits : liste ou array de 0 et 1
    EbN0_dB : SNR par bit en dB
    retourne : BER
    """
    
    bits = np.array(bits)
    N = len(bits)
    
    # =============================
    # 1) MODULATION BPSK
    # =============================
    # 0 -> -1
    # 1 -> +1
    symbols = 2*bits - 1   # transforme 0/1 en -1/+1
    
    # =============================
    # 2) CALCUL BRUIT (AWGN)
    # =============================
    
    EbN0 = 10**(EbN0_dB / 10)     # passage en linéaire
    Eb = 1                        # énergie par bit (car amplitude ±1)
    
    N0 = Eb / EbN0
    sigma = np.sqrt(N0/2)         # variance = N0/2
    
    noise = sigma * np.random.randn(N)
    
    # =============================
    # 3) SIGNAL REÇU
    # =============================
    
    received = symbols + noise
    
    # =============================
    # 4) DÉMODULATION
    # =============================
    
    detected_bits = (received > 0).astype(int)
    
    # =============================
    # 5) CALCUL BER
    # =============================
    
    errors = np.sum(bits != detected_bits)
    BER = errors / N
    
    return BER