# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq, fftshift
from scipy.signal import hilbert, butter, filtfilt


def sfft(signal, fs, fmax=None,fmin=0):
    N = len(signal)

    # FFT
    S = fft(signal)

    # Axe fréquence (Hz)
    f = fftfreq(N, d=1/fs)

    # Partie positive
    mask = f >= 0
    f = f[mask]
    S = S[mask]

    # Module normalisé
    S_mag = np.abs(S) / N

    # Option : spectre simple face (×2 sauf DC)
    S_mag[1:] *= 2

    # Limite fréquentielle
    if fmax is not None:
        idx = f <= fmax
        f = f[idx]
        S_mag = S_mag[idx]
    
    idx = f >= fmin
    f = f[idx]
    S_mag = S_mag[idx]

    plt.figure(figsize=(8,4))
    plt.plot(f, S_mag)
    plt.yscale("log")
    plt.xlabel("Fréquence (Hz)")
    plt.ylabel("Amplitude")
    plt.title("Spectre (fréquences positives)")
    plt.grid(True)
    plt.show()


fs = 50000 # Sampling fmuency (Hz)
t = np.arange(0, 0.10, 1.0/fs)  # 0.1 second of data
fm = 100  # signal frq (Hz)
msg = np.sin(2 * np.pi * fm * t)  #signal
nb_frame_perdiod=int(fs/fm)

fp = 5000
p = np.sin(2*np.pi*fp*t)

#------AM Mod------
m = 0.5
am_sig = (1+m*msg)*p

#------FM Mod------
kf = 1000
phase = 2*np.pi*fp*t + 2*np.pi*kf*np.cumsum(msg)/fs
fm_sig = np.cos(phase)


#Gaussian noise
np.random.seed(0)
noise = np.random.randn(len(t))
am_noisy = am_sig + 0.5 * noise #nbre aléa selon distrib normale
msg_noisy = msg + 0.5 * noise
fm_noisy = fm_sig + 0.5 * noise

#------Demod AM-----
am_analytic = hilbert(am_noisy) #Transforme le signal recu en signal complexe en bande de base (ex: cos(wt) -> e^jwt)
#La transfo d'hilbert fait : am_sig=(1+m*x)*sin(wp*t) -> am_sig=(1+m*x)*e^jwp*t

am_env=np.abs(am_analytic)
#On récupère l'envellope : |am_sig=(1+m*x)*p| -> |(1+m*x)*1| = |(1+m*x)|
# il ne reste plus qu'à retrancher 1 et on a m*x ou x est le signal modulant
# pas fou car abs rajoute une compo continue avec l'ajout du bruit

# basse bas avec fc=2khz pour enlever les harmoniques sup
b,a = butter(2,fm*2/fs)
am_demod = filtfilt(b,a,am_env)

filtered_msg_noisy= filtfilt(b,a,msg_noisy)

am_demod_dc = am_demod - np.mean(am_demod) #On retire la composante continue

gain = 1 / np.sqrt(np.mean(am_demod_dc**2)) # En vrai ACG
am_demod_norm = gain * am_demod_dc

#-----Version detection synchrone------
am_demod_s = am_sig * np.sin(2*np.pi*fp*t)
am_demod_sf = filtfilt(b,a,am_demod_s)

am_demod_sfdc = am_demod_sf - np.mean(am_demod_sf) #On retire la composante continue

gain = 1 / np.sqrt(np.mean(am_demod_sfdc**2)) # En vrai ACG
am_demod_synchr = gain * am_demod_sfdc


#------Demod FM------
"""
# Demodulation: dérivée de la phase depuis la phase... (inutile...)
fm_analytic = np.unwrap(np.angle(np.exp(1j*phase)))
fm_demod = np.diff(fm_analytic) * fs / (2*np.pi*kf)
fm_demod -= np.mean(fm_demod)
"""
fm_analytic = hilbert(fm_noisy)
phase = np.unwrap(np.angle(fm_analytic))

fm_demod = np.diff(phase) * fs / (2*np.pi*kf)
fm_demod -= np.mean(fm_demod)

fm_demod_filtered = filtfilt(b,a,fm_demod)

gain = 1 / np.sqrt(np.mean(fm_demod_filtered**2)) # En vrai ACG
fm_demod_fn = gain * fm_demod_filtered

#On retire la 1ere et derniere période : effet de bord simulation

def del_eff_bord(signal):
    return signal[nb_frame_perdiod:len(t)-nb_frame_perdiod]

sfft(msg_noisy,fs,fmax=1000)
sfft(msg,fs,fmax=1000)
sfft(am_noisy,fs,fmax=6000,fmin=4000)

am_demod_norm=del_eff_bord(am_demod_norm)
msg=del_eff_bord(msg)
msg_noisy=del_eff_bord(msg_noisy)
filtered_msg_noisy=del_eff_bord(filtered_msg_noisy)
fm_demod_fn=del_eff_bord(fm_demod_fn)
am_demod_synchr=del_eff_bord(am_demod_synchr)

t=del_eff_bord(t) - nb_frame_perdiod/fs


plt.plot(t, msg_noisy, alpha=0.3)
plt.plot(t, msg, '--')
plt.plot(t, am_demod_norm,alpha=0.5)
#plt.plot(t, filtered_msg_noisy, alpha=0.5)
plt.plot(t,fm_demod_fn,alpha=0.5)
plt.plot(t, am_demod_synchr)

plt.title("Noisy Signal")
plt.xlabel("Time [s]")
plt.ylabel("Amplitude")
plt.show()