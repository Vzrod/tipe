# -*- coding: utf-8 -*-
"""
Created on Mon Feb  9 13:32:28 2026

@author: arthu
"""
import numpy as np
import matplotlib.pyplot as plt #for plotting functions

from scipy.special import erfc


def nrz_encoder(ak,L):
    """
    On récupère la liste des bits et on retourne une liste où pour 1->1 et 0->-1
    """
    s=[]
    for a in ak:
        s+=L*[2*a-1] 
    return np.array(s)

def bpsk_mod(ak,L):
    """L est l'oversampling factor (ratio fs/fc) -> nbre d'échatillion pour le codage d'un bit"""
    # générer le nrz est suffisant car déphaser de pi reveviens à multiplier par -1
    # à montrer mathématiquement
    s_bb = nrz_encoder(ak, L) #signal en bande de base (1 bit sur une durée 1*L)
    t = np.arange(len(ak)*L) #échelle de temps
    return s_bb,t

def bpsk_demod(r_bb,L):
    x = np.real(r_bb) # signal recu
    x = np.convolve(x,[1]*L) #on intègre sur la durée d'1 bit
    x = x[L-1:-1:L] # I arm - sample at every L
    ak_r = (x > 0).transpose() # threshold detector
    return ak_r,x

def simu_bpsk(ak,SNRbdB,L,fc):
    fs=fc*L #on def la freqc de sampling
    BER = 0
    (s_bb,t)=bpsk_mod(ak,L) #on récupère le signal modulé
    t=t/fs #passage temps discret à temps réel
    
    s_p = np.cos(2*np.pi*fc*t) #signal de la porteuse
    
    sfft(s_p,fs,fmax=200)
    
    s=s_bb*s_p 
    
    a_signal(s_bb[0:L*10],t[0:L*10],"s_bb")
    a_signal(s[0:L*10],t[0:L*10],"s")
    
    sfft(s[0:L*10],fs,fmax=fc*2)
    
    r = awgn(s,SNRbdB,L) #on récupère le signal bruité
    
    sfft(r[0:L*10],fs,fmax=fc*2)
    
    r_bb=r*s_p #on convertit en bb en supposant le récepteur synchrone
    
    #par la double multi par la porteuse on multiplie par 0.5*L
    #quand on intègre, d'où la correction CF p9
    r_bb /= 0.5*L
    
    ak_r,x = bpsk_demod(r_bb, L) #démodulation
    
    constellation_graph(x)
    
    BER = np.sum(ak!=ak_r)/len(ak)
    
    
    a_signal(r[0:L*10],t[0:L*10],"r")
    a_signal(r_bb[0:L*10],t[0:L*10],"r_bb")
    
    return BER,x

def simu_bpsk_nog(ak,SNRbdB,L,fc):
    fs=fc*L #on def la freqc de sampling
    BER = 0
    (s_bb,t)=bpsk_mod(ak,L) #on récupère le signal modulé
    t=t/fs #passage temps discret à temps réel
    
    s_p = np.cos(2*np.pi*fc*t) #signal de la porteuse
    
    s=s_bb*s_p 
    
    r = awgn(s,SNRbdB,L) #on récupère le signal bruité
    
    r_bb=r*s_p #on convertit en bb en supposant le récepteur synchrone
    
    #par la double multi par la porteuse on multiplie par 0.5*L
    #quand on intègre, d'où la correction CF p9
    r_bb /= 0.5*L
    
    ak_r,x = bpsk_demod(r_bb, L) #démodulation
    
    x.astype(complex)
    
    constellation_graph(x)
    
    BER = np.sum(ak!=ak_r)/len(ak)
    
    return BER

def a_signal(s,t,title=""):
    plt.plot(t,s)
    plt.title(title)
    plt.show()
    plt.close()
    
def constellation_graph(s, title=""):
    X,Y=s.real,s.imag
    plt.scatter(X,Y)
    plt.title(title)
    plt.show()
    plt.close()
    

def awgn(s,SNRbdB,L):
    
    SNRb = 10**(SNRbdB/10) #SNR par bit
    
    Ps = sum(np.abs(s)**2)/len(s) #Puissance du signal facteur 
    
    Eb = Ps * L #ie un bit est étalé sur un temps L
    
    N0 = Eb/SNRb #densité spectrale de bruit
    
    r = s + np.sqrt(N0/2)*np.random.standard_normal(len(s)) #expliquer le sqrt(N0/2)
    return r
    


#%%

def qpsk_mod(ak,fc,L):
    I = ak[0::2] #on récup les bits pairs, (0:(fin):pas=2)
    Q = ak[1::2]
    
    I = nrz_encoder(I, L)
    Q = nrz_encoder(Q, L)
    
    t = np.arange(len(I))
    
    return I,Q,t

def qpsk_demod(r,fc,L):
    fs=L*fc
    t = np.arange(len(r))/fs 
    I = r * np.cos(2*np.pi*fc*t)
    Q = r * (-np.sin(2*np.pi*fc*t))
    
    I /= 0.5*L
    Q /= 0.5*L
    
    I = np.convolve(I, np.ones(L))
    Q = np.convolve(Q, np.ones(L))
    
    I = I[L-1::L]
    Q = Q[L-1::L]
    
    constellation_graph(I+1j*Q,'constellation QPSK')
    
    ak_r = np.zeros(2*len(I))
    ak_r[0::2] = (I>0)
    ak_r[1::2] = (Q>0)
    
    return ak_r

def simu_qpsk(ak,SNRbdB,L,fc):
    """
    SNRbdB : SNR par bit
    On a Lb l'oversampling pour 1 bit, d'ou Lb*2 = L = oversampling
    pour 1 symbole= 2 bits
    """
    SNRsdB = SNRbdB + 10*np.log10(2)
    fs=fc*L #on def la freqc de sampling
    BER = 0
    (I,Q,t)=qpsk_mod(ak,fc,L) #on récupère le signal modulé
    t=t/fs #passage temps discret à temps réel
    
    I_t = I * np.cos(2*np.pi*fc*t)
    Q_t = Q * np.sin(2*np.pi*fc*t)  
    
    s = I_t - Q_t
    
    a_signal(s[0:L*10],t[0:L*10],"s = I_t - Q_t")
    
    r = awgn(s,SNRsdB,L) #on récupère le signal bruité
    
    ak_r = qpsk_demod(r, fc, L) #démodulation
        
    BER = np.sum(ak!=ak_r)/len(ak)
    
    
    a_signal(r[0:L*10],t[0:L*10],"r")
    
    
    print('SNRsdB : ',SNRsdB,'dB')
    return BER,ak_r

def simu_qpsk_nog(ak,SNRbdB,L,fc):
    """
    SNRbdB : SNR par bit
    On a Lb l'oversampling pour 1 bit, d'ou Lb*2 = L = oversampling
    pour 1 symbole= 2 bits
    """
    SNRsdB = SNRbdB + 10*np.log10(2)
    fs=fc*L #on def la freqc de sampling
    BER = 0
    (I,Q,t)=qpsk_mod(ak,fc,L) #on récupère le signal modulé
    t=t/fs #passage temps discret à temps réel
    
    I_t = I * np.cos(2*np.pi*fc*t)
    Q_t = Q * np.sin(2*np.pi*fc*t)  
    
    s = I_t - Q_t
        
    r = awgn(s,SNRsdB,L) #on récupère le signal bruité
    
    ak_r = qpsk_demod(r, fc, L) #démodulation
        
    BER = np.sum(ak!=ak_r)/len(ak)
    
    return BER



#%%

ak=np.random.randint(2,size=int(100000))
fc=100
ber,ak_r=simu_bpsk(ak, 10, 64, fc) #SNRbdB
print(ber)

#%%

ak=np.random.randint(2,size=int(100000))
fc=100
ber,ak_r=simu_qpsk(ak, 8, 64, fc) #SNRbdB
print(ber)

#%%

l_SNRbdB = range(-4,14,2)
BER_bpsk = []
BER_qpsk = []

ak=np.random.randint(2,size=int(10e4))
fc=100

for SNRbdB in l_SNRbdB:
    BER_bpsk.append(float(simu_bpsk_nog(ak, SNRbdB, 16, fc)))
    BER_qpsk.append(float(simu_qpsk_nog(ak, SNRbdB, 16, fc)))
    print(SNRbdB)

plt.plot(l_SNRbdB,BER_bpsk)
plt.plot(l_SNRbdB,BER_qpsk)
plt.yscale('log')
plt.show()

#%%

from scipy.fft import fft, fftfreq

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

