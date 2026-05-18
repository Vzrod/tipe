# -*- coding: utf-8 -*-
"""
Created on Mon Feb  9 13:32:28 2026

@author: arthu
"""
import numpy as np
import matplotlib.pyplot as plt #for plotting functions azd

from scipy.special import erfc

#%% SFFT

from scipy.fft import fft, fftfreq

def sfft(signal, fs, fmax=None, fmin=0, ylim=(-80, 10), title="Spectre",legend=None):
    N = len(signal)
    S = fft(signal)
    f = fftfreq(N, d=1.0/fs)
    
    mask = f >= 0
    f, S = f[mask], S[mask]
    
    S_mag = np.abs(S) / N
    S_mag[1:] *= 2
    
    S_dB = 20 * np.log10(np.maximum(S_mag, 1e-12))
    
    if fmax is not None:
        idx = f <= fmax
        f, S_dB = f[idx], S_dB[idx]
    idx = f >= fmin
    f, S_dB = f[idx], S_dB[idx]
    
    plt.figure(figsize=(8, 4))
    plt.plot(f, S_dB,label=legend)
    plt.xlabel("Fréquence (Hz)")
    plt.ylabel("Amplitude (dB)")
    plt.ylim(ylim)
    plt.title(title)  
    plt.grid(True)
    plt.legend(loc="upper right",handlelength=0,handletextpad=0)
    plt.show()
    plt.close()
    
#%% Focntions de bases 

def legend_params(SNRbdB, Lc, Nc, fc):
    return f"SNRbdB={SNRbdB}\nNc={Nc}\nfc={fc}"


def a_signal(s,t,title="",xa="",ya="",legend=None):
    plt.plot(t,s,label=legend)
    plt.title(title)
    plt.xlabel(xa)
    plt.ylabel(ya)
    plt.legend(loc="upper right",handlelength=0,handletextpad=0)
    plt.show()
    plt.close()
    
def constellation_graph(s, title="Constellation graph"):
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
    
def fading(s,a,tau,dt):
    """On simule un rebond du signal avec une onde réfléchie s'ajoutant au signal principal (chemin
    direct). a % de réflexion et tau le retard de l'onde réfléchie en s"""
    
    #
    r = (1-a)*s + a*(np.concatenate((np.zeros(int(tau/dt)),s)))[0:s.shape[0]] #on cut la fin du signal rebondi
    return r

def rayleigh_fading(s, L, omega=1.0):
    """
    Flat slow fading : 1 coefficient alpha par bit (cohérence > T_b)
    omega = E[alpha^2] = puissance moyenne du canal A VERIFIER
    """
    Nb = len(s) // L
    sigma = np.sqrt(omega / 2)
    X = sigma * np.random.randn(Nb)
    Y = sigma * np.random.randn(Nb)
    alpha = np.sqrt(X**2 + Y**2)
    alpha_exp = np.repeat(alpha, L)[:len(s)]
    return alpha_exp * s


#%% BPSK

def nrz_encoder(ak,L):
    """
    On récupère la liste des bits et on retourne une liste où pour 1->1 et 0->-1
    """
    s=[]
    for a in ak:
        s+=L*[2*a-1] 
    return np.array(s)

def bpsk_mod(ak,Lc,Nc):
    """L est l'oversampling factor (ratio fs/fc) -> nbre d'échatillion pour le codage d'un bit"""
    #Lc : nb échantillion par période de la porteuse
    #Nc : nb de période de la porteuse par bit 
    
    # générer le nrz est suffisant car déphaser de pi reveviens à multiplier par -1
    # à montrer mathématiquement
    
    L=Lc*Nc
    s_bb = nrz_encoder(ak, L) #signal en bande de base (1 bit sur une durée 1*L)
    t = np.arange(len(ak)*L) #échelle de temps
    return s_bb,t

def bpsk_demod(r_bb,L):
    r_bb = np.convolve(r_bb,[1]*L) #on intègre sur la durée d'1 bit
    r_bb = r_bb[L-1:-1:L] #on recup la val de l'intégrale qui est toute les L valeurs 
    ak_r = (r_bb > 0).transpose() # threshold detector
    return ak_r,r_bb

def simu_bpsk(ak,SNRbdB,Lc,Nc,fc,a=0,tau=0):
    L=Lc*Nc  #nombre d'échantillions par bit
    legend=legend_params(SNRbdB, Lc, Nc, fc)
    
    fs=fc*Lc #on def la freqc de sampling
    Rb = fc/Nc #débit binaire Rb = 1/Tb
    BER = 0
    (s_bb,t)=bpsk_mod(ak,Lc,Nc) #on récupère le signal modulé
    t=t/fs #passage temps discret à temps réel
    dt = 1/fs #pas temporel
    print(dt)
    
    s_p = np.cos(2*np.pi*fc*t) #signal de la porteuse
    
    sfft(s_p,fs,fmax=200,title="Spectre du signal en bande de base BPSK",legend=legend)
    
    s=s_bb*s_p 
    
    a_signal(s_bb[0:L*5],t[0:L*5],title="Signal en bande de base BPSK",xa="Temps (s)",legend=legend)
    a_signal(s[0:L*5],t[0:L*5],title="Signal modulé BPSK", xa="Temps (s)",legend=legend)
    
    sfft(s,fs,fmax=fc*2,title='Spectre du signal modulé BPSK',legend=legend)
    
    #r = fading(s,a,tau,dt)

    r = rayleigh_fading(s, L, omega=1)
    
    r = awgn(s,SNRbdB,L) #on récupère le signal bruité
    a_signal(r[0:L*5],t[0:L*5],title="Signal bruité BPSK", xa="Temps (s)",legend=legend)

    
    sfft(r,fs,fmax=fc*2,title='Spectre du signal bruité BPSK',legend=legend)
    
    r_bb=r*s_p #on convertit en bb en supposant le récepteur synchrone
    
    #par la double multi par la porteuse on multiplie par 0.5*L
    #quand on intègre, d'où la correction CF p9
    r_bb /= 0.5*L
    
    ak_r,x = bpsk_demod(r_bb, L) #démodulation
    
    constellation_graph(x)
    
    BER = np.sum(ak!=ak_r)/len(ak)
    
    
    a_signal(r_bb[0:L*5],t[0:L*5],title="Signal démodulé BPSK", xa="Temps (s)",legend=legend)
    
    return BER,x

def simu_bpsk_nog(ak,SNRbdB,L,fc,a=0,tau=0):
    fs=fc*L #on def la freqc de sampling
    BER = 0
    (s_bb,t)=bpsk_mod(ak,L,1) #on récupère le signal modulé
    t=t/fs #passage temps discret à temps réel
    dt=1/fs
    
    s_p = np.cos(2*np.pi*fc*t) #signal de la porteuse
    
    s=s_bb*s_p 
    
    r_faded = rayleigh_fading(s, L, omega=1.0)
    
    r = awgn(r_faded,SNRbdB,L) #on récupère le signal bruité
    
    #r = fading(r, a, tau, dt)
    
    
    
    r_bb=r*s_p #on convertit en bb en supposant le récepteur synchrone
    
    #par la double multi par la porteuse on multiplie par 0.5*L
    #quand on intègre, d'où la correction CF p9
    r_bb /= 0.5*L
    
    ak_r,x = bpsk_demod(r_bb, L) #démodulation
    
    x.astype(complex)
    
    #constellation_graph(x)
    
    BER = np.sum(ak!=ak_r)/len(ak)
    
    return BER





#%% QPSK

def qpsk_mod(ak,fc,L):
    I = ak[0::2] #on récup les bits pairs, (0:(fin):pas=2)
    Q = ak[1::2]
    
    I = nrz_encoder(I, L)
    Q = nrz_encoder(Q, L)
    
    t = np.arange(len(I))
    
    return I,Q,t

def qpsk_demod(r,fc,L,graph=False):
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
    
    if graph: constellation_graph(I+1j*Q,'constellation QPSK')
    
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
    
    ak_r = qpsk_demod(r, fc, L,graph=True) #démodulation
        
    BER = np.sum(ak!=ak_r)/len(ak)
    
    
    a_signal(r[0:L*10],t[0:L*10],"r")
    
    
    print('SNRsdB : ',SNRsdB,'dB')
    return BER,np.sum(ak!=ak_r)

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



#%% ASK

def ask_mod(ak,L):
    """L est l'oversampling factor (ratio fs/fc) -> nbre d'échatillion pour le codage d'un bit"""
    # générer le nrz est suffisant car déphaser de pi reveviens à multiplier par -1
    # à montrer mathématiquement
    
    s_bb=[]
    for a in ak:
        s_bb+=L*[a] 
    t = np.arange(len(ak)*L) #échelle de temps
    return np.array(s_bb),t

def ask_demod(r_bb,L):
    x = np.real(r_bb) # signal recu
    x = np.convolve(x,[1]*L) #on intègre sur la durée d'1 bit
    x = x[L-1:-1:L] # I arm - sample at every L
    #print(x)
    #print("piqndinqzdizpqn")
    ak_r = (x > 0.5).transpose() # threshold detector
    return ak_r,x

def simu_ask(ak,SNRbdB,L,fc):
    fs=fc*L #on def la freqc de sampling
    BER = 0
    (s_bb,t)=ask_mod(ak,L) #on récupère le signal modulé
    t=t/fs #passage temps discret à temps réel
    
    s_p = np.cos(2*np.pi*fc*t) #signal de la porteuse
    
    sfft(s_p,fs,fmax=200,title="s_p")
    
    s=s_bb*s_p 
    
    a_signal(s_bb[0:L*5],t[0:L*5],"s_bb")
    a_signal(s[0:L*5],t[0:L*5],"s")
    
    sfft(s,fs,fmax=fc*2,title='s')
    
    r = awgn(s,SNRbdB,L) #on récupère le signal bruité
    
    sfft(r,fs,fmax=fc*2,title='r')
    
    r_bb=r*s_p #on convertit en bb en supposant le récepteur synchrone
    
    #par la double multi par la porteuse on multiplie par 0.5*L
    #quand on intègre, d'où la correction CF p9
    r_bb /= 0.5*L
    
    ak_r,x = ask_demod(r_bb, L) #démodulation
    
    constellation_graph(x)
    
    BER = np.sum(ak!=ak_r)/len(ak)
    
    
    a_signal(r[0:L*5],t[0:L*5],"r")
    a_signal(r_bb[0:L*5],t[0:L*5],"r_bb")
    
    return BER,

def simu_ask_nog(ak,SNRbdB,L,fc):
    fs=fc*L #on def la freqc de sampling
    BER = 0
    (s_bb,t)=ask_mod(ak,L) #on récupère le signal modulé
    t=t/fs #passage temps discret à temps réel
    
    s_p = np.cos(2*np.pi*fc*t) #signal de la porteuse
    
    s=s_bb*s_p 
    
    r = awgn(s,SNRbdB,L) #on récupère le signal bruité
    
    r_bb=r*s_p #on convertit en bb en supposant le récepteur synchrone
    
    #par la double multi par la porteuse on multiplie par 0.5*L
    #quand on intègre, d'où la correction CF p9
    r_bb /= 0.5*L
    
    ak_r,x = ask_demod(r_bb, L) #démodulation
    
    BER = np.sum(ak!=ak_r)/len(ak)
    
    return BER



#%% BFSK

def bfsk_mod(ak, L, fc):
    """L = échantillons/bit, f0=fc, f1=2fc (orthogonales sur Tb=1/fc)"""
    fs = fc * L
    t = np.arange(len(ak) * L) / fs
    f_inst = np.repeat(np.where(np.asarray(ak) == 1, 2*fc, fc), L)
    s = np.cos(2*np.pi * f_inst * t)
    return s, t




#%% Simu ASK

ak=np.random.randint(2,size=int(100000))
ak[0]=0
ak[1]=0
ak[2]=1
ak[3]=1
ak[4]=0
fc=200
ber=simu_ask(ak, 20, 16, fc) #SNRbdB
print(ber)

#! -> unité ordonnée r_bb graph 

#%% Simu BPSK

ak=np.random.randint(2,size=int(100000))
ak[0]=1
ak[1]=0
ak[2]=1
ak[3]=1
ak[4]=1
fc=100
Lc=16 #Res porteuse
Nc=1 #1bit = 4periodes de la porteuse
SNRbdB=8
ber,ak_r=simu_bpsk(ak,SNRbdB,Lc,Nc,fc,a=0.5,tau=0.000625*8) #SNRbdB
#ber2=simu_bpsk_nog(ak,SNRbdB,Lc,fc,a=0.5,tau=0.000625*8) #SNRbdB
print(f'{ber:.4e}')

#On remarque que meme en augmentant Nc (bit plus long) le BER ne change pas
#WTF à chaque décalage d'un bit on double le w du sin card

#%% Simu QPSK

ak=np.random.randint(2,size=int(100000))

fc=100
ber,ak_r=simu_qpsk(ak, 20, 64, fc) #SNRbdB
print(ber)

#%% Graph BER BPSK QPSK

l_SNRbdB = range(-4,20,1)
BER_bpsk = []
BER_qpsk = []
BER_ask = []


fc=100

for SNRbdB in l_SNRbdB:
    m_BPSK=0
    m_QPSK=0
    m_ASK=0
    for i in range(10):
        ak=np.random.randint(2,size=int(10000))
        m_BPSK+=float(simu_bpsk_nog(ak, SNRbdB, 16, fc))
        m_QPSK+=float(simu_qpsk_nog(ak, SNRbdB, 16, fc))
        m_ASK+=float(simu_ask_nog(ak, SNRbdB, 16, fc))
        print(i)
    m_BPSK/=100
    m_QPSK/=100
    m_ASK/=100
    BER_bpsk.append(m_BPSK)
    BER_qpsk.append(m_QPSK)
    BER_ask.append(m_ASK)
    print(SNRbdB)

plt.plot(l_SNRbdB,BER_bpsk)
plt.plot(l_SNRbdB,BER_qpsk)
plt.plot(l_SNRbdB,BER_ask)
plt.yscale('log')
plt.show()


#%% graph BPSK fct Nc

BER = []
Nc = [1,2,4,6,8]

fc=100
Lc=16 #Res porteuse
SNRbdB=4

for N in Nc:
    ak=np.random.randint(2,size=int(100000))
    ber,ak_r=simu_bpsk(ak,SNRbdB,Lc,N,fc) #SNRbdB
    BER.append(ber)
    print(ber)

plt.plot(Nc,BER)
plt.show()

#%% graph BPSK Fading fct 

l_BER = []
for i in range(0,20):
    ak=np.random.randint(2,size=int(100000))
    l_BER.append(simu_bpsk_nog(ak,SNRbdB,Lc,fc,a=0.7,tau=0.000625*i))
    print(i)
