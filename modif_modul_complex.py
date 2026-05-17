# -*- coding: utf-8 -*-
"""
Created on Mon Feb  9 13:32:28 2026

@author: arthu
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq

#%% Fonctions de bases

d_bit_par_symb={"bpsk_map":1,"qpsk_map":2,"ask_map":1,"_16qam_map":4}

def nrz_encoder(sym, L):
    """etale chaque symbole sur L echantillons"""
    return np.repeat(sym, L)

def to_passband(s_bb, fc, fs):
    """signal en baseband complexe -> signal réel passband
    """
    t = np.arange(len(s_bb)) / fs
    return np.real(s_bb * np.exp(1j * 2 * np.pi * fc * t))

def to_baseband(r, fc, fs):
    """signal réel en pb -> signal complexe en bb"""
    t = np.arange(len(r)) / fs
    return 2.0 * r * np.exp(-1j * 2 * np.pi * fc * t)

def demod_filter(r_bb, L):
    """filtre rectangulaire (intégrateur sur L échantillons)
    et échantillonnage 1 fois par symbole."""
    h = np.ones(L) / L
    y = np.convolve(r_bb, h, mode='full')
    return y[L - 1::L][:len(r_bb) // L] #récupere les res des convolutions=intégrations et enleve le reste de la convolution inutile
    #le 2e slicing garde l'intégration des N=len(r_bb) // L symboles du message

def awgn(s, SNRbdB, bit_par_symb,L):
    """Ajoute un bruit AWGN selon Eb/N0=SNRb.
    """
    SNRb = 10 ** (SNRbdB / 10)
    Ps = np.mean(np.abs(s) ** 2)
    Es = Ps*L #energie par symbole
    Eb = Es / bit_par_symb
    N0 = Eb / SNRb
    sigma = np.sqrt(N0 / 2)
    return s + sigma * np.random.randn(len(s))


#%% Mapping

def bpsk_map(bk):
    """0 -> -1 et 1 -> +1 (symboles sur l'axe I)."""
    return (2 * bk - 1).astype(complex)

def bpsk_demap(r_sym):
    """Décision : signe de la partie réelle (axe I)."""
    return (r_sym.real > 0).astype(int)


def qpsk_map(bk):
    """1 symbole = 2 bits"""
    n_bits=len(bk)-(len(bk) % 2)
    n_symb=n_bits//2
    
    symbols=np.zeros(n_symb, dtype=complex)
    
    for i in range(n_symb):
        b0=bk[2*i]
        b1=bk[2*i+1]
        
        I= 2*b0-1
        Q= 2*b1-1
        
        #normalisation /sqrt(2) pour Es=1
        symbols[i] = (I+1j*Q)/np.sqrt(2)
        
    return symbols

def qpsk_demap(r_sym):
    n_symb = len(r_sym)
    bk_r = np.zeros(2*n_symb, dtype=int)
    
    for i in range(n_symb):
        #décision axe I
        if r_sym[i].real > 0: bk_r[2*i] = 1
        else: bk_r[2*i] = 0
            
        #Axe Q
        if r_sym[i].imag > 0: bk_r[2*i+1] = 1
        else: bk_r[2*i+1] = 0
            
    return bk_r

def ask_map(bk):
     return bk

def ask_demap(r_sym):
    bk_r = (r_sym > 0.5)
    return bk_r
    

#%%16 QAM
#code de Gray :associe 2 bits à une amplitude et inv
GRAY_MAP = {
    (0, 0): -3,
    (0, 1): -1,
    (1, 1):  1,
    (1, 0):  3 }

GRAY_INV = {
    -3: (0, 0),
    -1: (0, 1),
     1: (1, 1),
     3: (1, 0) }

def _16qam_map(ak):
    """ 
    Principe : 1 symbole = 4 bits consécutifs -> on les places dans le plan complexe
    ce qui nous donne une amplitude et une phase par symbole
    Rq : on traite les bits 4 par 4
    """
    n_bits = len(ak) - (len(ak) % 4) # on coupe le msg pour avoir un mult de 4
    n_symb = n_bits // 4
    
    symbols = np.zeros(n_symb, dtype=complex)
    
    for i in range(n_symb):
        b0 = ak[4 * i]
        b1 = ak[4 * i + 1]
        b2 = ak[4 * i + 2]
        b3 = ak[4 * i + 3]
        
        I = GRAY_MAP[(b0, b1)]
        Q = GRAY_MAP[(b2, b3)]
        
        # la division permet normalisation tq Es=1
        symbols[i] = (I + 1j * Q) / np.sqrt(10)
        
    return symbols

def _16qam_demap(y):
    n_symb = len(y)
    ak = np.zeros(4 * n_symb, dtype=int)
    
    for i in range(n_symb):
        #denormalisation
        I_r = np.real(y[i]) * np.sqrt(10)
        Q_r = np.imag(y[i]) * np.sqrt(10)
        
        #decision sur axe I (seuils à -2,0,2 pour avoir une amplitude de 1 centré en chaque point)
        if I_r < -2: I_decide = -3
        elif I_r < 0: I_decide = -1
        elif I_r < 2: I_decide = 1
        else: I_decide = 3
            
        #decision axe Q
        if Q_r < -2: Q_decide = -3
        elif Q_r < 0: Q_decide = -1
        elif Q_r < 2: Q_decide = 1
        else: Q_decide = 3
            
        # on récup les pairs de bits correspondantes
        b0, b1 = GRAY_INV[I_decide]
        b2, b3 = GRAY_INV[Q_decide]
        
        ak[4*i]=b0
        ak[4*i+1]=b1
        ak[4*i+2]=b2
        ak[4*i+3]=b3
        
    return ak

#%% Simu canal modulations linéaires

def simu_canal_lin(bk,b2s,s2b,SNRbdB,Lc, Nc, fc, plots=True):
    """
    Paramètres
    ----------
    bk   : ndarray de 0/1
    SNRbdB : Eb/N0 en dB
    Lc     : échantillons par période porteuse
    Nc     : périodes porteuse par bit         (durée du bit, Rb = fc/Nc)
    fc     : fréquence porteuse (Hz)

    Retour
    ------
    (BER, bk_r)   bk_r : bits reçus
    """
    L=Lc*Nc           # échantillons par bit
    fs=Lc*fc          # fréquence d'échantillonnage

    ak = b2s(bk) #Mapping
    #print('mapping ok')
    s_bb = nrz_encoder(ak, L)     #signal complexe en bb sur-échantillonée
    s = to_passband(s_bb, fc, fs)  #signal réel mod dans le canal

    # ---------- CANAL ----------
    r = awgn(s, SNRbdB, bit_par_symb=d_bit_par_symb.get(b2s.__name__),L=L)
    # ---------------------------
    
    r_bb_raw = to_baseband(r, fc, fs)   #signal complexe en bb
    r_sym = demod_filter(r_bb_raw, L) #filtre intégrateur
    bk_r = s2b(r_sym)

    BER = np.mean(bk != bk_r)

    if plots:
        _plot_chain(s_bb, s, r, r_sym, fc, fs, L, SNRbdB)

    return BER, bk_r


#%% Simu BFSK

def simu_canal_bfsk(bk, SNRbdB, Lc, Nc, fc, plots=True):
    """
    BFSK orthogonal cohérent :
        bit 0 → cos(2π f0 t) ,  bit 1 → cos(2π f1 t)
        f0 = fc ,  f1 = fc + Δf  avec  Δf = 1/Tb = fc/Nc
    (Δf · Tb = 1 garantit l'orthogonalité sur la durée d'un bit.)
    """
    L=Lc*Nc
    fs=Lc*fc
    df=fc/Nc
    f0,f1=fc, fc+df
    
    t = np.arange(len(bk) * L) / fs
    f_inst = [f0 if k==0 else f1 for k in bk]
    f_inst = np.repeat(f_inst,L)
    s = np.cos(2 * np.pi * f_inst * t)

    r = awgn(s, SNRbdB, bit_par_symb=1,L=L)

    # ---------- DÉMODULATION (corrélation cohérente sur chaque porteuse) ----------
    y0 = demod_filter(r * np.cos(2 * np.pi * f0 * t), L)
    y1 = demod_filter(r * np.cos(2 * np.pi * f1 * t), L)
    bk_r = (y1 > y0).astype(int)

    BER = np.mean(bk != bk_r)

    if plots:
        _plot_chain_bfsk(s, r, y0, y1, fc, df, fs, L, SNRbdB)

    return BER, bk_r


def _plot_chain_bfsk(s, r, y0, y1, fc, df, fs, L, SNRbdB):
    n = min(10 * L, len(s))
    t = np.arange(n) / fs
    _plot_signal(t, s[:n], r"Signal passband émis $s(t)$ (BFSK)", "s(t)")
    _plot_signal(t, r[:n], f"Signal reçu $r(t)$, $E_b/N_0$ = {SNRbdB} dB", "r(t)")

    # « Constellation » BFSK dans la base orthonormée (ψ0, ψ1)
    plt.figure(figsize=(5, 5))
    plt.scatter(y0, y1, s=8, alpha=0.4)
    lim = max(np.max(np.abs(y0)), np.max(np.abs(y1))) * 1.1
    plt.plot([-lim, lim], [-lim, lim], 'r--', lw=0.8, label=r'frontière $y_0 = y_1$')
    plt.axhline(0, color='k', lw=0.5); plt.axvline(0, color='k', lw=0.5)
    plt.xlabel(r"$y_0 = \langle r,\,\cos(2\pi f_0 t)\rangle$")
    plt.ylabel(r"$y_1 = \langle r,\,\cos(2\pi f_1 t)\rangle$")
    plt.title(f"Constellation BFSK orthogonale, $E_b/N_0$ = {SNRbdB} dB")
    plt.grid(True); plt.axis('equal'); plt.legend()
    plt.tight_layout(); plt.show(); plt.close()

    _plot_spectrum(s, fs, fmax=2.5 * fc + df, title="Spectre du signal BFSK passband")

bk = np.random.randint(2, size=100_000)
BER, bk_r = simu_canal_bfsk(bk, SNRbdB=10, Lc=16, Nc=1, fc=100, plots=True)
print(f"BER BFSK : {BER:.4e}")



#%% Graphiques

def _plot_signal(t, s, title, ylabel=""):
    plt.figure(figsize=(8, 3))
    plt.plot(t, s)
    plt.xlabel("Temps (s)"); plt.ylabel(ylabel); plt.title(title)
    plt.grid(True); plt.tight_layout(); plt.show(); plt.close()


def _plot_constellation(sym, title="Constellation"):
    plt.figure(figsize=(5, 5))
    plt.scatter(sym.real, sym.imag, s=8, alpha=0.4)
    plt.axhline(0, color='k', lw=0.5); plt.axvline(0, color='k', lw=0.5)
    plt.xlabel("I (in-phase)"); plt.ylabel("Q (quadrature)")
    plt.title(title); plt.grid(True); plt.axis('equal')
    plt.tight_layout(); plt.show(); plt.close()


def _plot_spectrum(s, fs, fmax=None, title="Spectre"):
    N = len(s); S = fft(s); f = fftfreq(N, d=1 / fs)
    mask = f >= 0; f, S = f[mask], S[mask]
    mag = 2 * np.abs(S) / N; mag[0] /= 2
    dB = 20 * np.log10(np.maximum(mag, 1e-12))
    if fmax is not None:
        idx = f <= fmax; f, dB = f[idx], dB[idx]
    plt.figure(figsize=(8, 3))
    plt.plot(f, dB); plt.ylim(-80, 10)
    plt.xlabel("Fréquence (Hz)"); plt.ylabel("Amplitude (dB)")
    plt.title(title); plt.grid(True); plt.tight_layout(); plt.show(); plt.close()


def _plot_chain(s_bb, s, r, r_sym, fc, fs, L, SNRbdB):
    n = min(10 * L, len(s))
    t = np.arange(n) / fs
    _plot_signal(t, s_bb.real[:n], r"Enveloppe complexe — composante I(t)", "I")
    _plot_signal(t, s[:n],         r"Signal passband émis $s(t)$ (réel)", "s(t)")
    _plot_signal(t, r[:n],         f"Signal reçu $r(t)$, $E_b/N_0$ = {SNRbdB} dB", "r(t)")
    _plot_constellation(r_sym, f"Constellation BPSK, $E_b/N_0$ = {SNRbdB} dB")
    _plot_spectrum(s, fs, fmax=2.5 * fc, title="Spectre du signal passband")


#%% Simu BPSK

bk = np.random.randint(2, size=100_000)

BER, bk_r = simu_canal_lin(bk,bpsk_map,bpsk_demap, SNRbdB=4, Lc=16, Nc=1, fc=100, plots=True)
print(f"BER : {BER:.4e}")

#%% Simu ASK

bk = np.random.randint(2, size=100_000)

BER, bk_r = simu_canal_lin(bk,ask_map,ask_demap, SNRbdB=4, Lc=16, Nc=1, fc=100, plots=True)
print(f"BER : {BER:.4e}")


#%%Simu 16QAM

bk = np.random.randint(2, size=100_000)
BER, _  = simu_canal_lin(bk, _16qam_map, _16qam_demap, SNRbdB=15, Lc=16,Nc=1,fc=100, plots=True)
print(f"BER : {BER:.4e}")

#%%Graphique comparaison BER
l_SNRbdB = range(-4,16,1)
BER_bpsk,BER_qpsk,BER_ask,BER_bfsk,BER_16qam = [],[],[],[],[]

fc=100

for _SNRbdB in l_SNRbdB:
    m_BPSK,m_QPSK,m_ASK,m_BFSK,m_16QAM=0,0,0,0,0
    for i in range(1):
        ak=np.random.randint(2,size=int(100_000))
        m_BPSK+=float(simu_canal_lin(ak,bpsk_map,bpsk_demap,SNRbdB=_SNRbdB,Lc=16,Nc=1,fc=100,plots=False)[0])
        m_QPSK+=float(simu_canal_lin(ak,qpsk_map,qpsk_demap,SNRbdB=_SNRbdB,Lc=16,Nc=1,fc=100,plots=False)[0])
        m_ASK+=float(simu_canal_lin(ak,ask_map,ask_demap,SNRbdB=_SNRbdB,Lc=16,Nc=1,fc=100,plots=False)[0])
        m_16QAM+=float(simu_canal_lin(ak,_16qam_map,_16qam_demap,SNRbdB=_SNRbdB,Lc=16,Nc=1,fc=100,plots=False)[0])
        m_BFSK+=float(simu_canal_bfsk(ak, SNRbdB=_SNRbdB, Lc=16, Nc=1, fc=100, plots=False)[0])
        
        print(i)
    m_BPSK/=1
    m_QPSK/=1
    m_ASK/=1
    m_16QAM/=1
    m_BFSK/=1
    
    BER_bpsk.append(m_BPSK)
    BER_qpsk.append(m_QPSK)
    BER_ask.append(m_ASK)
    BER_16qam.append(m_16QAM)
    BER_bfsk.append(m_BFSK)
    print(_SNRbdB)

"""
plt.plot(l_SNRbdB,BER_bpsk, "o")
plt.plot(l_SNRbdB,BER_qpsk,"o")
plt.plot(l_SNRbdB,BER_ask,"o")
plt.plot(l_SNRbdB,BER_16qam,"o")
plt.plot(l_SNRbdB,BER_bfsk,"o")
plt.yscale('log')
plt.show()
"""
#%%BER theorique
from scipy.special import erfc

def Q(x):
    return 0.5*erfc(np.sqrt(0.5)*x)

def dB_to_dec(SNRbdB):    
    return 10 ** (SNRbdB / 10)

BER_bpsk_th = [Q(np.sqrt(2*dB_to_dec(_SNRbdB))) for _SNRbdB in l_SNRbdB]
BER_qpsk_th = BER_bpsk_th
BER_ask_th = [Q(np.sqrt(dB_to_dec(_SNRbdB))) for _SNRbdB in l_SNRbdB]
BER_16qam_th = [0.75*Q(np.sqrt(0.8*dB_to_dec(_SNRbdB))) for _SNRbdB in l_SNRbdB]
BER_bfsk_th = BER_ask_th


#Tracés BER simu
plt.plot(l_SNRbdB,BER_bpsk, "+b")
plt.plot(l_SNRbdB,BER_qpsk,"xg")
plt.plot(l_SNRbdB,BER_ask,"+r")
plt.plot(l_SNRbdB,BER_16qam,"xm")
plt.plot(l_SNRbdB,BER_bfsk,"+c")

#Tracés BER théoriques
plt.plot(l_SNRbdB,BER_bpsk_th, "-.b")
plt.plot(l_SNRbdB,BER_qpsk_th, "--g")
plt.plot(l_SNRbdB,BER_ask_th, "--r")
plt.plot(l_SNRbdB,BER_16qam_th, "--m")
plt.plot(l_SNRbdB,BER_bfsk_th, "-.c")

plt.yscale('log')
plt.ylim(1e-6, 1)
plt.show()














