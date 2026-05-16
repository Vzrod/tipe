# -*- coding: utf-8 -*-
"""
Relation fondamentale :
        s(t) = Re{ s_bb(t) · e^{j 2π fc t} }
             = I(t) cos(2π fc t) - Q(t) sin(2π fc t)
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq


#%% Fonctions de bases

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
    return y[L - 1::L][:len(r_bb) // L] #A détailler


def awgn(s, SNRbdB, symb_par_bit):
    """Ajoute un bruit AWGN selon Eb/N0 = SNRbdB dB.

    Détecte automatiquement la nature (réelle ou complexe) du signal :
      - réel    : bruit gaussien réel de variance N0/2
      - complexe: bruit gaussien réel de variance N0/2 par composante I et Q

    Calcul : Eb = Ps · symb_par_bit
    """
    SNRb = 10 ** (SNRbdB / 10)
    Ps = np.mean(np.abs(s) ** 2)
    Eb = Ps * symb_par_bit
    N0 = Eb / SNRb
    sigma = np.sqrt(N0 / 2)
    if np.iscomplexobj(s): #Expliquer pk on stack pas 2 fois le bruit...
        return s + sigma * (np.random.randn(len(s)) + 1j * np.random.randn(len(s)))
    return s + sigma * np.random.randn(len(s))


#%% Mapping

def bpsk_map(bk):
    """0 → -1+0j ;  1 → +1+0j   (symboles sur l'axe I)."""
    return (2 * bk - 1).astype(complex)

def bpsk_demap(r_sym):
    """Décision : signe de la partie réelle (axe I)."""
    return (r_sym.real > 0).astype(int)


def qpsk_map(bk):
    b = bk.reshape(-1, 2)
    return ((2*b[:,0]-1) + 1j*(2*b[:,1]-1)) / np.sqrt(2)  # /sqrt(2) pour Es=1

def qpsk_demap(r_sym):
    out = np.zeros(2*len(r_sym), dtype=int)
    out[0::2] = (r_sym.real > 0); out[1::2] = (r_sym.imag > 0)
    return out


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
    s_bb = nrz_encoder(ak, L)     #signal complexe en bb sur-échantillonée
    s = to_passband(s_bb, fc, fs)  #signal réel mod dans le canal

    # ---------- CANAL ----------
    r = awgn(s, SNRbdB, symb_par_bit=L)
    # ---------------------------
    
    r_bb_raw = to_baseband(r, fc, fs)   #signal complexe en bb
    r_sym = demod_filter(r_bb_raw, L) #filtre intégrateur
    bk_r = s2b(r_sym)

    BER = np.mean(bk != bk_r)

    if plots:
        _plot_chain(s_bb, s, r, r_sym, fc, fs, L, SNRbdB)

    return BER, bk_r


#%% Simu BFSK

def 

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


#%%

#np.random.seed(0)
bk = np.random.randint(2, size=100_000)

BER, bk_r = simu_canal_lin(bk,bpsk_map,bpsk_demap, SNRbdB=8, Lc=16, Nc=1, fc=100, plots=True)
print(f"BER simulé : {BER:.4e}")
