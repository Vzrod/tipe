# -*- coding: utf-8 -*-
"""
Relation fondamentale :
        s(t) = Re{ s_bb(t) · e^{j 2π fc t} }
             = I(t) cos(2π fc t) - Q(t) sin(2π fc t)
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq


# =============================================================
# Fonctions de bases réutilisables
# =============================================================

def nrz_encoder(sym, L):
    """Étale chaque symbole sur L échantillons"""
    return np.repeat(sym, L)


def to_passband(s_bb, fc, fs):
    """Enveloppe complexe -> signal réel passband
        s(t) = Re{ s_bb(t) * exp(j 2π fc t) } = s_bb(t)*cos(2πfc*t)
    """
    t = np.arange(len(s_bb)) / fs
    return np.real(s_bb * np.exp(1j * 2 * np.pi * fc * t))


def to_baseband(r, fc, fs):
    """Signal réel passband -> enveloppe complexe (non filtrée).

    Multiplication par 2·exp(-j 2π fc t) :
      - le facteur 2 compense le 1/2 issu de cos·e^{-jωt} = (1 + e^{-j2ωt})/2 ;
      - le terme à 2fc sera supprimé par le filtre adapté (passe-bas).
    """
    t = np.arange(len(r)) / fs
    return 2.0 * r * np.exp(-1j * 2 * np.pi * fc * t)


def demod_filter(r_bb, L):
    """Filtre adapté rectangulaire (intégrateur sur L échantillons)
    suivi de l'échantillonnage 1 fois par symbole."""
    h = np.ones(L) / L #pk pas (1) ????
    y = np.convolve(r_bb, h, mode='full')
    return y[L - 1::L][:len(r_bb) // L]


def awgn(s, SNRbdB, samples_per_bit):
    """Ajoute un bruit AWGN selon Eb/N0 = SNRbdB dB.

    Détecte automatiquement la nature (réelle ou complexe) du signal :
      - réel    : bruit gaussien réel de variance N0/2
      - complexe: bruit circulaire complexe, N0/2 par composante I et Q

    Convention : Eb = Ps · samples_per_bit (énergie par bit = puissance × durée).
    """
    SNRb = 10 ** (SNRbdB / 10)
    Ps = np.mean(np.abs(s) ** 2)
    Eb = Ps * samples_per_bit
    N0 = Eb / SNRb
    sigma = np.sqrt(N0 / 2)
    if np.iscomplexobj(s): #Expliquer pk on stack pas 2 fois le bruit...
        return s + sigma * (np.random.randn(len(s)) + 1j * np.random.randn(len(s)))
    return s + sigma * np.random.randn(len(s))


# =============================================================
# Mapping (bit -> forme adaptée mod)
# =============================================================

def bpsk_map(bits):
    """0 → -1+0j ;  1 → +1+0j   (symboles sur l'axe I)."""
    return (2 * bits - 1).astype(complex)

def bpsk_demap(symbols):
    """Décision : signe de la partie réelle (bras I)."""
    return (symbols.real > 0).astype(int)


def qpsk_map(bits):
    b = bits.reshape(-1, 2)
    return ((2*b[:,0]-1) + 1j*(2*b[:,1]-1)) / np.sqrt(2)  # /√2 pour Es=1

def qpsk_demap(sym):
    out = np.zeros(2*len(sym), dtype=int)
    out[0::2] = (sym.real > 0); out[1::2] = (sym.imag > 0)
    return out




# =============================================================
# 3) Chaîne complète BPSK
# =============================================================

def simu_bpsk(bits, SNRbdB, Lc, Nc, fc, plots=True):
    """
    Paramètres
    ----------
    bits   : ndarray de 0/1
    SNRbdB : Eb/N0 en dB
    Lc     : échantillons par période porteuse (qualité du sinus numérisé)
    Nc     : périodes porteuse par bit         (durée du bit, Rb = fc/Nc)
    fc     : fréquence porteuse (Hz)

    Retour
    ------
    (BER, r_sym)   r_sym : symboles reçus (complexes, 1 par bit)
    """
    L = Lc * Nc           # échantillons par bit
    fs = Lc * fc          # fréquence d'échantillonnage

    # ---------- ÉMETTEUR (tout en complexe) ----------
    sym  = bpsk_map(bits)             # complexe, 1 par bit (ici Im=0)
    s_bb = nrz_encoder(sym, L)           # complexe, fs échantillons/s
    s    = to_passband(s_bb, fc, fs)  # *** RÉEL : signal sur le canal ***

    # ---------- CANAL ----------
    r = awgn(s, SNRbdB, samples_per_bit=L)        # réel + bruit réel

    # ---------- RÉCEPTEUR (retour en complexe) ----------
    r_bb_raw = to_baseband(r, fc, fs)             # complexe à partir d'ici
    r_sym    = demod_filter(r_bb_raw, L) # complexe, 1 par bit
    bits_r   = bpsk_demap(r_sym)

    BER = np.mean(bits != bits_r)

    if plots:
        _plot_chain(s_bb, s, r, r_sym, fc, fs, L, SNRbdB)

    return BER, r_sym


# =============================================================
# 4) Visualisation
# =============================================================

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


# =============================================================
# 5) Test
# =============================================================

if __name__ == "__main__":
    np.random.seed(0)
    bits = np.random.randint(2, size=10_000)

    BER, r_sym = simu_bpsk(bits, SNRbdB=8, Lc=16, Nc=1, fc=100, plots=True)
    print(f"BER simulé : {BER:.4e}")

    # Comparaison à la théorie : BER_BPSK = (1/2) erfc(sqrt(Eb/N0))
    from scipy.special import erfc
    EbN0 = 10 ** (8 / 10)
    print(f"BER théorique : {0.5 * erfc(np.sqrt(EbN0)):.4e}")