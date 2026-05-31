# -*- coding: utf-8 -*-
"""
Created on Mon Feb  9 13:32:28 2026

@author: arthu
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
from scipy.stats import norm
from reedsolo import RSCodec, ReedSolomonError
from scipy.special import erfc

#%% Fonctions de bases

d_bit_par_symb={"bpsk_map":1,"qpsk_map":2,"ask_map":1,"_16qam_map":4}
d_nom_mod={"bpsk_map":"BPSK","qpsk_map":"QPSK","ask_map":"ASK","_16qam_map":"16-QAM"}

def Q(x):
    return 0.5*erfc(np.sqrt(0.5)*x)

def dB_to_dec(SNRbdB):    
    return 10 ** (SNRbdB / 10)

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

def awgn(s, SNRbdB, bit_par_symb,L, plot=False):
    """Ajoute un bruit AWGN selon Eb/N0=SNRb.
    """
    SNRb = 10 ** (SNRbdB / 10)
    Ps = np.mean(np.abs(s) ** 2)
    Es = Ps*L #energie par symbole
    Eb = Es / bit_par_symb
    N0 = Eb / SNRb
    sigma = np.sqrt(N0 / 2)
    
    n = sigma * np.random.randn(len(s))
    
    if plot :
        _plot_awgn_hist(n,sigma,SNRbdB,L)
    
    return s + n

def nakagami_channel(s,L, m=1):
    """
    Canal de Nakagami a partir des données du doc Narrowband Channel Measurements and Statistical Characterization
    """
    Nb=len(s)//L
    p_fading = np.random.gamma(shape=m, scale=1.0/m, size=Nb) #puissance fading
    h=np.repeat(np.sqrt(p_fading), L)[:len(s)]
    # L'enveloppe d'amplitude h est la racine carrée de la puissance 
    #####a expliquer

    return s * h

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
    

#%% Mapping 16 QAM
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
        
        I = GRAY_MAP[(b0, b1)] #2 premiers bits sur l'axe I
        Q = GRAY_MAP[(b2, b3)] #les 2 suivants sur l'axe Q
        
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

def simu_canal_lin(bk,b2s,s2b,SNRbdB,Lc=16, Nc=1, fc=100, faded=False, m=0, rs=False,plots=False):
    """
    Paramètres
    ----------
    bk   : array de 0/1
    SNRbdB : Eb/N0 en dB
    Lc     : échantillons par période porteuse
    Nc     : nb de périodes de porteuse par bit (durée 1 bit, Rb = fc/Nc)
    fc     : fréquence porteuse (Hz)

    Retour
    ------
    (BER, bk_r)   bk_r : bits reçus
    """
    L=Lc*Nc #échantillons par bit
    fs=Lc*fc #fréquence d'échantillonnage
    
    bk_original = bk.copy()
    
    if rs : 
        bk,rempli = encod_rs(bk)
    
    ak = b2s(bk) #Mapping
    s_bb = nrz_encoder(ak, L)  #signal complexe en bb sur-échantillonée
    s = to_passband(s_bb, fc, fs)  #signal réel mod dans le canal

    # ---------- CANAL ----------
    if faded : 
        s = nakagami_channel(s,L, m=m)
    
    r = awgn(s, SNRbdB, bit_par_symb=d_bit_par_symb.get(b2s.__name__),L=L)
    # ---------------------------
    
    r_bb_raw = to_baseband(r, fc, fs) #signal complexe en bb
    r_sym = demod_filter(r_bb_raw, L) #filtre intégrateur
    bk_r = s2b(r_sym)
    
    if rs : 
        bk_r,error = decod_rs(bk_r,rempli)

    BER = np.mean(bk_original != bk_r)

    if plots:
        _plot_chain(s_bb, s, r, r_sym, fc, fs, L, SNRbdB, b2s)

    if rs : 
        return BER, bk_r,error
    return BER, bk_r


#%%Mod BFSK

def simu_canal_bfsk(bk, SNRbdB, Lc=16, Nc=1, fc=100,faded=False,m=0,rs=False, plots=False):

    L=Lc*Nc
    fs=Lc*fc
    df=fc/Nc
    f0,f1=fc, fc+df
    
    bk_original = bk.copy()
    
    if rs : 
        bk,rempli = encod_rs(bk)
    
    t = np.arange(len(bk) * L) / fs
    f_inst = [f0 if k==0 else f1 for k in bk]
    f_inst = np.repeat(f_inst,L)
    s = np.cos(2 * np.pi * f_inst * t)

    if faded : 
        s = nakagami_channel(s,L, m=m)
    r = awgn(s, SNRbdB, bit_par_symb=1,L=L)

    y0 = demod_filter(r * np.cos(2 * np.pi * f0 * t), L)
    y1 = demod_filter(r * np.cos(2 * np.pi * f1 * t), L)
    bk_r = (y1 > y0).astype(int)
    
    if rs : 
        bk_r,error = decod_rs(bk_r,rempli)

    BER = np.mean(bk_original != bk_r)

    if plots:
        _plot_chain_bfsk(s, r, y0, y1, fc, df, fs, L, SNRbdB)

    if rs : 
        return BER, bk_r,error
    return BER, bk_r


def _plot_chain_bfsk(s, r, y0, y1, fc, df, fs, L, SNRbdB):
    n = min(10 * L, len(s))
    t = np.arange(n) / fs
    _plot_signal(t, s[:n], r"Signal passband émis $s(t)$ (BFSK)", "s(t)")
    _plot_signal(t, r[:n], f"Signal reçu $r(t)$, $E_b/N_0$ = {SNRbdB} dB", "r(t)")

    #graph constellation bfsk
    plt.figure(figsize=(5, 5))
    plt.scatter(y0, y1, s=8, alpha=0.4)
    lim = max(np.max(np.abs(y0)), np.max(np.abs(y1))) * 1.1
    plt.plot([-lim, lim], [-lim, lim], 'r--', label=r'frontière $y_0 = y_1$')
    plt.axhline(0, color='k', lw=0.5); plt.axvline(0, color='k', lw=0.5)
    plt.xlabel(r"$y_0 = \langle r,\,\cos(2\pi f_0 t)\rangle$")
    plt.ylabel(r"$y_1 = \langle r,\,\cos(2\pi f_1 t)\rangle$")
    plt.title(f"Constellation BFSK projetée, "+ r"$SNR_{b,dB}$" + f" = {SNRbdB} dB")
    plt.grid(True)
    plt.axis('equal')
    plt.legend()
    plt.show()
    plt.close()

    _plot_spectrum(s, fs, fmax=2.5 * fc + df, title="Spectre du signal BFSK passband")

#%% Reed-Solomon
def bits_to_bytes(bk):
    remplissage = (8 - len(bk) % 8) % 8
    bk_complet = list(bk) + [0] * remplissage #on complete avec des 0 pour avoir un multiple de 8
    octets = bytearray()
    for i in range(0, len(bk_complet), 8):
        byte = 0
        for bit in bk_complet[i:i+8]:
            byte = (byte << 1) | int(bit) #constuit un octet bit par bit 
        octets.append(byte)
    return bytes(octets), remplissage

def bytes_to_bits(bk, remplissage=None):
    bits = []
    for byte in bk:
        for i in range(7, -1, -1):
            bits.append((byte >> i) & 1) #on reconstruit les bits depuis un octet
    
    if remplissage is not None and remplissage > 0:
        bits = bits[:-remplissage]
    return np.array(bits)


def encod_rs(bk):
    msg,rempli = bits_to_bytes(bk)
    msg_encod=RSCodec(32).encode(msg)
    bits_encod=bytes_to_bits(msg_encod)
    return bits_encod,rempli

def decod_rs(bk_r, rempli, nsym=32, nsize=255):
    """On décode le msg recu bloc par bloc de 255 symboles/bits??? 
    et si un bloc est irréparable on laisse sa version erronnée"""
    msg_r, _ = bits_to_bytes(bk_r)
    rsc = RSCodec(nsym, nsize=nsize)
    decoded = bytearray()
    n_fail = 0

    for i in range(0, len(msg_r), nsize):
        block = bytes(msg_r[i:i+nsize])
        try:
            decoded.extend(rsc.decode(block)[0])
        except ReedSolomonError:
            n_fail += 1
            # bloc non corrigible : on garde les octets de données reçus bruts
            data_len = max(0, len(block) - nsym)
            decoded.extend(block[:data_len])

    bk_r_decod = bytes_to_bits(bytes(decoded), remplissage=rempli)
    return bk_r_decod, n_fail

#%% Graphiques

def _plot_signal(t, s, title, ylabel="", legend=False):
    plt.figure(figsize=(8, 3))
    plt.plot(t, s.real, label='Composante Q')
    if not(np.allclose(s.imag, 0)):
        plt.plot(t, s.imag, '--r', label='Composante I')
    plt.xlabel("Temps (s)"); plt.ylabel(ylabel); plt.title(title)
    if legend :
        plt.legend(loc='upper right')    
    plt.grid(True)
    plt.show(); plt.close()


def _plot_constellation(sym, title="Constellation"):
    plt.figure(figsize=(5, 5))
    plt.scatter(sym.real, sym.imag, s=8, alpha=0.4)
    plt.axhline(0, color='k', lw=0.5); plt.axvline(0, color='k', lw=0.5)
    plt.xlabel("I (en-phase)"); plt.ylabel("Q (quadrature)")
    plt.title(title)
    plt.grid(True); plt.axis('equal')
    plt.show(); plt.close()


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
    plt.title(title); plt.grid(True)
    plt.show(); plt.close()


def _plot_chain(s_bb, s, r, r_sym, fc, fs, L, SNRbdB,b2s):
    n = min(10 * L, len(s))
    t = np.arange(n) / fs
    _plot_signal(t, s_bb[:n], f"{d_nom_mod[b2s.__name__]} - Enveloppe complexe","Amplitude", legend=True)
    _plot_signal(t, s[:n], f"{d_nom_mod[b2s.__name__]} - Signal passband émis $s(t)$ (réel)","s(t)")
    _plot_signal(t, r[:n], f"{d_nom_mod[b2s.__name__]} - Signal reçu $r(t)$,"+r"$SNR_{b,dB}$"+f" = {SNRbdB} dB","r(t)")
    _plot_constellation(r_sym, f"{d_nom_mod[b2s.__name__]} - Constellation,"+r" $SNR_{b,dB}$ "+f"= {SNRbdB} dB")
    _plot_spectrum(s, fs, fmax=2.5 * fc, title=f"{d_nom_mod[b2s.__name__]} - Spectre du signal passband")


def _plot_awgn_hist(bruit, sigma, SNR_dB, L):
    
    plt.figure(figsize=(10, 6))
    count, bins, ignored = plt.hist(bruit, bins=100, density=True, alpha=0.6, color='blue', edgecolor='black', label='Bruit généré')

    x = np.linspace(-4*sigma, 4*sigma, 1000)
    
    densi_th = norm.pdf(x, loc=0, scale=sigma)
    
    plt.plot(x, densi_th, 'r-', linewidth=2.5, label=f'Loi Gaussienne Théorique\n($\mu=0$, $\sigma={sigma:.4f}$)')
    
    plt.title(f'Modèle du générateur AWGN (SNR = {SNR_dB} dB, L = {L})', fontsize=12)
    plt.xlabel('Amplitude du bruit', fontsize=12)
    plt.ylabel('Densité de probabilité', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(loc='upper right', fontsize=12)
    plt.show();plt.close()


#%% Simu BPSK

bk = np.random.randint(2, size=100_000)
bk[0]=1
bk[1]=0
bk[2]=1
bk[3]=1
bk[4]=0
bk[5]=1
bk[6]=0
BER, bk_r = simu_canal_lin(bk,bpsk_map,bpsk_demap, SNRbdB=20, Lc=32, Nc=1, fc=100, plots=True)
print(f"BER : {BER:.4e}")
#%% Simu QPSK

bk = np.random.randint(2, size=100_000)
BER, bk_r = simu_canal_lin(bk,qpsk_map,qpsk_demap, SNRbdB=20, Lc=32, Nc=1, fc=100, plots=True)
print(f"BER : {BER:.4e}")
#%% Simu ASK

bk = np.random.randint(2, size=100_000)
BER, bk_r = simu_canal_lin(bk,ask_map,ask_demap, SNRbdB=4, Lc=32, Nc=1, fc=100, plots=True)
print(f"BER : {BER:.4e}")
#%%Simu 16QAM

bk = np.random.randint(2, size=100_000)
BER, _  = simu_canal_lin(bk, _16qam_map, _16qam_demap, SNRbdB=200, Lc=32,Nc=1,fc=100, plots=True)
print(f"BER : {BER:.4e}")
#%%Simu BFSK

bk = np.random.randint(2, size=100_000)
BER, bk_r = simu_canal_bfsk(bk, SNRbdB=10, Lc=16, Nc=1, fc=100, plots=True)
print(f"BER BFSK : {BER:.4e}")

#%%Simulations comparatives

rng = np.random.default_rng()

h_simu={'BPSK':{'MAP':bpsk_map,'DEMAP':bpsk_demap},
        'QPSK':{'MAP':qpsk_map,'DEMAP':qpsk_demap},
        'ASK':{'MAP':ask_map,'DEMAP':ask_demap},
        '16QAM':{'MAP':_16qam_map,'DEMAP':_16qam_demap}
        }

def simu(l_SNRbdB,B_SIZE,fc=100,Lc=16,Nc=1,MOY=1,faded=False,m=1,rs=False):
    
    BER_bpsk,BER_qpsk,BER_ask,BER_bfsk,BER_16qam = [],[],[],[],[]
    
    for _SNRbdB in l_SNRbdB:
        moy={'BPSK':0,'QPSK':0,'ASK':0,'16QAM':0,'BFSK':0}
        for i in range(MOY):
            bk = rng.integers(0, 2, size=B_SIZE, dtype=np.int8)
            for mod in ['BPSK','QPSK','ASK','16QAM']:
                moy[mod]+=float(simu_canal_lin(bk,h_simu[mod]['MAP'],h_simu[mod]['DEMAP'],SNRbdB=_SNRbdB,Lc=Lc,Nc=Nc,fc=fc,faded=faded,m=m,rs=rs,plots=False)[0])
            moy['BFSK']+=float(simu_canal_bfsk(bk, SNRbdB=_SNRbdB, Lc=Lc,Nc=Nc,fc=fc,faded=faded,m=m,rs=rs,plots=False)[0])
            
            print(i)
        
        for mod in moy:
            moy[mod]/=MOY
        
        BER_bpsk.append(moy['BPSK'])
        BER_qpsk.append(moy['QPSK'])
        BER_ask.append(moy['ASK'])
        BER_16qam.append(moy['16QAM'])
        BER_bfsk.append(moy['BFSK'])
        print(_SNRbdB)
    
    return {'l_SNRbdB':l_SNRbdB,
            'B_SIZE':B_SIZE,
            'fc':fc,
            'Lc':Lc,
            'Nc':Nc,
            'MOY':MOY,
            'faded':faded,
            'm':m,
            'rs':rs,
            'BER':{'BPSK':BER_bpsk,
                   'QPSK':BER_qpsk,
                   'ASK':BER_ask,
                   '16QAM':BER_16qam,
                   'BFSK':BER_bfsk
                   }
            }

#%%BER theoriques AWGN
def simu_th(l_SNRbdB):

    BER_bpsk_th = [Q(np.sqrt(2*dB_to_dec(_SNRbdB))) for _SNRbdB in l_SNRbdB]
    BER_qpsk_th = BER_bpsk_th[:]
    BER_ask_th = [Q(np.sqrt(dB_to_dec(_SNRbdB))) for _SNRbdB in l_SNRbdB]
    BER_16qam_th = [0.75*Q(np.sqrt(0.8*dB_to_dec(_SNRbdB))) for _SNRbdB in l_SNRbdB]
    BER_bfsk_th = BER_ask_th[:]
    
    return {'l_SNRbdB':l_SNRbdB,
            'B_SIZE':None,
            'fc':None,
            'Lc':None,
            'Nc':None,
            'MOY':None,
            'faded':None,
            'm':None,
            'rs':None,
            'BER':{'BPSK':BER_bpsk_th,
                   'QPSK':BER_qpsk_th,
                   'ASK':BER_ask_th,
                   '16QAM':BER_16qam_th,
                   'BFSK':BER_bfsk_th
                   }
            }


l_SNRbdB = range(-4,25,1)
fc=100
MOY=1
B_SIZE=1_000_000
Lc=16
Nc=1

#%%
    
#nom_simu:{(keys = l_snrbdb,ber_...,fc,moy,_b_size,m,faded,rs,Lc,Nc)}
d_simu = {}

d_simu['TH']=simu_th(np.arange(-4,20,0.1))

#d_simu['AWGN'] = simu(l_SNRbdB,B_SIZE,fc=fc,Lc=Lc,Nc=Nc,MOY=MOY)

#d_simu['NAGA-1'] = simu(l_SNRbdB,B_SIZE,fc=fc,Lc=Lc,Nc=Nc,MOY=MOY,faded=True,m=1)
d_simu['NAGA-2.5'] = simu(l_SNRbdB,B_SIZE,fc=fc,Lc=Lc,Nc=Nc,MOY=MOY,faded=True,m=2.5)

#d_simu['NAGA-0.5'] = simu(l_SNRbdB,B_SIZE,fc=fc,Lc=Lc,Nc=Nc,MOY=MOY,faded=True,m=0.5)
#d_simu['NAGA-0.75'] = simu(l_SNRbdB,B_SIZE,fc=fc,Lc=Lc,Nc=Nc,MOY=MOY,faded=True,m=0.75)



#%%


l_SNRbdB_rs = list(range(-4,20,1)) + list(map(lambda x:float(round(x,ndigits=2)),np.arange(4.2,10,0.2)))
d_simu['RS'] = simu(l_SNRbdB_rs,B_SIZE,fc=fc,Lc=Lc,Nc=Nc,MOY=MOY,rs=True)


l_SNRbdB_rs_naga = list(range(-4,20,1)) + list(map(lambda x:float(round(x,ndigits=2)),np.arange(7,14,0.2)))
d_simu['RS+NAGA-2.5'] = simu(l_SNRbdB_rs_naga,B_SIZE,fc=fc,Lc=Lc,Nc=Nc,MOY=MOY,rs=True, faded=True,m=2.5)

#%%SIMU NAGA 2.5 PROBLEME
bk = rng.integers(0, 2, size=1_000_000, dtype=np.int8)
BER, _  = simu_canal_lin(bk, _16qam_map, _16qam_demap, SNRbdB=20, Lc=32,Nc=1,fc=100, plots=False, faded=True, m=2.5)
print(BER)
#%%



#Ajouter la bsup ###################################


#Couleur + forme des points
color_p = {'BPSK':'+b',
       'QPSK':'xg',
       'ASK':'+r',
       '16QAM':'xm',
       'BFSK':'+c'
       }

#Couleur + forme des courbes
color_c = {'BPSK':'-.b',
       'QPSK':'--g',
       'ASK':'--r',
       '16QAM':'--m',
       'BFSK':'-.c'
       }
#%%Graphe BER AWGN
for mod,ber in d_simu['AWGN']['BER'].items():
    plt.plot(d_simu['AWGN']['l_SNRbdB'],ber, color_p[mod])

for mod,ber in d_simu['TH']['BER'].items():
    plt.plot(d_simu['TH']['l_SNRbdB'],ber, color_c[mod],label=mod,lw=0.7,alpha=0.8)

plt.xlabel(r"$SNR_{b,dB}$"); plt.ylabel(r"$BER$")
plt.title(r"$BER$ théoriques et simulés en fonction du $SNR_{b,dB}$")
plt.text(0.05, 0.15, r"Canal: $AWGN$"+"\n"+f"Nb bits: {d_simu['AWGN']['B_SIZE']}", 
         transform=plt.gca().transAxes, 
         fontsize=10, 
         verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.2))
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
#Ajouter en petit en haut/bas les autres parametres de la simu
plt.yscale('log')
plt.xticks(range(-4,21,2))
plt.ylim(1e-6, 1)
plt.show(); plt.close()

#%%Graphe BER AWGN+Nagakami

m=2.5

for mod,ber in d_simu[f'NAGA-{m}']['BER'].items():
    plt.plot(d_simu[f'NAGA-{m}']['l_SNRbdB'],ber, color_p[mod])

for mod,ber in d_simu['TH']['BER'].items():
    plt.plot(d_simu['TH']['l_SNRbdB'],ber, color_c[mod],label=mod,lw=0.7,alpha=0.8)

plt.xlabel(r"$SNR_{b,dB}$"); plt.ylabel(r"$BER$")
plt.title(r"$BER$ théoriques et simulés en fonction du $SNR_{b,dB}$")
plt.text(0.05, 0.15, r"Canal: $AWGN$ + "+f"Nagakami-{m}"+"\n"+f"Nb bits: {d_simu['NAGA-2.5']['B_SIZE']}", 
         transform=plt.gca().transAxes, 
         fontsize=10, 
         verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.2))
#Ajouter en petit en haut/bas les autres parametres de la simu
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
plt.yscale('log')
plt.xticks(range(-4,21,2))
plt.ylim(1e-6, 1)
plt.show(); plt.close()

#%%Graphe BER AWGN+Nagakami+RS

m=2.5

for mod,ber in d_simu['RS+NAGA-2.5']['BER'].items():
    plt.plot(d_simu['RS+NAGA-2.5']['l_SNRbdB'],ber, color_p[mod])

for mod,ber in d_simu['TH']['BER'].items():
    plt.plot(d_simu['TH']['l_SNRbdB'],ber, color_c[mod],label=mod,lw=0.7,alpha=0.8)

plt.xlabel(r"$SNR_{b,dB}$"); plt.ylabel(r"$BER$")
plt.title(r"$BER$ théoriques et simulés en fonction du $SNR_{b,dB}$")
plt.text(0.05, 0.15, r"Canal: $AWGN$ + "+f"Nagakami-{m}"+ r" $RS$" + "\n"+f"Nb bits: {d_simu['NAGA-1']['B_SIZE']}", 
         transform=plt.gca().transAxes, 
         fontsize=10, 
         verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.2))
#Ajouter en petit en haut/bas les autres parametres de la simu
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
plt.yscale('log')
plt.xticks(range(-4,21,2))
plt.ylim(1e-6, 1)
plt.show(); plt.close()


#%%Graphe BER AWGN+RS
for mod,ber in d_simu['RS']['BER'].items():
    plt.plot(d_simu['RS']['l_SNRbdB'],ber, color_p[mod])

for mod,ber in d_simu['TH']['BER'].items():
    plt.plot(d_simu['TH']['l_SNRbdB'],ber, color_c[mod],label=mod,lw=0.7,alpha=0.8)

plt.xlabel(r"$SNR_{b,dB}$"); plt.ylabel(r"$BER$")
plt.title(r"$BER$ théoriques et simulés en fonction du $SNR_{b,dB}$")
plt.text(0.05, 0.15, r"Canal: $AWGN$ + $RS$"+"\n"+f"Nb bits: {d_simu['RS']['B_SIZE']}", 
         transform=plt.gca().transAxes, 
         fontsize=10, 
         verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.2))
#Ajouter en petit en haut/bas les autres parametres de la simu
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
plt.yscale('log')
plt.xticks(range(-4,21,2))
plt.ylim(1e-6, 1)
plt.show(); plt.close()

#%%
import pickle

with open(r"C:\Users\arthu\Documents\GitHub\tipe\d_simu3.pkl", "wb") as f:
    pickle.dump(d_simu, f)


#%%
import pickle
with open(r"C:\Users\arthu\Documents\GitHub\tipe\d_simu3.pkl", "rb") as f:
    d_simu = pickle.load(f)


