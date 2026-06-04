# -*- coding: utf-8 -*-
"""
Created on Tue May 26 20:57:40 2026

@author: arthu
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats


#Récupération données
d, v = [], []
with open(r'C:\Users\arthu\Documents\GitHub\tipe\res_tipe_exp.csv') as f:
    next(f)  # saute l'en-tête
    for line in f:
        parts = line.strip().split('\t')
        if len(parts) >= 2:
            try:
                d.append(float(parts[0].replace(',', '.')))
                v.append(float(parts[1].replace(',', '.')))
            except: pass
        
d, v = np.array(list(reversed(d))), np.array(list(reversed(v)))

#%%Graph points mesurés
plt.scatter(d,v,s=1)
plt.xlabel('Distance à la source (cm)')
plt.ylabel(r'$V_{pp}$ (mV)')
plt.title(r'$V_{pp}$ fonction de la distance - 214 points, pas=2mm')
plt.grid(alpha=0.3)
plt.show()

#%%Calcul large scale loss
#modèle papier : L_dB(d) = A + 10*n*log_10(d) + X_sigma
#or L_dB(d) = 20*log10(v)

#la reg lin fait que le + X_sigma shadowing vaut 0 car suit loi normale

v_dB = 20*np.log10(v) #passage en dB
log_d = np.log10(d)

pente, A, r, p, se = stats.linregress(log_d, v_dB)

n = -pente/10
n_incertitude = (se/10)

#Calcul path shadowing
X_sigma = v_dB - (A + pente*log_d)
s_shadowing = np.std(X_sigma)
print(f"Sigma shadowing = {s_shadowing:.1f} dB")

plt.semilogx(d, v_dB, '.', alpha=0.6, label='Mesures')
x = np.linspace(17.4, 60, 100)
plt.semilogx(x, A + pente*np.log10(x), 'r-', label=f'n = {n:.2f}, incertitude = {n_incertitude:.2f}\nA = {A:.1f}\n'+r'$X_{sigma}$ = '+f'{s_shadowing:.1f} dB')

plt.xlabel('Distance (cm)')
plt.ylabel(r'$20log_{10}(V_pp)$ (dB-mV)')
plt.legend()
plt.grid(alpha=0.3, which='both')
plt.title(r'$L_{dB}(d) = A + 10*n*log_{10}(d) + X_{sigma}$')
plt.show()


#%%Calcul de m


V_modele = 10**((A + pente*log_d)/20)

#On normalise l'enveloppe autour de V_tendance -> on retire le path loss et omega env =1
alpha = v / V_modele

# floc : translation loi
m, _, omeg = stats.nakagami.fit(alpha, floc=0)
Omega = omeg**2
print(f"m estimé = {m:.2f},  Omega = {Omega:.3f}")

#Calcul densité de proba théorique 
x = np.linspace(0.01, alpha.max()*1.05, 300)
distrib_calc = stats.nakagami.pdf(x, m, loc=0, scale=omeg)
distrib_rayleigh = stats.nakagami.pdf(x, 1, loc=0, scale=np.sqrt(Omega))

#%%Affichage graph

bar = np.linspace(0, alpha.max()*1.05, 30) #30 bar sur l'histogramme
plt.hist(alpha, bins=bar, density=True, alpha=0.5, color='lightblue',edgecolor='blue', label=f'Mesures (N={len(alpha)})')

x = np.linspace(0.01, alpha.max()*1.05, 300)
plt.plot(x, distrib_calc, 'r-', label=f'Nakagami m={m:.2f}')
plt.plot(x, distrib_rayleigh, 'g--', label='Rayleigh (m=1)')

plt.xlabel(r'Enveloppe normalisée')
plt.ylabel('Densité de probabilité')
plt.title("Distribution de l'enveloppe du signal")
plt.legend(); plt.grid(alpha=0.3)

plt.show()
