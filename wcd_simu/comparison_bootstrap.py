#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  6 12:27:05 2022

@author: katarina
"""

import numpy as np
import numpy.random as rnd
import matplotlib.pyplot as plt
import matplotlib as mpl
from particles import MUON, PHOTON, ELECTRON
import time
import glob
from scipy.stats import norm


plt.rcParams['text.usetex'] = True
plt.rcParams.update({'font.size': 26})

n_part=250
R, H = 180,120
Vtot=np.pi*R**2 *H
V_Vtot=np.linspace(0.05,0.95,46)
# R_layered=R
h_layered=H*V_Vtot
# R nested/h nested = 180/120 => R_nested=1.5*h_nested
# V=pi*R**2 *h
R_nested=(1.5/np.pi*Vtot*V_Vtot)**(1/3)
h_nested=R_nested/1.5

def bootstrap(zen):
    muons=glob.glob('output/zen-'+str(zen)+'*.npy')[:n_part]
    electrons=glob.glob('output/el_zen-'+str(zen)+'*.npy')[:n_part]    
    
    tak=time.perf_counter()
    results=[]
    for k in range(100):
        # 1 cycle took 667 s with 200 particles
        # 1. SAMPLE FROM muons
        files=[]
        for i in range(n_part):
            files.append(muons[np.random.randint(len(muons))])
        # files=muons
        # for each muon:
        total_mu_l=np.zeros(len(V_Vtot),dtype='long')
        total_mu_n=np.zeros(len(V_Vtot),dtype='long')
        # c=0
        for file in files:
            # c+=1
            # print(c)
            photons=np.load(file,allow_pickle=True)
            # make array of gamma pos
            pos=[]
            for pho in photons:
                pos.append([pho.pos[-1][0], pho.pos[-1][1],pho.pos[-1][2], pho.lamb])
            posi=np.array(pos)
            # for each V/Vtot take number of bottom segment
            for i,h in enumerate(h_layered):
                n_phot=len(np.where(posi[:,2]<h)[0])
                # add to total number
                total_mu_l[i]+=n_phot
                
            for i,h in enumerate(h_nested):
                h_accepted=posi[np.where(posi[:,2]<h)[0]]
                n_phot=len(np.where( h_accepted[:,0]**2+\
                                    h_accepted[:,1]**2<R_nested[i]**2 )[0])
                # add to total number
                total_mu_n[i]+=n_phot
        # sample from electrons
        files=[]
        for i in range(n_part):
            files.append(electrons[np.random.randint(len(electrons))])
        # files=electrons
        total_el_l=np.zeros(len(V_Vtot),dtype='long')
        total_el_n=np.zeros(len(V_Vtot),dtype='long')
        # c=0
        for file in files:
            # c+=1
            # print(c)
            photons=np.load(file,allow_pickle=True)
            # make array of gamma pos
            pos=[]
            for pho in photons:
                pos.append([pho.pos[-1][0], pho.pos[-1][1],pho.pos[-1][2], pho.lamb])
            posi=np.array(pos)
            # for each V/Vtot take number of bottom segment
            for i,h in enumerate(h_layered):
                n_phot=len(np.where(posi[:,2]<h)[0])
                # add to total number
                total_el_l[i]+=n_phot
                
            for i,h in enumerate(h_nested):
                h_accepted=posi[np.where(posi[:,2]<h)[0]]
                n_phot=len(np.where( h_accepted[:,0]**2+\
                                    h_accepted[:,1]**2<R_nested[i]**2 )[0])
                # add to total number
                total_el_n[i]+=n_phot
        # take ratio  
        # rat_l=np.divide(total_el_l,total_mu_l)
        # rat_n=np.divide(total_el_n,total_mu_n) 
        tik=time.perf_counter()
        print(k,tik-tak)
        results.append([total_el_l, total_mu_l, total_el_n, total_mu_n])
    return results

res = bootstrap(0)

np.save('result_bootstrap_zen-0.npy', res)

# plt.figure(figsize=(8,5))
# plt.plot(V_Vtot, np.divide(el_l_0,mu_l_0), linestyle='', marker='o', label='layered')
# plt.plot(V_Vtot, np.divide(el_n_0,mu_n_0), linestyle='', marker='*', label='nested')
# plt.xlabel('$V_\mathrm{segment}/V_\mathrm{total}$')
# plt.ylabel('$S_\mathrm{electrons}/S_\mathrm{muons}$')
# plt.legend()
# plt.show()

# el_l_30, mu_l_30, el_n_30, mu_n_30 = signals(30)
# el_l_60, mu_l_60, el_n_60, mu_n_60 = signals(60)

res =np.load('result_bootstrap_zen-0.npy',allow_pickle=True)
res_nested = []
for i in range(100):
    el_n=res[i][2][10]
    mu_n=res[i][3][10]
    res_nested.append(el_n/mu_n)
    
nbins=25
plt.figure(figsize=(7,7))
n, bins, patches = plt.hist(res_nested, nbins,\
    density=False,fill=False,histtype='step',\
    edgecolor='royalblue', linewidth=3, alpha=1.0)
(mean, sigma) = norm.fit(res_nested)
pdf_fitted = norm.pdf(np.linspace(np.min(res_nested),np.max(res_nested),100),\
                                    mean,sigma)
#Get bin edges
xh = [0.5 * (bins[r] + bins[r+1]) for r in range(len(bins)-1)]
#Get bin width from this
binwidth = (max(xh) - min(xh)) / len(bins)
pdf_fitted = pdf_fitted * (len(res_nested) * binwidth)
plt.plot(np.linspace(np.min(res_nested),np.max(res_nested),100), pdf_fitted,\
                'r--', linewidth=2, label=\
      'Gaussian fit\n$\mu$=%.2f$\pm$%.2f\n'\
          %(mean,sigma/np.sqrt(len(res_nested)))\
       +'$\sigma$=%.2f$\pm$%.2f' %(sigma,sigma/np.sqrt(2*len(res_nested))))

plt.xlabel('$S_\mathrm{e}/S_\mathrm{\mu}$ nested')
plt.ylabel('number of resamples')
plt.legend()
plt.savefig('bootstrap_histo.pdf', bbox_inches='tight')
plt.show()

def plot_ratio(S_layered, err_S_layered, S_nested, err_S_nested):
    plt.figure(figsize=(8,5))
    plt.errorbar(V_Vtot, S_layered, yerr=err_S_layered, linestyle='', marker='_',\
             c='royalblue', label='layered', capsize=2)
    plt.errorbar(V_Vtot, S_nested, yerr=err_S_nested, linestyle='', marker='_',\
             c='red', label='nested', capsize=2)
    plt.xlabel('$V_\mathrm{segment}/V_\mathrm{total}$')
    plt.ylabel('$S_\mathrm{e}/S_\mathrm{\mu}$')
    plt.legend()
    plt.savefig('bootstrap_ratios_zen-0.pdf', bbox_inches='tight')
    plt.show()

res =np.load('result_bootstrap_zen-0.npy',allow_pickle=True)
S_l, err_S_l, S_n, err_S_n = [],[],[],[]
for j,_ in enumerate(V_Vtot):
    S_nested, S_layered = [], []
    for i in range(100):
        el_l=res[i][0][j]
        mu_l=res[i][1][j]
        el_n=res[i][2][j]
        mu_n=res[i][3][j]
        S_nested.append(el_n/mu_n)
        S_layered.append(el_l/mu_l)
    (mean, sigma) = norm.fit(S_nested)
    S_n.append(mean)
    err_S_n.append(sigma)
    (mean, sigma) = norm.fit(S_layered)
    S_l.append(mean)
    err_S_l.append(sigma)
    
plot_ratio(S_l, err_S_l, S_n, err_S_n)
        