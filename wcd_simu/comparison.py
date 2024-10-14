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


plt.rcParams['text.usetex'] = True
plt.rcParams.update({'font.size': 26})

n_part=500
R, H = 180,120
Vtot=np.pi*R**2 *H
V_Vtot=np.linspace(0.05,0.95,46)
# R_layered=R
h_layered=H*V_Vtot
# R nested/h nested = 180/120 => R_nested=1.5*h_nested
# V=pi*R**2 *h
R_nested=(1.5/np.pi*Vtot*V_Vtot)**(1/3)
h_nested=R_nested/1.5

def signals(zen):
    muons=glob.glob('output/zen-'+str(zen)+'*.npy')[:n_part]
    electrons=glob.glob('output/el_zen-'+str(zen)+'*.npy')[:n_part]    
    
    tak=time.perf_counter()
    # 1 cycle took 667 s with 200 particles
    # 1. SAMPLE FROM muons
    # files=[]
    # for i in range(n_part):
    #     files.append(muons[np.random.randint(len(muons))])
    files=muons
    # for each muon:
    total_mu_l=np.zeros(len(V_Vtot),dtype='long')
    total_mu_n=np.zeros(len(V_Vtot),dtype='long')
    c=0
    for file in files:
        c+=1
        print(c)
        photons=np.load(file,allow_pickle=True)
        # make array of gamma pos
        pos=[]
        for pho in photons:
            pos.append([*pho.pos[-1], pho.lamb])
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
    # files=[]
    # for i in range(n_part):
    #     files.append(electrons[np.random.randint(len(electrons))])
    files=electrons
    total_el_l=np.zeros(len(V_Vtot),dtype='long')
    total_el_n=np.zeros(len(V_Vtot),dtype='long')
    c=0
    for file in files:
        c+=1
        print(c)
        photons=np.load(file,allow_pickle=True)
        # make array of gamma pos
        pos=[]
        for pho in photons:
            pos.append([*pho.pos[-1], pho.lamb])
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
    rat_l=np.divide(total_el_l,total_mu_l)
    rat_n=np.divide(total_el_n,total_mu_n) 
    tik=time.perf_counter()
    print(tik-tak)
    return [total_el_l, total_mu_l, total_el_n, total_mu_n]

el_l_0, mu_l_0, el_n_0, mu_n_0 = signals(0)
el_l_30, mu_l_30, el_n_30, mu_n_30 = signals(30)
el_l_60, mu_l_60, el_n_60, mu_n_60 = signals(60)

np.save('zen0counts.npy', [el_l_0, mu_l_0, el_n_0, mu_n_0])
np.save('zen30counts.npy', [el_l_30, mu_l_30, el_n_30, mu_n_30])
np.save('zen60counts.npy', [el_l_60, mu_l_60, el_n_60, mu_n_60])

def plot_ratio(el_l_0, mu_l_0, el_n_0, mu_n_0, zen):
    plt.figure(figsize=(8,5))
    plt.plot(V_Vtot, np.divide(el_l_0,mu_l_0), linestyle='', marker='o',\
             c='royalblue', label='layered')
    plt.plot(V_Vtot, np.divide(el_n_0,mu_n_0), linestyle='', marker='^',\
             c='red', label='nested')
    plt.xlabel('$V_\mathrm{segment}/V_\mathrm{total}$')
    plt.ylabel('$S_\mathrm{e}/S_\mathrm{\mu}$')
    plt.legend()
    plt.savefig('ratios_zen-'+str(zen)+'.pdf', bbox_inches='tight')
    plt.show()
    
plot_ratio(el_l_0, mu_l_0, el_n_0, mu_n_0, 0)



plot_ratio(el_l_30, mu_l_30, el_n_30, mu_n_30, 30)
plot_ratio(el_l_60, mu_l_60, el_n_60, mu_n_60, 60)

el_l_all=np.add(el_l_0,np.add(el_l_30, el_l_60))
el_n_all=np.add(el_n_0,np.add(el_n_30, el_n_60))
mu_l_all=np.add(mu_l_0,np.add(mu_l_30, mu_l_60))
mu_n_all=np.add(mu_n_0,np.add(mu_n_30, mu_n_60))

plot_ratio(el_l_all, mu_l_all, el_n_all, mu_n_all, 'all')
