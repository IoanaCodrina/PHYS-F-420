#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  4 16:15:31 2023

@author: katarina
"""

import numpy as np
import numpy.random as rnd
import matplotlib.pyplot as plt
import matplotlib as mpl
from particles import MUON, PHOTON
import time

rnd.seed()

plt.rcParams['text.usetex'] = True
plt.rcParams.update({'font.size': 26})

nmu = 500
mus = []
H, R = 120, 180
photons_all=[]
for i in range(nmu):
    rho, alpha, z1 = R*np.sqrt(rnd.uniform(0,1)), 2*np.pi*rnd.random(),\
                        H*rnd.uniform()
    # azim, zen = 2*np.pi*rnd.random(),\
    #     np.arccos(np.sqrt(np.random.choice(indices[1:],1,p=weights)[0]))
    # azim, zen = 0, 0
    azim, zen = 2*np.pi*rnd.random(), 60/180*np.pi
    
    x1=rho*np.cos(alpha)
    y1=rho*np.sin(alpha)
    l=np.sin(zen-np.pi)*np.cos(azim-np.pi)
    m=np.sin(zen-np.pi)*np.sin(azim-np.pi)
    n=np.cos(zen-np.pi)
    
    if zen==0:
        za=H
        xa=l/n*(H-z1)+x1
        ya=m/n*(H-z1)+y1
        P=[xa,ya,za]
        
    else:
        a=l**2/m**2+1
        b=2*l/m*x1-2*l**2/m**2*y1
        c=l**2/m**2*y1**2-2*l/m*x1*y1+x1**2-R**2
        
        ya=(-b+np.sqrt(b**2-4*a*c) )/(2*a)
        yb=(-b-np.sqrt(b**2-4*a*c) )/(2*a)
        za=n/m*(ya-y1)+z1
        zb=n/m*(yb-y1)+z1
        # print('Intersection with infinite cyl: ',za,zb)
        if za>zb:
            if za>H:
                za=H
                xa=l/n*(H-z1)+x1
                ya=m/n*(H-z1)+y1
                P=[xa,ya,za]
            else:
                xa=l/m*(ya-y1)+x1
                P=[xa,ya,za]
        else:
            if zb>H:
                zb=H
                xb=l/n*(H-z1)+x1
                yb=m/n*(H-z1)+y1
                P=[xb,yb,zb]
            else:
                xb=l/m*(yb-y1)+x1
                P=[xb,yb,zb]
    # need to repair rounding error in x,y coord if z!=120
    if (not(P[2]==H)) and (np.sqrt(P[0]**2+P[1]**2)>R):
        P[0]=P[0]-1e-5
        P[1]=P[1]-1e-5
        
    # print(np.sqrt(P[0]**2+P[1]**2),P[2])
    mus.append(MUON(P, zen, azim, 1000) )
    
np.save('output/parent_muons_zen-60.npy', mus)

for i,thismu in enumerate(mus):
    f=open('log.txt', 'a+')     
    tak=time.perf_counter()
    # photons_all.append(thismu.go_Cerenkov(H=H, R=180, dl=1))
    photon_l=thismu.go_Cerenkov(H=H, R=180, dl=1)
    np.save('output/zen-60_%i.npy'%i, photon_l)
    tik=time.perf_counter()
    f.write("Finished muon %i took %.2f s.\n"%(i, tik-tak))
    f.close()