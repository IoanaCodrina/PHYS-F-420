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
from particles import ELECTRON
import time

rnd.seed()

plt.rcParams['text.usetex'] = True
plt.rcParams.update({'font.size': 26})

# el1 = ELECTRON([0,0,30], 0., 0., 1000)

# 4. propagate ELECTRON through water
# deterministic 36.7 s (40 cm) 80.33 (100 cm)

h=120
el1 = ELECTRON([0,0,h], 0., 0., 1000)
el1.photons=[]
tak=time.perf_counter()
el1.go_Cerenkov(prop_len=100)
tik=time.perf_counter()
print(tik-tak)

# 5. plot
def data_for_cylinder_along_z(center_x,center_y,radius,height_z):
    z = np.linspace(0, height_z, 50)
    theta = np.linspace(0, 2*np.pi, 50)
    theta_grid, z_grid=np.meshgrid(theta, z)
    x_grid = radius*np.cos(theta_grid) + center_x
    y_grid = radius*np.sin(theta_grid) + center_y
    return x_grid,y_grid,z_grid

def plot_pho3d(photons,H=0,R=0, figsz=(10,7), every=1000):
    pos=[]
    for pho in photons:
        pos.append([*pho.pos[-1], pho.lamb])
    posi=np.array(pos)
    
    norm = mpl.colors.Normalize(vmin=posi[:,3].min(), vmax=posi[:,3].max())
    cmap = mpl.cm.ScalarMappable(norm=norm, cmap=mpl.cm.jet)
    cmap.set_array([])
    
    fig = plt.figure(figsize=figsz)
    ax = fig.add_subplot(projection='3d')
    
    for n,i in enumerate(posi[::every]):
        colorVal = cmap.to_rgba(i[3])
        ax.plot(i[0],i[1],i[2], marker='.', c=colorVal, markersize=2)
    cbar = fig.colorbar(cmap)
    cbar.set_label('vacuum wavelength [nm]',fontsize=26)
    
    if H>0:
        Xc,Yc,Zc = data_for_cylinder_along_z(0,0,R,H)
        ax.plot_surface(Xc, Yc, Zc, alpha=0.2)
        ax.set_ylim(-R,R)
        ax.set_xlim(-R,R)
        ax.set_zlim(0,H)
    else:
        ax.set_ylim(-40,40)
        ax.set_xlim(-40,40)
    
    # ax.set_xlabel('x [cm]')
    # ax.set_ylabel('y [cm]')
    # ax.set_zlabel('z [cm]')
    ax.xaxis._axinfo['label']['space_factor'] = 4.0
    ax.yaxis._axinfo['label']['space_factor'] = 2.0
    ax.zaxis._axinfo['label']['space_factor'] = 2.0
    filename = 'electron_Cerenkov_3D.pdf'
    plt.savefig(filename, bbox_inches='tight')
    plt.show()

photons=el1.photons
plot_pho3d(photons, every=100)

# 6. histogram
# binning wavelengths in 9 groups
def plot_num_vs_z(photons):
    pos=[]
    for pho in photons:
        pos.append([*pho.pos[-1], pho.lamb])
    posi=np.array(pos)
    
    # ran=[np.min(posi[:,2])-0.5,np.max(posi[:,2])+0.5]
    # ran=[24.7,26]
    ran=[24,28]
    
    norm = mpl.colors.Normalize(vmin=posi[:,3].min(), vmax=posi[:,3].max())
    cmap = mpl.cm.ScalarMappable(norm=norm, cmap=mpl.cm.jet)
    cmap.set_array([])
    
    wl_bins=9
    nbins=120
    wl = np.linspace(200,1100,wl_bins+1)
    fig, axs = plt.subplots(1, 1, figsize=(10,7))
    for i in range(wl_bins):
        to_plot=posi[np.where(posi[:,3]>wl[i])[0],:]
        to_plot=to_plot[np.where(to_plot[:,3]<wl[i+1])[0],:]
        colorVal = cmap.to_rgba( (wl[i+1]+wl[i])/2 )
        
        n, bins, patches = axs.hist(to_plot[:,2], nbins, range=ran,\
            density=False,fill=False,histtype='step',\
            edgecolor=colorVal, linewidth=5-0.5*i, alpha=1.0,label='%.0f - %.0f nm'\
                %(wl[i],wl[i+1]))
        
    n, bins, patches = axs.hist(posi[:,2], nbins, range=ran, density=False,
        color='plum', alpha=0.75,label="all photons")
        
    axs.set_xlabel('z coord [cm]')
    axs.set_ylabel('number of photons')
    axs.legend(fontsize=18,loc='best',ncol=2, title="photon vacuum wavelength:")
    plt.tight_layout()
    filename = 'electron_1GeV_Cerenkov_histo_zoomed.pdf'
    plt.savefig(filename, bbox_inches='tight')
    plt.show()
    
plot_num_vs_z(el1.photons) # photon finished at z=24.716

# number of photons as function of wavelength
# nbins=75

# pos=[]
# for pho in photons:
#     pos.append([*pho.pos[-1], pho.lamb])
# posi=np.array(pos)
# plt.figure(figsize=(7,7))
# n, bins, patches = plt.hist(posi[:,3], nbins,\
#     density=False,fill=False,histtype='step',\
#     edgecolor='royalblue', linewidth=3, alpha=1.0)
# plt.xlabel('vacuum wavelength [nm]')
# plt.ylabel('number of photons')
# plt.title('1 GeV vertical muon passing 40 cm of water')
# filename = 'photon_number_vs_lambda.pdf'
# plt.savefig(filename, bbox_inches='tight')
# plt.show()

# 1.) propagate through tank of certain height H and radius R
H=120
R=180
el1 = ELECTRON([0,0,H], 0, 0., 0.1)
tak=time.perf_counter()
el1.go_Cerenkov(H=H, R=180)
tik=time.perf_counter()
print(tik-tak)

# plot_pho3d(el1.photons, H, R)
plot_num_vs_z(el1.photons)



