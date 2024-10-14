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
from particles import MUON, PHOTON, MUON_L
import time
import glob

rnd.seed()

plt.rcParams['text.usetex'] = True
plt.rcParams.update({'font.size': 26})

mu1 = MUON([0,0,30], 0., 0., 1000)

# 1. n(lambda) between 200-1100 nm [Huibers 1997] t of 25 deg.C
xs=np.linspace(200,1100,900)
ys=mu1._n_water(xs)
plt.figure(figsize=(14,8))
plt.plot(xs,ys)
plt.show()

# 2. make function _Cerenkov(dx, nlam)
mu1._Cerenkov()

# 3. create object photons
ph1 = PHOTON([0,0,0],0.,0.,400)

# 4. propagate muon through water
# Landau 1448 s, deterministic 12 s

h=120
mu1 = MUON([0,0,h], 0., 0., 1000)
tak=time.perf_counter()
photons=mu1.go_Cerenkov(prop_len=40, dl=1)
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
    
    ax.set_xlabel('x [cm]')
    ax.set_ylabel('y [cm]')
    ax.set_zlabel('z [cm]')
    ax.xaxis._axinfo['label']['space_factor'] = 4.0
    ax.yaxis._axinfo['label']['space_factor'] = 2.0
    ax.zaxis._axinfo['label']['space_factor'] = 2.0
    filename = 'muon_Cerenkov_3D_1track.pdf'
    plt.savefig(filename, bbox_inches='tight')
    plt.show()
    
plot_pho3d(photons, every=100)

# 6. histogram
# binning wavelengths in 9 groups
def plot_num_vs_z(photons):
    pos=[]
    for pho in photons:
        pos.append([*pho.pos[-1], pho.lamb])
    posi=np.array(pos)
    
    norm = mpl.colors.Normalize(vmin=posi[:,3].min(), vmax=posi[:,3].max())
    cmap = mpl.cm.ScalarMappable(norm=norm, cmap=mpl.cm.jet)
    cmap.set_array([])
    
    wl_bins=9
    nbins=50
    wl = np.linspace(200,1100,wl_bins+1)
    fig, axs = plt.subplots(1, 1, figsize=(10,7))
    for i in range(wl_bins):
        to_plot=posi[np.where(posi[:,3]>wl[i])[0],:]
        to_plot=to_plot[np.where(to_plot[:,3]<wl[i+1])[0],:]
        colorVal = cmap.to_rgba( (wl[i+1]+wl[i])/2 )
        
        n, bins, patches = axs.hist(to_plot[:,2], nbins,\
            density=False,fill=False,histtype='step',\
            edgecolor=colorVal, linewidth=5-0.5*i, alpha=1.0,label='%.0f - %.0f nm'\
                %(wl[i],wl[i+1]))
        
    
    n, bins, patches = axs.hist(posi[:,2], nbins, density=False,
        color='plum', alpha=0.75,label="all photons")
        
    axs.set_xlabel('z coord [cm]')
    axs.set_ylabel('number of photons')
    axs.legend(fontsize=18,loc='best',ncol=2, title="photon vacuum wavelength:")
    plt.tight_layout()
    filename = 'muon_Cerenkov_histo_num_z.pdf'
    plt.savefig(filename, bbox_inches='tight')
    plt.show()
    
plot_num_vs_z(photons)

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

nbins=75
fig, ax = plt.subplots(figsize=(7,7))
pos=[]
for pho in photons:
    pos.append([*pho.pos[-1], pho.lamb])
posi=np.array(pos)
n, bins, patches = ax.hist(posi[:,3], nbins,\
    density=False,fill=False,histtype='step',\
    edgecolor='royalblue', linewidth=3, alpha=1.0)
ax.set_xlabel('vacuum wavelength [nm]')
ax.set_ylabel('number of photons', c='royalblue')
ax2=ax.twinx()
xs=np.linspace(200,1100,900)
ys=mu1._n_water(xs)
ax2.plot(xs,ys,c='red')
ax2.set_ylabel('refractive index of water', c='r')
plt.title('1 GeV vertical muon passing 40 cm of water')
filename = 'photon_number_vs_lambda_w_title.pdf'
plt.savefig(filename, bbox_inches='tight')
plt.show()

### calib histo ######################################################
# 0.) Muon probability distribution zenith angle
muon_distro=np.loadtxt('muon_zenith.txt', skiprows=1)
bin_edge=[0]
binw=muon_distro[0,0]*2
for i in muon_distro[:,1]:
    bin_edge.append(bin_edge[-1]+binw)
    
plt.figure(figsize=(7,7))
plt.stairs(muon_distro[:,1], bin_edge, color='royalblue', linewidth=3)
# plt.plot([0,muon_distro[-2,0]],[0,muon_distro[-2,1]],c='r')
plt.xlabel('cos$^2$'+r'$\theta$')
plt.ylabel('prob')
plt.title('Zenith angle distribution of muons')
# filename = 'muon_distro.pdf'
# plt.savefig(filename, bbox_inches='tight')
plt.show()

x=np.arange(1001)/1000
y=muon_distro[-2,1]/muon_distro[-2,0]*x[1:]
# b=np.array([i+(x[-1]-x[-2])/2 for i in x])

values,indices=y, x
values=values.astype(np.float32)
weights=values/np.sum(values)

#Below, 5 is the dimension of the returned array.
n=1e6
new_random=np.random.choice(indices[1:],10000000,p=weights)

plt.figure()
plt.hist(np.arccos(np.sqrt(new_random)), 90, color='royalblue')
# plt.hist(new_random, 90, color='royalblue')
plt.show()



# 1.) propagate through tank of certain height H and radius R
H=120
mu1 = MUON([0,0,H], 10/180*np.pi, 0., 1000)
tak=time.perf_counter()
photons1=mu1.go_Cerenkov(H=H, R=180, dl=1)
tik=time.perf_counter()
print(tik-tak)

plot_pho3d(photons1, H=120, R=180)
plot_num_vs_z(photons1)

# 2.) Generate 60 muons to start
# rnd.random returns [0,1) and rnd.uniform [a,b]
nmu = 500
mus = []
H, R = 120, 180
photons_all=[]
for i in range(nmu):
    rho, alpha, z1 = R*np.sqrt(rnd.uniform(0,1)), 2*np.pi*rnd.random(),\
                        H*rnd.uniform()
    # azim, zen = 2*np.pi*rnd.random(),\
    #     np.arccos(np.sqrt(np.random.choice(indices[1:],1,p=weights)[0]))
    azim, zen = 0, 0
    
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
    
np.save('output/parent_muons_zen-0.npy', mus)

for i,thismu in enumerate(mus):      
    tak=time.perf_counter()
    # photons_all.append(thismu.go_Cerenkov(H=H, R=180, dl=1))
    photon_l=thismu.go_Cerenkov(H=H, R=180, dl=1)
    np.save('output/zen-0_%i.npy'%i, photon_l)
    tik=time.perf_counter()
    print("Finished muon ",i," took ",tik-tak," s.")

for i,pho in enumerate(photons_all):
    np.save('output/first_10_%i.npy'%i, pho)
    
photons_all_flat=[]
for i,pho in enumerate(photons_all):
    photons_all_flat.extend(pho)
    
plot_pho3d(photons_all_flat[::30], H,R)


###### coinci histo
SSD1=[20,-40,180,80] #x1,y1,x2,y2
SSD2=[-180,-40,-20,80]
H, R=150, 180

file='output_muon_distro/parent_muons.npy'
mu_list=np.load(file,allow_pickle=True)
coinci=[]
zens=[]
for i, mu in enumerate(mu_list):
    # print(mu.zen)
    zen=mu.zen
    azim=mu.azim
    x1=mu.pos[-1][0]
    y1=mu.pos[-1][1]
    z1=mu.pos[-1][2]
    l=np.sin(zen-np.pi)*np.cos(azim-np.pi)
    m=np.sin(zen-np.pi)*np.sin(azim-np.pi)
    n=np.cos(zen-np.pi)
    zens.append(mu.zen)
    
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
    if P[2]==H:
        xtop, ytop=P[0],P[1]
        if ytop>-40 and ytop<80:
            if (xtop>20 and xtop<180)or (xtop>-180 and xtop<20):
                coinci.append(i)




# files=glob.glob('output_muon_distro/round1*.npy')
# pho_nums=[]
# tak=time.perf_counter()
# for file in files:
#     ph_list=np.load(file,allow_pickle=True)
#     pho_nums.append(len(ph_list))
#     # pho_nums.extend(ph_list)
# tik=time.perf_counter()
# print(tik-tak)
# plot_pho3d(pho_nums, H=120,R=180, every=2000)
    
nbins=30
plt.figure(figsize=(7,7))
n, bins, patches = plt.hist(pho_nums, nbins,\
    density=False,fill=False,histtype='step',\
    edgecolor='royalblue', linewidth=3, alpha=1.0)
plt.xlabel('\# of photons in an event')
plt.ylabel('\# of events')
plt.title('348 1-GeV coincident muons')
filename = 'CCH_sim.pdf'
plt.savefig(filename, bbox_inches='tight')
plt.show()

ph_list=np.array([])
for file in files[:10]:
    ph_list=np.append(ph_list,np.load(file,allow_pickle=True))
    
plot_pho3d(ph_list, H, R, figsz=(10,7))

nbins=30
plt.figure(figsize=(7,7))
n, bins, patches = plt.hist(np.cos(zens)**2, nbins,\
    density=False,fill=False,histtype='step',\
    edgecolor='royalblue', linewidth=3, alpha=1.0)
plt.xlabel('zen')
plt.ylabel('\# of events')
# plt.savefig(filename, bbox_inches='tight')
plt.show()

