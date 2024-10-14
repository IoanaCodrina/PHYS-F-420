#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 28 15:25:20 2022

@author: katarina
"""
import numpy as np
import matplotlib.pyplot as plt
from particles import MUON, MUON_L, ELECTRON
import time

mu1 = MUON([0,0,0], 0., 0., 1000)
mu2 = MUON_L([0,0,0], 0., 0., 1000)

plt.rcParams['text.usetex'] = True
plt.rcParams.update({'font.size': 26})

# working in MeV (1e6 eV) and natural units
print(mu1.beta*mu1.gamma, mu1.m0/mu1.me)
# 10 MeV muon 6.998e-1 g/cm2 -> 0.76 cm
test=mu1._range() #got 0.77 cm

mu1.upd_E(100)
print(mu1.beta*mu1.gamma, mu1.m0/mu1.me)
# 100 MeV muon 31.24 g/cm2 -> 34 cm
test=mu1._range() #got 34 cm

# What is the right step size for water (in cm)?
dxs=[1e-2,1e-3,1e-4,1e-5,1e-6,5e-7,1e-7,5e-8]
# dxs=[1e-4]
# dxs=[1e-2,1e-3,1e-4,1e-5,1e-6]
e_loss=[]
timing=[time.perf_counter()]
for i in dxs:
    print("At dx = ",i)
    mu1.upd_E(1000)
    e_loss.append(mu1._Eloss(1, i))
    timing.append(time.perf_counter())
    
np.savetxt('timing2.txt', timing)
np.savetxt('e_loss2.txt', e_loss)
# timing=np.loadtxt('timing2.txt')
# e_loss=np.loadtxt('e_loss2.txt')
    
timing_diffs=[]
for i in range(len(timing)):
    timing_diffs.append(timing[i]-timing[i-1])
# timing_diffs.reverse()
# e_loss.reverse()

# In relative numbers, the difference between e_loss at 1e-5 and 1e-6 cm
# is ~3.3e-5 % (2*(e_loss[4]-e_loss[3])/(e_loss[3]+e_loss[4])).
# Between 1e-4 and 1e-6 cm: ~1.2e-6 %.

fig, ax = plt.subplots(figsize=(10,7))
ax.set_xscale('log')
ax.plot(dxs, e_loss/30, linestyle='', marker='*', c=mu1.cols[0], markersize=15)
ax.set_xlabel("dx [cm]", c='k')
ax.set_ylabel("kinetic energy loss [MeV/cm]", c=mu1.cols[0])
ax2=ax.twinx()
ax2.set_yscale('log')
ax2.plot(dxs, timing_diffs[1:], linestyle='', marker='s', c=mu1.cols[2])
ax2.set_ylabel("time [s]", c=mu1.cols[2])
plt.grid()
plt.savefig('Eloss_dx_time_accuracy.pdf', bbox_inches='tight')
plt.show()

############## timing deterministic vs stochastic

mu1.upd_E(1000)
tak=time.perf_counter()
mu1._Eloss(1)
tik=time.perf_counter()
print('1 cm passage without Landau takes ',tik-tak,' s.') # 0.3 s

mu2.upd_E(1000)
tak=time.perf_counter()
mu2._Eloss(1)
tik=time.perf_counter()
print('1 cm passage with Landau takes ',tik-tak,' s.') # 43.35 s

tak=time.perf_counter()
n_piece=30
e_loss=[]
mu1.upd_E(1000)
for i in range(n_piece):
    e_loss.append(mu1._Eloss(30/n_piece))
tik=time.perf_counter()
print('Took ',tik-tak,' s.') # 437 s

tak=time.perf_counter()
n_piece=30
e_loss2=[]
mu2.upd_E(1000)
for i in range(n_piece):
    e_loss2.append(mu2._Eloss(30/n_piece))
tik=time.perf_counter()
print('Took ',tik-tak,' s.') # 1486.51 s
    
x=np.linspace(1,len(e_loss),len(e_loss))
# plt.figure(figsize=(10,7))
# plt.scatter(x, e_loss, c='royalblue', label='deterministic')
# plt.scatter(x, e_loss2, c='red', marker='^', label='stochastic (Landau)')
# plt.xlabel('distance traveled [cm]')
# plt.ylabel('Energy lost [MeV]')
# plt.title('1 GeV muon passing through water')
# plt.legend()
# plt.savefig('e_loss_d.pdf')
# plt.show()

fig, ax = plt.subplots(figsize=(10,7))
ax.plot(x, e_loss, linestyle='', marker='*', c=mu1.cols[0], markersize=10)
ax.set_xlabel("distance traversed [cm]", c='k')
ax.set_ylabel("deterministic E loss [MeV/cm]", c=mu1.cols[0])
ax2=ax.twinx()
ax2.plot(x, e_loss2, linestyle='', marker='s', c=mu1.cols[2])
ax2.set_ylabel("stochastic E loss (Landau) [MeV/cm]", c=mu1.cols[2])
# plt.legend()
plt.savefig('e_loss_d.pdf', bbox_inches='tight')
plt.show()

############################ electrons ############################################
el1 = ELECTRON([0,0,0], 0., 0., 1000)

tak=time.perf_counter()
n_piece=30
e_loss=[]
el1.upd_E(1000)
for i in range(n_piece):
    e_loss.append(el1._Eloss(30/n_piece))
tik=time.perf_counter()
print('Took ',tik-tak,' s.') # 323.4 s

x=np.linspace(1,len(e_loss),len(e_loss))
plt.figure(figsize=(10,7))
plt.scatter(x, e_loss, c='royalblue')
plt.xlabel('distance traversed [cm]')
plt.ylabel('energy loss [MeV]')
plt.title('1 GeV electron passing through water')
# plt.legend()
plt.savefig('electron_e_loss_d.pdf')
plt.show()

tak=time.perf_counter()
n_piece=1200
e_loss=[]
el1.upd_E(1000)
for i in range(n_piece):
    ke=el1.kE
    e_lo = el1._Eloss(120/n_piece)
    e_loss.append(e_lo)
    if ke==e_lo:
        break
tik=time.perf_counter()
print('Took ',tik-tak,' s.') #  s

el_E=[1000]
for i in e_loss:
    el_E.append(el_E[-1]-i)

x=np.linspace(0,abs(el1.pos[-1][2]),len(el_E))
plt.figure(figsize=(10,7))
plt.scatter(x, el_E, c='royalblue')
plt.xlabel('distance traversed [cm]')
plt.ylabel('electron energy [MeV]')
plt.title('1 GeV electron passing through water')
# plt.legend()
plt.savefig('electron_energy_d.pdf')
plt.show()

x=np.linspace(120/n_piece,abs(el1.pos[-3][2]),len(e_loss)-2)
plt.figure(figsize=(10,7))
plt.scatter(x, 10*np.array(e_loss[:-2]), c='royalblue')
plt.xlabel('distance traversed [cm]')
plt.ylabel('energy loss [MeV/cm]')
plt.title('1 GeV electron passing through water')
# plt.legend()
plt.savefig('electron_e_loss_d.pdf')
plt.show()
