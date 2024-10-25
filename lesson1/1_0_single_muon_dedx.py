#!/usr/bin/python3

"""
Created on Nov 24 2024

@author: Andrea, starting from Katarina's work    

Consider a relativistic muon crossing a WCD vertically. 
The energy deposit is firstly computed from the Bethe formula, 
then considering a Landau distribution.     
"""

import numpy as np  

# Define relevant physical constants   
m0 = 105.6583745 # MeV, muon mass
me = 0.510998950 # MeV, electron mass 

# Define WCD cylinder tank dimensions 
h = 120 #cm 
r = 180 #cm 

# Choose kinetic energy of the muon 
ek = 1000. # 1000 MeV = 1 GeV 

# Compute beta and gamma for the particle 

p =  np.sqrt(ek*ek + 2*ek*m0)

def _beta(ek):
    return np.sqrt(ek*ek + 2*ek*m0) / (ek+m0) 
def _gamma(ek): 
    return 1 / np.sqrt( 1 - _beta(ek)*_beta(ek) ) 

beta = _beta(ek)
gamma = _gamma(ek)

print('\n***** Starting values for the muon *****')
print('p = ',p,' MeV  c^-1') 
print('beta = ',_beta(ek))
print('gamma = ',_gamma(ek)) 
print('beta * gamma =  ', _beta(ek)*_gamma(ek),'\n')

# For 0.1 < beta*gamma < 1000, the mean energy loss rate of a heavy charged particle in a medium is described by the Bethe equation 
# Start preparing the "ingredients" for the Bethe equation

# First, compute maximum energy transfer to an electron of target material in a single collision 

def _Wmax(_ek):
    tmp_beta = _beta(_ek)
    tmp_gamma = _gamma(_ek)
    return 2*me*(tmp_beta*tmp_gamma)**2 / (1 + 2*tmp_gamma*me / m0 + (me / m0)**2  )

print('***** Maximum energy transfer to an electron for starting muon: ***** ')
print('Wmax = ',_Wmax(ek) / 1e3,' GeV\n' ) 

# Then, look up material properties (water)
# https://pdg.lbl.gov/2024/AtomicNuclearProperties/HTML/water_liquid.html  

Z_A = 0.55509 #mol g^-1, atomic number Z over atomic mass mol g^-1 
rho = 1 #g cm^-3, density of liquid water 
I = 7.97e-5 #MeV, mean excitation energy

K = 0.307075 #MeV mol^−1 cm2 

# Define the Bethe formula as a funcion, with the muon kinetic energy as argument. Dimension of dE/dx is MeV/cm 

def de_dx(_ek):
    tmp_beta = _beta(_ek)
    tmp_gamma = _gamma(_ek) 
    return K*Z_A*(1/tmp_beta**2) * (0.5 *np.log(2*me* tmp_beta**2 *tmp_gamma**2 * _Wmax(_ek) / I**2) - tmp_beta**2 ) * rho  

# Compute energy loss according to the Bethe formula assuming dE/dx constant across full tank 
 
 
e_loss = de_dx(ek) * h 

print('*****  Energy loss across the tank with constant dE/dx *****')
print(e_loss, 'MeV')

# Compute "G" coefficient 

G_dx = 2*me / (beta**2 * _Wmax(ek)) * 0.15 * Z_A
print(0.01 / G_dx) 

# Try splitting the muon path inside the tank into smaller steps of dx = 10 cm
# Update energy of muon after each step  

print('\n*****  Computing dE/dx in steps across the tank *****\n')


def _eloss_step(_ek,_dx): 
    _eloss = de_dx(_ek) * _dx 
    print('Step energy loss: ', _eloss, 'MeV')
    return _eloss

dx = 10 #cm 

# As a first check, compute energy loss for a dx = 10 cm step in water  
eloss = _eloss_step(ek,dx)

# Update kinetic energy, beta and gamma
ek = ek - eloss  
beta = _beta(ek)
gamma = _gamma(ek)  
print('Muon energy after one step: ', ek,'\n')

# Now compute energy loss with nsteps to cover the full height of the tank (h = 120 cm )

print('*****  Now iterate steps for the full height of the tank *****\n')

ek = 1000. # re-set the energy to 1 GeV  

nsteps = int(h/dx)
tot_eloss = 0.

# Initialize arrays to save energy loss and beta*gamma at each step 
eloss_array = np.zeros(nsteps)
bg_array = np.zeros(nsteps)

for i in range(nsteps): 
    eloss = _eloss_step(ek, dx)
    tot_eloss += eloss
     
    ek = ek - eloss  
    
    eloss_array[i] = eloss / (dx * rho)
    bg_array[i] = _beta(ek) * _gamma(ek)
    
print('\nMuon kinetic energy after passage in the tank: ', ek, 'MeV')    
print('Total energy deposited in the tank:', tot_eloss, 'MeV')
print('Check: ', ek + tot_eloss )

# Plot dE/dx as a function of beta*gamma 

import matplotlib.pyplot as plt

# Choose directory where to save plots   
outdir = './plots/' 

plt.figure(1)
plt.xscale('log')
plt.xlim(6,10.5)
plt.xlabel(r'$\beta \gamma$', size = 15)
plt.ylabel(r'dE/dx (MeV g$^{-1}$ cm$^{2}$)', size =  15)
plt.scatter(bg_array, eloss_array)

plt.savefig(outdir+'bethe_formula.eps',format = 'eps',bbox_inches='tight')

# In this way we have computed the energy loss based on the mean rate given by the Bethe formula
# Now let's take into account the energy loss probability distribution, described by a Landau distribution   
# Install landaupy with "pip install git+https://github.com/SengerM/landaupy"

from landaupy import landau

# Define function to compute energy loss from a Landau distribution at each step 
 
def _eloss_step_landau(_ek,_dx):
    beta = _beta(_ek)
    gamma = _gamma(_ek) 
    
    # Width of the Landau 
    width =  K/2*Z_A*(_dx*rho/beta**2) 
    
    # MPV of the Landau 
    mpv = width \
            *( np.log(2*me*beta**2*gamma**2/I) \
                +np.log(width/I) + 0.2
                         -beta**2 )
    
    # Extract a value from the Landau distribution         
    _eloss = landau.sample(x_mpv=mpv, xi=width, n_samples=1) 
    
    print('Step energy loss: ', _eloss, 'MeV')
    return _eloss

# Now let's simulate steps inside the tank, each time extracting the energy loss from the Landau distribution

print('\n*****  Iterate steps for the full height of the tank *****')
print('*****  computing energy loss from a Landau distribution *****\n')


ek = 1000. # reset primary kinetic energy 

d_eloss_array = np.zeros(nsteps)
tot_eloss_landau = 0.
 
for i in range(nsteps): 
    eloss = _eloss_step_landau(ek,dx)
    tot_eloss_landau += eloss 
    d_eloss_array[i] = eloss
    tot_eloss += eloss
     
    ek = ek - eloss  
    
    
print('\nMuon kinetic energy after passage in the tank: ', ek, 'MeV')    
print('Total energy deposited in the tank:', tot_eloss_landau, 'MeV')

# Now plot histogram of energy losses at each step 

plt.figure(2)
plt.xlabel(r'E$_{loss}$ (MeV)', size = 14)
plt.ylabel('Entries', size = 14)
plt.hist(d_eloss_array, bins = np.linspace(10,40,30), histtype='stepfilled')
plt.savefig(outdir+'de_dx_landau.eps',format = 'eps',bbox_inches='tight')
    
