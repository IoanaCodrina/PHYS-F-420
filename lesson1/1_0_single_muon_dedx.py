#!/usr/bin/python3

"""
Created on Oct 24 2024

@author: Andrea, starting from Katarina's work    

Consider a relativistic muon crossing a WCD vertically. 
The energy deposit is firstly computed from the Bethe formula, 
then considering a Landau distribution.     
"""

import numpy as np  
from astropy import units as u

# Define relevant physical constants   
m0 = 105.6583745 * u.MeV # MeV, muon mass
me = 0.510998950 * u.MeV # MeV, electron mass 

# # Define WCD cylinder tank dimensions  
h = 120 * u.cm #cm 
r = 180 * u.cm #cm 

# Choose kinetic energy of the muon 
ek = 1000.* u.MeV # 1000 MeV = 1 GeV 

# Compute beta and gamma for the particle 

p =  np.sqrt(ek*ek + 2*ek*m0)
def _beta(ek):
    return np.sqrt(ek*ek + 2*ek*m0) / (ek+m0) 
def _gamma(ek): 
    return 1 / np.sqrt( 1 - _beta(ek)*_beta(ek) )  

beta = _beta(ek)
gamma = _gamma(ek)

print('\n***** Starting values for the muon *****')
print('p = ',p) 
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

Wmax = _Wmax(ek)

print('***** Maximum energy transfer to an electron for starting muon: ***** ')

# Example of how to use units to convert a quantity 
print('Wmax = ',Wmax.to(u.GeV) ) 

# Look up material properties (water)
# https://pdg.lbl.gov/2024/AtomicNuclearProperties/HTML/water_liquid.html  

Z_A=0.55509 *u.mol / u.g # mol g^-1, atomic number Z over atomic mass mol g^-1 
rho=1 * u.g / u.cm**3 # g cm^-3, density of liquid water 
I=7.97e-5 * u.MeV # MeV, mean excitation energy

K=0.307075 * u.MeV * u.cm **2 / u.mol # MeV mol^−1 cm2

# Define the Bethe formula as a funcion, with the muon kinetic energy as argument. Dimension of dE/dx is MeV/cm 

def de_dx(_beta,_gamma, _Wmax):
    return K*Z_A*(1/_beta**2) * (0.5 *np.log(2*me* _beta**2 *_gamma**2 * _Wmax / I**2) - _beta**2 ) * rho  

# Check correct implementation of the Bethe formula 
import matplotlib.pyplot as plt

# Check correct implementation of the Bethe formula 
ek_values = np.logspace(-1,4,100)
ek_values = ek_values * u.MeV 

beta_values = _beta(ek_values)
gamma_values = _gamma(ek_values) 
Wmax_values = _Wmax(ek_values)

# Choose directory where to save plots   
outdir = './plots/' 

plt.figure(1)
plt.scatter(ek_values, de_dx(beta_values, gamma_values, Wmax_values).value, c='black', marker='.')
plt.ylabel('dE/dx (MeV/cm) ', size=16)
plt.xlabel(r'E$_k$ (MeV) ', size=16)

plt.xscale('log')
plt.yscale('log')
plt.savefig(outdir+'bethe_formula_vs_ek.svg',format = 'svg',bbox_inches='tight')

plt.figure(2)
plot =  plt.scatter(beta_values*gamma_values, de_dx(beta_values, gamma_values, Wmax_values).value, c='black', marker='.')
plt.ylabel('dE/dx [MeV/cm ] ', size=16)
plt.xlabel(r'$\beta \gamma$ ', size=16)

plt.xscale('log')
plt.yscale('log')
plt.savefig(outdir+'bethe_formula_vs_betagamma.svg',format = 'svg',bbox_inches='tight')

# Compute energy loss according to the Bethe formula assuming dE/dx constant across full tank 

e_loss = de_dx(beta,gamma,Wmax) * h 
print('*****  Energy loss across the tank with constant dE/dx *****')
print(e_loss)

# Compute "G" coefficient 

G_dx = 2*me / (beta**2 * _Wmax(ek)) * 0.15 * Z_A
print(0.01 / G_dx) 

# Try splitting the muon path inside the tank into smaller steps of dx = 10 cm
# Update energy of muon after each step  

print('*****  Computing dE/dx in steps across the tank *****\n')

def _eloss_step(_beta,_gamma,_Wmax,_dx): 
    _eloss = de_dx(_beta,_gamma,_Wmax) * _dx 
    print('Step energy loss: ', _eloss)
    return _eloss

dx = 10 * u.cm #cm 

# As a first check, compute energy loss for a dx = 10 cm step in water  

# Initialize again 1 GeV muon
ek = 1000 * u.MeV # MeV
beta = _beta(ek)
gamma = _gamma(ek)
Wmax = _Wmax(ek)

eloss = _eloss_step(beta,gamma,Wmax,dx)

# Update kinetic energy, beta and gamma
ek = ek - eloss  
beta = _beta(ek)
gamma = _gamma(ek)  
Wmax = _Wmax(ek)
print('Muon energy after one step: ', ek)

# Now compute energy loss with nsteps to cover the full height of the tank (h = 120 cm )

# Initialize again 1 GeV muon

ek = 1000. * u.MeV # re-set the energy to 1 GeV  
beta = _beta(ek)
gamma = _gamma(ek)
Wmax = _Wmax(ek)

nsteps = int(h/dx)
tot_eloss = 0. * u.MeV

# Initialize arrays to save energy loss and beta*gamma at each step 
eloss_array = np.zeros(nsteps) * u.MeV * u.cm**2 / u.g
bg_array = np.zeros(nsteps)

for i in range(nsteps): 
    eloss = _eloss_step(beta, gamma, Wmax, dx)
    tot_eloss += eloss
     
    # Update particle 
    ek = ek - eloss  
    beta = _beta(ek)
    gamma = _gamma(ek)
    Wmax = _Wmax(ek)

    eloss_array[i] = eloss / (dx * rho)
    bg_array[i] = beta * gamma
    
print('\nMuon kinetic energy after passage in the tank: ', ek)    
print('Total energy deposited in the tank:', tot_eloss)
print('Check: ', ek + tot_eloss )


# Plot dE/dx as a function of beta*gamma 

plt.figure(3)
plt.xscale('log')
plt.xlim(6,10.5)
plt.xlabel(r'$\beta \gamma$', size = 15)
plt.ylabel(r'dE/dx (MeV g$^{-1}$ cm$^{2}$)', size =  15)
plt.scatter(bg_array, eloss_array)

plt.savefig(outdir+'bethe_formula_steps.svg',format = 'svg',bbox_inches='tight')

# In this way we have computed the energy loss based on the mean rate given by the Bethe formula
# Now let's take into account the energy loss probability distribution, described by a Landau distribution   
# Install landaupy with "pip install git+https://github.com/SengerM/landaupy"

from landaupy import landau

# Define function to compute energy loss from a Landau distribution at each step 
 
def _eloss_step_landau(_beta,_gamma,_dx):
    
    # Width of the Landau 
    width =  K/2*Z_A*(_dx*rho/_beta**2) 
    
    # MPV of the Landau 
    mpv = width \
            *( np.log(2*me*_beta**2*_gamma**2/I) \
                +np.log(width/I) + 0.2
                         -_beta**2 )
    
    # Extract a value from the Landau distribution         
    _eloss = landau.sample(x_mpv=mpv.value, xi=width.value, n_samples=1) 
    
    print('Step energy loss: ', _eloss, 'MeV')
    return _eloss * u.MeV

# Now let's simulate steps inside the tank, each time extracting the energy loss from the Landau distribution

print('\n*****  Iterate steps for the full height of the tank *****')
print('*****  computing energy loss from a Landau distribution *****\n')

ek = 1000. * u.MeV # reset primary kinetic energy 
beta = _beta(ek)
gamma = _gamma(ek)
Wmax = _Wmax(ek)

d_eloss_array = np.zeros(nsteps) * u.MeV
tot_eloss_landau = 0. * u.MeV 
 
for i in range(nsteps): 
    eloss = _eloss_step_landau(beta,gamma,dx)
    tot_eloss_landau += eloss 
    d_eloss_array[i] = eloss
    tot_eloss += eloss
     
    # Update particle 
    ek = ek - eloss  
    beta = _beta(ek)
    gamma = _gamma(ek)
    Wmax = _Wmax(ek)
    
print('\nMuon kinetic energy after passage in the tank: ', ek)    
print('Total energy deposited in the tank:', tot_eloss_landau)

# Now plot histogram of energy losses at each step 

plt.figure(4)
plt.xlabel(r'E$_{loss}$ (MeV)', size = 14)
plt.ylabel('Entries', size = 14)
plt.hist(d_eloss_array.value, bins = np.linspace(10,40,30), histtype='stepfilled')
plt.savefig(outdir+'de_dx_landau.svg',format = 'svg',bbox_inches='tight')
    
