"""
Created on Nov 4 2024

@author: Andrea   

Python script to collect functions useful for the project 
"""
from astropy import units as u
import numpy as np

m0 = 105.6583745 * u.MeV# MeV, muon mass
me = 0.510998950 * u.MeV # MeV, electron mass 
alpha = 1. / 137.035999139

def _beta(ek,m):
    return np.sqrt(ek*ek + 2*ek*m) / (ek+m) 

def _gamma(ek,m): 
    return 1 / np.sqrt( 1 - _beta(ek,m)*_beta(ek,m) ) 

def _Wmax(_ek,m0):
    tmp_beta = _beta(_ek,m0)
    tmp_gamma = _gamma(_ek,m0)
    return 2*me*(tmp_beta*tmp_gamma)**2 / (1 + 2*tmp_gamma*me / m0 + (me / m0)**2  )


def de_dx_muon(_beta,_gamma, _Wmax, K, Z_A, I, rho ):
    return K*Z_A*(1/_beta**2) * (0.5 *np.log(2*me* _beta**2 *_gamma**2 * _Wmax / I**2) - _beta**2 ) * rho  

# Need to import delta factor values from txt file
e_delta , delta_factor = np.loadtxt('/home/workspace/Software/WCD_sim/PHYS-F-420/lesson2/delta_elec.txt', skiprows=8, usecols=(0,1), unpack=True) 

def de_dx_elec_ion_density(ek, K, Z_A, I): 
    tmp_beta = _beta(ek,me)
    tmp_gamma = _gamma(ek,me)
    delta = np.interp(ek.value,e_delta,delta_factor) # Interplate table values 
    
    return 0.5*K*Z_A/tmp_beta**2 * ( np.log(me* tmp_beta**2 * tmp_gamma**2*(me*(tmp_gamma -1)/2.)/I**2) + 
                             (1 - tmp_beta**2) - (2*tmp_gamma-1)/tmp_gamma**2 *np.log(2) + 1/8. * ((tmp_gamma-1)/tmp_gamma)**2 - delta)

def de_dx_elec_brem(ek, X0): 
    return ek/X0 

def de_dx_elec_tot(ek, K, Z_A, I, rho, X0): 
    return de_dx_elec_ion_density(ek, K, Z_A, I) * rho + de_dx_elec_brem(ek, X0)

def cherenkov_photons(beta,lam1,lam2): 
    n = 1.333
    return 2.*np.pi * alpha * (1. / lam1 - 1. / lam2) * ( 1. - 1. / (beta**2 * n**2 )) 