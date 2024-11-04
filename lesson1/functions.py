"""
Created on Nov 4 2024

@author: Andrea   

Python script to collect functions useful for the project 
"""
from astropy import units as u
import numpy as np

m0 = 105.6583745 * u.MeV# MeV, muon mass
me = 0.510998950 * u.MeV # MeV, electron mass 

def _beta(ek):
    return np.sqrt(ek*ek + 2*ek*m0) / (ek+m0) 

def _gamma(ek): 
    return 1 / np.sqrt( 1 - _beta(ek)*_beta(ek) ) 

def _Wmax(_ek):
    tmp_beta = _beta(_ek)
    tmp_gamma = _gamma(_ek)
    return 2*me*(tmp_beta*tmp_gamma)**2 / (1 + 2*tmp_gamma*me / m0 + (me / m0)**2  )


def de_dx(_beta,_gamma, _Wmax, K, Z_A, I, rho ):
    return K*Z_A*(1/_beta**2) * (0.5 *np.log(2*me* _beta**2 *_gamma**2 * _Wmax / I**2) - _beta**2 ) * rho  