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


def cherenkov_photons_array(beta,dx):
    n_lam_bins = 61 
    lam_bin_width = 10
    n = 1.333
    lambda_points = np.arange(200,800,lam_bin_width) * u.nm 
    lambda_low = lambda_points[:-1] 
    lambda_hi = lambda_points[1:] 
    lambda_centers = (lambda_low + lambda_hi) / 2

    return [lambda_low,lambda_hi], 2.*np.pi * alpha * (1. / lambda_low - 1. / lambda_hi) * ( 1. - 1. / (beta**2 * n**2 )) * dx.to(u.nm)



# Define a class that describes a 3D vector  

class Vector():
    
    def __init__(self,x0,y0,z0,zenith,azimuth):
        self.x = x0
        self.y = y0
        self.z = z0
        self.theta = zenith 
        self.phi = azimuth
        self.r = np.sqrt(self.x*self.x + self.y*self.y)
        
    def update_pos(self,step_size): 
        self.x = self.x + step_size * np.sin(self.theta)*np.cos(self.phi)  
        self.y = self.y + step_size * np.sin(self.theta)*np.sin(self.phi)
        self.z = self.z + step_size * np.cos(self.theta)
        self.r = np.sqrt(self.x*self.x + self.y*self.y)
        
    def in_tank(self):
        r_tank = 180 * u.cm
        h_tank = 120 * u.cm 
        if self.z >= 0. * u.cm  and self.z <= h_tank and self.r <= r_tank: 
            return True 
        else: 
            return False 
        
    def rnd_ch_photon_pos(self, N_ph): 
        rnd_step = np.random.uniform(0,1,size=N_ph)
        rnd_x = self.x.value + rnd_step * np.sin(self.theta)*np.cos(self.phi) 
        rnd_y = self.y.value + rnd_step * np.sin(self.theta)*np.sin(self.phi)
        rnd_z = self.z.value + rnd_step * np.cos(self.theta)
        rnd_r = np.sqrt(rnd_x*rnd_x + rnd_y*rnd_y)
        return rnd_r, rnd_z
