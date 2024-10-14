"""
Created on Mon Nov 28 15:19:12 2022

@author: katarina
"""

import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 12})

class PARTICLE:
    def __init__(self, pos, zen, azim, kE):
        self.pos=np.array([pos])
        self.c = 299792458 #m/s
        self.zen = zen
        self.azim = azim
        self.kE = kE
        self.cols = ['r', 'orange', 'royalblue']
        self.me = 0.510998950 #MeV/c**2
        self.alpha = 1/137.035999139
        self.h=6.62607015e-34 #J s
        self.e0 = 1.602176634e-19 #C
        return
    
    def upd_pos(self, pos):
        self.pos=np.concatenate((self.pos,[pos]))
        return
    
    def reset_pos_history(self):
        self.pos=np.array([self.pos[-1]])
        return
    
    def delete_pos_history(self):
        self.pos=np.array([self.pos[0]])
        return
        
    def upd_dir(self, zen, azim):
        self.zen = zen
        self.azim = azim
        return
    
    def _n_water(self, lam):
        """
        Calculates refractive index of water given wavelenght in vacuum.

        Parameters
        ----------
        lam : float
            Photon wavelenght in vacuum in nm.

        Returns
        -------
        float
            Refractive index of water for photon at given wavelenght.

        """
        return 1.31279 + 15.762/lam - 4382/lam**2 + 1.1455e6/lam**3
        
        
