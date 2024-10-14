"""
Created on Mon Nov 28 15:19:12 2022

@author: katarina
"""

import numpy as np
import matplotlib.pyplot as plt
from particles import PARTICLE
plt.rcParams.update({'font.size': 12})

class PHOTON(PARTICLE):
    def __init__(self, pos, zen, azim, lam):
        c = 299792458
        h=6.62607015e-34 #J s
        e0 = 1.602176634e-19
        kE = h*c/(lam*1e-9)/e0/1e6 # kE in MeV
        super().__init__(pos, zen, azim, kE)
        self.m0 = 0 # MeV
        self.upd_E(self.kE)
        return
    
    def upd_E(self, kE):
        self.kE=kE
        self.Etot=self.kE
        self.lamb=1e9*self.h*self.c/(self.kE*1e6*self.e0)
        return
    
    # def dE_dx(self):
    #     # water properties: https://pdg.lbl.gov/2022/AtomicNuclearProperties/HTML/water_ice.html
    #     K=0.307075 #MeV mol−1 cm2
    #     z=-1
    #     Z_A=0.55509 #mol g-1
    #     rho=0.918 #g cm-3
    #     I=7.97e-5 #MeV
    #     return K*z**2*Z_A*(1/self.beta**2) \
    #         *( 0.5*np.log(2*self.me*self.beta**2*self.gamma**2*self.Wmax/I**2)\
    #                      -self.beta**2 )  * rho
    #     # returns E loss in MeV/cm
        
    # def _range(self, dx=1e-4):
    #     # ranges of muon in water: https://pdg.lbl.gov/2022/AtomicNuclearProperties/MUE/muE_water_ice.pdf
    #     e_loss=0
    #     rr=0
    #     while(self.kE>0.1):
    #         S = self.dE_dx()
    #         self.upd_E(self.kE-S*dx)
    #         e_loss+=S*dx
    #         rr+=dx
    #     return rr, e_loss #in cm
    
    # def _Eloss(self, th, dx=1e-4):
    #     # th, dx in cm!
    #     n_layer = int(th/dx) 
    #     ke_init=self.kE
    #     e_loss=0
    #     for i in range(n_layer):
    #         S = self.dE_dx()
    #         self.upd_E(self.kE-S*dx)
    #         e_loss+=S*dx
    #         if self.kE<=0.1:
    #             e_loss = ke_init
    #             break
    #     return e_loss #in MeV
    
        
        
