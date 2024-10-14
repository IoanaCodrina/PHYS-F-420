"""
Created on Mon Nov 28 15:19:12 2022

@author: katarina
"""

import numpy as np
import matplotlib.pyplot as plt
from particles import PARTICLE, PHOTON
plt.rcParams.update({'font.size': 12})

class ELECTRON(PARTICLE):
    def __init__(self, pos, zen, azim, kE):
        super().__init__(pos, zen, azim, kE)
        self.m0 = self.me # MeV
        self.upd_E(self.kE)
        self.v=self._v()
        self.lams=np.linspace(200,1100,901)
        self.cer_pho=np.zeros(len(self.lams))
        self.photons=[]
        return
    
    def upd_E(self, kE):
        self.kE=kE
        self.Etot=np.sqrt(self.kE**2+self.m0**2)
        self.beta=self.f_beta()
        self.gamma=1/np.sqrt(1-self.beta**2)
        self.Wmax = self.f_Wmax()
        self.v=self._v()
        return
    
    def _v(self): #need to update angles if adding scaterring
        v=self.beta*self.c
        return [v*np.sin(self.zen-np.pi)*np.cos(self.azim-np.pi), \
                v*np.sin(self.zen-np.pi)*np.sin(self.azim-np.pi),\
                    v*np.cos(self.zen-np.pi)]
    
    def f_beta(self):
        #   for kE in MeV
        beta=np.sqrt(1- 1/( self.kE/self.m0 +1)**2 )
        return beta
    
    def f_Wmax(self):
        # if m0 >> me
        Wmax=2*self.me*self.beta**2*self.gamma**2/\
            (1+2*self.gamma*self.me/self.m0+(self.me/self.m0)**2 )
        return Wmax
    
    def dE_dx(self):
        # water properties: https://pdg.lbl.gov/2022/AtomicNuclearProperties/HTML/water_ice.html
        K=0.307075 #MeV mol−1 cm2
        Z_A=0.55509 #mol g-1
        rho=0.918 #g cm-3
        I=7.97e-5 #MeV
        X0=36.08 # g/cm2
        return (self.kE/X0 + 0.5*K*Z_A*(1/self.beta**2) \
            *( np.log(self.me*self.beta**2*self.gamma**2*\
                      (self.me*(self.gamma-1)/2)/I**2)\
                         +(1-self.beta**2) -\
                             ((2*self.gamma-1)*np.log(2)/self.gamma**2) +\
                                 1/8*((self.gamma-1)/self.gamma)**2 ) ) * rho
        # returns E loss in MeV/cm
        
    
    def _Eloss_inf(self, th, H=0, R=0, dx=1e-4):
        n_layer = int(th/dx) 
        ke_init=self.kE
        e_loss=0
        err_code=0
        for i in range(n_layer):
            S = self.dE_dx()
            dt=dx*1e-2/np.linalg.norm(np.array(self.v))
            new_pos=np.add(np.array(self.pos[-1]),100*dt*np.array(self.v))
            if (new_pos[2]<0) or (new_pos[2]>H) or\
                (new_pos[0]**2+new_pos[1]**2>R**2):
                err_code=2
                break
            self.upd_pos(new_pos)
            self.upd_E(self.kE-S*dx)
            e_loss+=S*dx
            if self.kE<=0.1:
                e_loss = ke_init
                err_code=1
                break
        return e_loss, err_code
    
    def _Eloss(self, th, dx=1e-4):
        # th, dx in cm!
        n_layer = int(th/dx) 
        ke_init=self.kE
        e_loss=0
        for i in range(n_layer):
            S = self.dE_dx()
            dt=dx*1e-2/np.linalg.norm(np.array(self.v))
            self.upd_pos(np.add(np.array(self.pos[-1]),100*dt*np.array(self.v)))
            self.upd_E(self.kE-S*dx)
            e_loss+=S*dx
            if self.kE<=0.1:
                e_loss = ke_init
                break
        return e_loss #in MeV
    
    def _Cerenkov(self, dx=1e-4, nlam=901):
        dxnm=dx*1e7
        lams=np.linspace(200,1100,nlam)
        dlam=lams[-1]-lams[-2]
        ns=self._n_water(lams)
        N_mean=2*np.pi*self.alpha*(1/(lams-dlam/2)-1/(lams+dlam/2))*\
                  (1 - 1/(self.beta**2*ns**2) )*dxnm
        # if particle speed lower than speed of light at given lambda N_mean<0
        N_mean[np.where(N_mean<0)[0]]=0
        return lams, N_mean
    
    def go_Cerenkov(self, H=0, R=0, prop_len=0, dl=2e-4):
        if prop_len>0:
            for k in range(int(prop_len/dl)):
                self.reset_pos_history()
                ke=self.kE
                e_loss = self._Eloss(dl/2)
                if ke==e_loss:
                    break
                pho_lam, pho_num = self._Cerenkov(dx=dl)
                ke=self.kE
                e_loss = self._Eloss(dl/2)
                if ke==e_loss:
                    break
                self.cer_pho=np.add(self.cer_pho, pho_num)
                ind=np.where(self.cer_pho>=1.0)[0]
                lams=pho_lam[ind]
                for i,lam in enumerate(lams):
                    for j in range(int(np.round(self.cer_pho[ind[i]]))):
                        self.photons.append(PHOTON(self.pos[-1], 0,0, lam))
                        pho_eneMev=self.h*self.c/(lam*1e-9)/self.e0*1e-6
                        self.upd_E(self.kE-pho_eneMev)
                    self.cer_pho[ind[i]]=0.
        elif R>0 and H>0:
            while True:
                self.reset_pos_history()
                ke=self.kE
                e_loss,err=self._Eloss_inf(dl/2, R=R, H=H)
                if ke==e_loss:
                    print('Absorbed at ', self.pos[-1])
                    break
                if err!=0:
                    print('Finishing propagation at ', self.pos[-1])
                    # print(dl/2)
                    # print(np.linalg.norm(np.subtract(self.pos[-1],self.pos[0])))
                    break
                pho_lam, pho_num = self._Cerenkov(dl)
                ke=self.kE
                e_loss,err=self._Eloss_inf(dl/2, R=R, H=H)
                if ke==e_loss:
                    print('Absorbed at ', self.pos[-1])
                    break
                if err!=0:
                    print('Finishing propagation at ', self.pos[-1])
                    # print(dl/2)
                    # print(np.linalg.norm(np.subtract(self.pos[-1],self.pos[0])))
                    break
                self.cer_pho=np.add(self.cer_pho, pho_num)
                ind=np.where(self.cer_pho>=1.0)[0]
                lams=pho_lam[ind]
                for i,lam in enumerate(lams):
                    for j in range(int(np.round(self.cer_pho[ind[i]]))):
                        self.photons.append(PHOTON(self.pos[-1], 0,0, lam))
                        pho_eneMev=self.h*self.c/(lam*1e-9)/self.e0*1e-6
                        self.upd_E(self.kE-pho_eneMev)
                    self.cer_pho[ind[i]]=0.
                        
            dl=np.linalg.norm(np.subtract(self.pos[-1],self.pos[0]))
            print('Last piece is',dl,' cm.')
            self.delete_pos_history()
            ke=self.kE
            e_loss = self._Eloss(dl/2)
            if ke==e_loss:
                return
            pho_lam, pho_num = self._Cerenkov(dl)
            ke=self.kE
            e_loss = self._Eloss(dl/2)
            if ke==e_loss:
                return
            self.cer_pho=np.add(self.cer_pho, pho_num)
            ind=np.where(self.cer_pho>=1.0)[0]
            lams=pho_lam[ind]
            for i,lam in enumerate(lams):
                for j in range(int(np.round(self.cer_pho[ind[i]]))):
                    self.photons.append(PHOTON(self.pos[-1], 0,0, lam))
                    pho_eneMev=self.h*self.c/(lam*1e-9)/self.e0*1e-6
                    self.upd_E(self.kE-pho_eneMev)
                self.cer_pho[ind[i]]=0.
        return
        
        
