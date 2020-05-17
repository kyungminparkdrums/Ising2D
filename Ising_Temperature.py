""" 
Monte Carlo simulation for the Ising model in 2D with the Metropolis algorithm.
Class written by JaeDong Noh
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time
import os

class IsingModel2D():
    """ Ising model object """
    def __init__(self, lsize=32, beta=1.0, bfield=0.0):
        self.lsize = lsize # lateral size
        self.beta = beta   # inverse temperature
        self.bfield = bfield
        self.n_total = self.lsize*self.lsize
        # spin configuration as a 1-dim array
        self.spin = np.ones(lsize*lsize, dtype=int)
        self.spin = 2*(np.random.random(lsize*lsize)<0.5)-1
        tmp_arr = np.array(range(lsize*lsize)).reshape((lsize, lsize))
        self.up = np.roll(tmp_arr, -1, axis=0).flatten()
        self.dw = np.roll(tmp_arr, +1, axis=0).flatten()
        self.lft = np.roll(tmp_arr, +1, axis=1).flatten()
        self.rgt = np.roll(tmp_arr, -1, axis=1).flatten()
        '''
        self.even = np.array( ['True'] * self.n_total, dtype=bool)
        self.odd = np.array( ['True'] * self.n_total, dtype=bool) 
        for s in range(lsize*lsize):
            if (s//lsize)%2 == (s%lsize)%2 :
                self.even[s] = True
                self.odd[s] = False
            else :
                self.even[s] = False
                self.odd[s] = True
        ''' 
        self.even = np.zeros((self.lsize, self.lsize), dtype=bool)
        self.even[::2, ::2] = True
        self.even[1::2, 1::2] = True
        self.even = self.even.flatten()
        self.odd = np.ones((self.lsize, self.lsize), dtype=bool)
        self.odd[::2, ::2] = False
        self.odd[1::2, 1::2]  = False
        self.odd = self.odd.flatten()

    def single_spin_flip(self):
        """ flipping a single spin using the Metropolis rule """
        # select a site at random <= random sequential update
        site = np.random.randint(self.n_total)
        local_field = self.bfield + self.spin[self.up[site]] + \
            self.spin[self.dw[site]] + self.spin[self.lft[site]] + \
            self.spin[self.rgt[site]]
        delE = 2.*self.spin[site]*local_field
        # Metropolis rule
        if np.random.random() < np.exp(-self.beta*delE):
            self.spin[site] *= -1

    def MC_random_seq_update(self, steps):
        """ random sequential spin flips for 'steps' MC steps """
        for n in range(self.n_total*steps):
            self.single_spin_flip()

    def MC_sublattice_update(self, steps):
        """ random sublattice updates """
        for _ in range(2*steps):
            # energy change
            d_e = 2.*self.spin * (self.bfield + \
                            self.spin[self.up] + self.spin[self.dw] + \
                            self.spin[self.lft] + self.spin[self.rgt] )
            # flipping probility
            b_w = np.exp(-self.beta*d_e)

            # list of flipping sites
            r_n = np.random.random(self.n_total)
            flipping_site = r_n <= b_w

            if np.random.random()<0.5 :
                """ even sublattice update """ 
                self.spin[ flipping_site * self.even ] *= -1
            else :
                """ odd sublattice update """
                self.spin[ flipping_site * self.odd ] *= -1

    def get_energy_density(self):
        energy = -self.bfield*np.sum(self.spin) - \
            np.sum(self.spin*(self.spin[self.up] + self.spin[self.rgt]))
        return energy/self.n_total

    def get_mag_density(self):
        return np.sum(self.spin)/self.n_total


def mc_tseries(model, t_temp, t_meas, dt, n_repeat):
    en = []
    mag = []
    for n in range(n_repeat): # independent runs
        # reset to ferromagnetic initial state
        model.spin[:] = np.ones(model.n_total, dtype=int)
        # MC simulation for transient period
        model.MC_sublattice_update(t_temp)
        # MC simulation for measurement period
        for t in range(t_meas//dt):
            model.MC_sublattice_update(dt)
            en.append(model.get_energy_density())
            mag.append(model.get_mag_density())
    en = np.asarray(en)
    mag = np.asarray(mag)
    return en, mag 
 
if __name__ == "__main__":
    
    lsize = 128      # N spin = lsize*lsize
    bfield = 0.0
    t_temp = 1000   # t_temp > t0
    t_meas = 1000   
    n_repeat = 5
    dt = 1
 
    np.random.seed()  # random number seed initialization
    model = IsingModel2D(lsize = lsize, beta = 1.0, bfield = bfield)
    
    if not os.path.exists('DATA'):
        os.makedirs('DATA')        
    fname = f'DATA/L{lsize}_ens_av.dat'

    with open(fname, 'w') as fn:
        fn.write('# MC Simulations of 2D Ising Model from Random and Ferromagnetic Initial Condition \n')
        fn.write(f'# L = {lsize}\n')
        fn.write(f'# t_0 = {t_temp} t_meas = {t_meas} dt = {dt} \n')
        fn.write(f'# beta energy_density abs_magnetization_density specific_heat susceptibility binder_parameter\n')
       
        # beta from 0.2 to 5.2; T from 0.19 to 5
        for n in range(400):
            beta0 = 0.2+0.0125*n
            model.beta = beta0
            en, mag = mc_tseries(model, t_temp, t_meas, dt, n_repeat)
            energy_density = np.average(en)
            abs_magnetization_density = np.average(np.abs(mag))
            specific_heat = (np.average(en**2)-(np.average(en)**2))*(lsize*lsize)*(beta0**2)
            susceptibility = (np.average(mag**2)-(np.average(np.abs(mag))**2))*(lsize*lsize)*(beta0)
            binder_parameter = 1-(np.average(mag**4)/(3.*(np.average(mag**2)**2)))
            fn.write(f'{beta0} {energy_density:+e} {abs_magnetization_density:+e} {specific_heat:+e} {susceptibility:+e} {binder_parameter:+e}\n')

