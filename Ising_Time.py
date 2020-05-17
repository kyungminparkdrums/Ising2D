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


if __name__ == "__main__":
    # boltzamann constant = 1
    beta = 1
    lsize = 128      # N spin = lsize*lsize
    bfield = 0.0    
    t_meas = 1000   
    dt = 1         
    
    np.random.seed(0)  # random number seed initialization

    model_rand = IsingModel2D(lsize = lsize, beta = beta, bfield = bfield)
    model_ferr = IsingModel2D(lsize = lsize, beta = beta, bfield = bfield) 
    model_ferr.spin[:] = np.ones(model_ferr.n_total, dtype=int)
       
    if not os.path.exists('DATA'):
        os.makedirs('DATA')        
    fname = f'DATA/Kyungmin.dat'
    with open(fname, "w") as fn:
        fn.write('# MC Simulations of 2D Ising Model from Random and Ferromagnetic Initial Condition \n')
        fn.write(f'# L = {lsize}, Beta = {beta:.2f}\n')
        fn.write(f'# t ener_rand mag_rand ener_ferr mag_ferr\n')
        for t in range((t_meas//dt)+1):
            model_rand.MC_sublattice_update(dt)
            model_ferr.MC_sublattice_update(dt)
            fn.write(f'{t*dt} {model_rand.get_energy_density():+.12e} {model_rand.get_mag_density():+.12e} {model_ferr.get_energy_density():+.12e} {model_ferr.get_mag_density():+.12e}\n')

    DATA = np.loadtxt('DATA/Kyungmin.dat')
   
    plt.xlabel('Time')
    plt.ylabel('Energy Density')
    plt.plot(DATA[:,0], DATA[:,1], label='Random Initial Condition')
    plt.plot(DATA[:,0], DATA[:,3], label='Ferromagnetic Initial Condition')
    plt.title('Energy Density (T = 1K, kB = 1J/K)')
    plt.legend()
    plt.savefig('1_Energy_density_Time.png') 
    plt.show() 
    
    plt.xlabel('Time')
    plt.ylabel('|Magnetization| Density')
    plt.plot(DATA[:,0], np.abs(DATA[:,2]), label='Random Initial Condition')
    plt.plot(DATA[:,0], np.abs(DATA[:,4]), label='Ferromagnetic Initial Condition')
    plt.title('|Magnetization| Density  (T = 1K, kB = 1J/K)')
    plt.legend()
    plt.savefig('1_Abs_magnetization_density_Time.png') 
    plt.show() 
    
