# Ising2D
## Monte Carlo simulation for the Ising model in 2D with the Metropolis algorithm

Statistical Physics II / 2020 Spring / Midterm Project

Kyungmin Park / Department of Physics / University of Seoul

**1. 'Ising_Time.py'**
- Save as a .dat file the energy density and |magnetization| density from random spin initial condition(T=inf) and ferromagnetic spin initial condition(T=0) for MC time [0,1000] with dt = 1 (lsize = 128, Nspin = lsize*lsize)
- From the .dat file, plot Energy density - Time, |Magnetizaton| density - Time

**2. 'Ising_Temperature.py'**
- Save as a .dat file the [ energy density, |magnetization| density, specific heat, susceptibility, binder parameter ] from ferromagnetic spin initial condition(T=0) for MC time [0,1000] with five independent runs for each beta, after thermal equilibrium. (lsize = 128, beta from 0.2 to 5.2 with d(beta) = 0.0125)

**3. 'Critical_Temperature.py'**
- Find the critical temperature where the phase transition takes place: find where specfic heat and susceptibility "diverge".
- From the .dat file, plot Energy density - Time, |Magnetization| density - Time, Specific Heat - Time, Susceptibility - Time
- Find the critical exponent alpha, beta, and gamma.

## Sidenote
* If you are running these scripts on terminal, './DATA' directory will be created if it doesn't exist.

* Expected running time for the scripts: 'Ising_Time.py' - 30s, 'Ising_Temperature.py' - 1h 30m, 'Critical_Temperature.py' - 30s
