'''
Plot energy and magnetization density, specific heat, susceptibility
Find critical temperature
Find critical exponents

Written by Kyungmin Park
'''
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# import data
data = np.loadtxt('DATA/L128_ens_av.dat')

# find the critical temperature Tc
criticalT_indices = [ np.argmax(data[:,3]), np.argmax(data[:,4]) ]
criticalT = []
for i in range(len(criticalT_indices)):
    criticalT.append(1./data[criticalT_indices[i],0])

criticalT_indices = np.asarray(criticalT_indices)
criticalT = np.average(np.asarray(criticalT))

# fitting function for finding the critical exponents
def powerlaw(x, a, b):
    return a * np.power(x,b)    # x = (T - Tc) or (Tc - T)
def powerlaw2(x, a, b, c):
    return a * np.power(x,b) + c    # x = (T - Tc) or (Tc - T)

# find alpha and gamma ( x = T - Tc )
alpha_gamma_x = 1./data[:np.max(criticalT_indices),0]  # for alpha and gamma
beta_x = 1./data[np.max(criticalT_indices)-5:,0]

alpha_y = data[:np.max(criticalT_indices),3]           # for alpha
gamma_y = data[:np.max(criticalT_indices),4]           # for gamma
beta_y = data[np.max(criticalT_indices)-5:,2]

fit_alpha, cov_alpha = curve_fit(powerlaw, alpha_gamma_x, alpha_y)
fit_beta, cov_beta = curve_fit(powerlaw2, beta_x, beta_y, p0=[-1,5,1])
fit_gamma, cov_gamma = curve_fit(powerlaw, alpha_gamma_x, gamma_y)

print(fit_alpha)
print(fit_beta)
print(fit_gamma)

# set signs for alpha, beta, gamma
alpha  = -fit_alpha[1]
beta = fit_beta[1]
gamma = -fit_gamma[1]

# plot on canvas
fig = plt.figure()

ax1 = fig.add_subplot(2,2,1)
ax1.set(xlabel='Temperature',  ylabel='Energy Density')
ax1.plot(1./data[:,0], data[:,1], label="lsize = 128")
ax1.set_xticks([1,3,5,criticalT])

ax2 = fig.add_subplot(2,2,2)
ax2.set(xlabel='Temperature',  ylabel='|Magnetization| Density')
ax2.plot(1./data[:,0], data[:,2], label="lsize = 128")
ax2.plot(beta_x[4:], powerlaw2(beta_x[4:], fit_beta[0], fit_beta[1], fit_beta[2]))
ax2.set_xticks([1,3,5,criticalT])

ax3 = fig.add_subplot(2,2,3)
ax3.set(xlabel='Temperature',  ylabel='Specific Heat')
ax3.plot(1./data[:,0], data[:,3], label="lsize = 128")
ax3.plot(alpha_gamma_x, powerlaw(alpha_gamma_x, fit_alpha[0], fit_alpha[1]))
ax3.set_xticks([1,3,5,criticalT])

ax4 = fig.add_subplot(2,2,4)
ax4.set(xlabel='Temperature',  ylabel='Susceptibility')
ax4.plot(1./data[:,0], data[:,4], label="lsize = 128")
ax4.plot(alpha_gamma_x, powerlaw(alpha_gamma_x, fit_gamma[0], fit_gamma[1]))
ax4.set_xticks([1,3,5,criticalT])

print(f'Critical temperature Tc = {criticalT:.2f}K')
print(f'Alpha = {alpha:.2f}, Beta = {beta:.2f}, Gamma = {gamma:.2f}')

#plt.legend()
plt.show()
