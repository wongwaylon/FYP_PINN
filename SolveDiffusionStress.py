"""
Solve solid spherical diffusion with stress impact
based on [Zhang et al., JES, 2007]
author: HY
"""

import numpy as np
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
import pandas as pd

# Parameters from [Zhang et al., JES, 2007]
E = 10e9 # Young's modulus [Pa]
nu = 0.3 # Possion's ratio
D = 7.08e-15 # Diffusion coefficient [m2/s]
omega = 3.497e-6 # Partial molar volume [m3/mol]
c_max = 2.29e4 # Max molar concentration [mol/m^3]
c_0 = 0e4 # Initial molar concentration [mol/m3]
T = 273+30 # Temperature [K]
R = 8.314 # Gas constant [J/mol/K]
F = 96485 # Faraday constant [As/mol]
Rs = 5e-6 # Particel radius [m]
In = -2 # Current density [A/m2] negative for charging
theta = (omega/R/T)*(2*omega*E/9/(1-nu)) # [m3/mol]
# theta = 0

# Discretisation
N = 100 # Number of grid for CVM
dt = 1 # time step 1s
c_prev = np.ones((N))*c_0 # Previous molar concentration
c_prev = c_prev/c_max # Normalisation
Y = 10
r = np.zeros((N,))
for i in range(1,N+1):
    r[i-1] = 1*(1-(Y**((N-i)/(N-1))-1)/(Y-1))
dr = r[1:]-r[0:-1]
r_avg = (r[0:-1]+r[1:])/2

# Discretised governing equation based on CVM
def diffusionStressEqns(c, *c_prev):
    # construct discretised governing equations
    Func = np.zeros(N)
    for i in range(N):
        if i == 0: # at centre
            Func[i] = 1/24*dr[0]**3*Rs**2*(c[0]-c_prev[0])/dt - \
                1/2*D*(1+theta*c_max*(c_prev[0]+(c_prev[1]-c_prev[0])/4))*r_avg[0]**2*(c_prev[1]-c_prev[0])/dr[0] - \
                D/24*theta*c_max*((c_prev[1]-c_prev[0])/(r_avg[0]-r[0]))**2*(r_avg[0]**3-r[0]**3) - \
                1/2*D*(1+theta*c_max*(c[0]+(c[1]-c[0])/4))*r_avg[0]**2*(c[1]-c[0])/dr[0] - \
                D/24*theta*c_max*((c[1]-c[0])/(r_avg[0]-r[0]))**2*(r_avg[0]**3-r[0]**3)
        elif i == N-1: # at surface
            Func[i] = 1/3*(r[-1]**3-r_avg[-1]**3)*Rs**2*(c[-1]-c_prev[-1])/dt - \
                1/2*D*(1+theta*c_max*(c_prev[-1]+(c_prev[-1]-c_prev[-2])/4)) * \
                (r[-1]**2*(-In*Rs/D/c_max/F/(1+theta*c_max*c_prev[-1]))-r_avg[-1]**2*(c_prev[-1]-c_prev[-2])/dr[-1]) - \
                D/24*theta*c_max*((c_prev[-1]-c_prev[-2])/(r[-1]-r_avg[-1]))**2*(r[-1]**3-r_avg[-1]**3) - \
                1/2*D*(1+theta*c_max*(c[-1]+(c[-1]-c[-2])/4)) * \
                (r[-1]**2*(-In*Rs/D/c_max/F/(1+theta*c_max*c[-1]))-r_avg[-1]**2*(c[-1]-c[-2])/dr[-1]) - \
                D/24*theta*c_max*((c[-1]-c[-2])/(r[-1]-r_avg[-1]))**2*(r[-1]**3-r_avg[-1]**3)
        else: # in the middle
            Func[i] = 1/3*(r_avg[i]**3-r_avg[i-1]**3)*Rs**2*(c[i]-c_prev[i])/dt - \
                1/2*D*(1+theta*c_max*c_prev[i])*(r_avg[i]**2*((c_prev[i+1]-c_prev[i])/(r[i+1]-r[i]))-r_avg[i-1]**2*((c_prev[i]-c_prev[i-1])/(r[i]-r[i-1]))) - \
                D/24*theta*c_max*((c_prev[i+1]-c_prev[i-1])/(r_avg[i]-r_avg[i-1]))**2*(r_avg[i]**3-r_avg[i-1]**3) - \
                1/2*D*(1+theta*c_max*c[i])*(r_avg[i]**2*((c[i+1]-c[i])/(r[i+1]-r[i]))-r_avg[i-1]**2*((c[i]-c[i-1])/(r[i]-r[i-1]))) - \
                D/24*theta*c_max*((c[i+1]-c[i-1])/(r_avg[i]-r_avg[i-1]))**2*(r_avg[i]**3-r_avg[i-1]**3)
    return Func

# solve discretised equations
c_guess = c_prev
c_all = np.zeros((N,1000))
for hh in range(1000):
    solutionTuple = fsolve(diffusionStressEqns,c_guess,args=tuple(c_prev),full_output=1)        
    c_prev = solutionTuple[0]
    c_all[...,hh] = c_prev
    if hh % 250 == 0:
        plt.plot(r*Rs, c_prev*c_max)
        plt.xlabel('r (m)')
        plt.xlim(0, 6e-6)
        plt.ylabel('Concentration (mol/m3)')
        # plt.ylim(0.8e4, 1.6e4)
        plt.title(f"Time Step {hh}")
        plt.show()

print(len(c_prev))         
# plot results
plt.plot(r*Rs, c_prev*c_max)
plt.xlabel('r (m)')
plt.xlim(0, 6e-6)
plt.ylabel('Concentration (mol/m3)')
plt.ylim(0.8e4, 1.6e4)
plt.title(f"Time Step {hh}")
plt.show()

# save data
save_r = pd.DataFrame(r*Rs)
save_t = pd.DataFrame(np.arange(0,1000,1))
save_c = pd.DataFrame(c_all*c_max) # r=rows, t=cols
save_r.to_csv("CVM_r.csv", header=False, index=False)
save_t.to_csv("CVM_t.csv", header=False, index=False)
save_c.to_csv("CVM_c.csv", header=False, index=False)