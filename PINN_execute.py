import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from timeit import default_timer as timer

import PINN_code as PINN

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

start = timer() # start time

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
Rs = 5e-6 # Particle radius [m]
In = -2 # Current density [A/m2] negative for charging
theta = (omega/R/T)*(2*omega*E/9/(1-nu)) # [m3/mol]

# Normalisation
tmax = 1000 # time domain max
I = In*Rs/(D*c_max*F)
theta_norm = theta*c_max
c_ini_norm = c_0/c_max

# Model and training parameters
mse_loss_func = nn.MSELoss() # mean squared error loss function
randomseed = 12121 # random seed
layers = 4 # network layers
nodes = 256 # neurons per layer
input_dim = 2 # input dimensions
output_dim = 1 # output dimensiosn
act_func = nn.Tanh # activation function
bf = 't0' # boundary forcing function

# load validation files
CVM_r_data = np.loadtxt('CVM_r.csv', delimiter=',')
CVM_t_data = np.loadtxt('CVM_t.csv', delimiter=',')
CVM_c_data = np.loadtxt('CVM_c.csv', delimiter=',')
CVM_r = CVM_r_data/Rs
CVM_t = CVM_t_data[:tmax+1]/tmax
CVM_c = CVM_c_data[:,:tmax+1]/c_max

# model file
model = '4x256tanh_b128N2k_a5k_l100_tmax1000_t0'
filename = f'PINN_state_{model}.pt'

# load PINN and import variables
net = PINN.PINN(input_dim, output_dim, lays=layers, n_units=nodes, 
                act_func=act_func, loss=mse_loss_func, device=device, 
                seed=randomseed, boundary_force=bf).to(device)
net.vals(tmax=tmax,rmax=Rs,cmax=c_max,D=D,theta_norm=theta_norm)
net.load_state_dict(torch.load(filename))
net.eval()
net.eval()

# visualise PDE + calculate PDE and validation losses
pde_val = net.pde_visual()

# compute PINN prediction
net_r, net_t, ms_r, ms_t = PINN.meshgrid_plot(CVM_r_data/Rs, CVM_t_data/tmax)
c = net.predict(net_r, net_t)
c = c.detach().cpu().numpy()
ms_c = c.reshape(ms_r.shape)

end = timer() # end time
train_time = end-start # execution time
print(train_time,'s')

# plot prediction surface plot
fig, ax = plt.subplots(figsize=(6, 4), subplot_kw={"projection": "3d"})
ax.plot_surface(Rs*ms_r[:,:tmax+1],tmax*ms_t[:,:tmax+1],c_max*ms_c[:,:tmax+1], cmap='viridis', antialiased=False)
plt.title("Neural Network Prediction")
plt.xlabel("r (m)")
plt.ylabel("t (s)")
plt.show()

# compute and plot error
val_err = np.absolute(ms_c-CVM_c_data/c_max)

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
ax.plot_surface(Rs*ms_r[:,:tmax+1],tmax*ms_t[:,:tmax+1],val_err[:,:tmax+1], cmap='viridis', antialiased=False)
plt.title("Validation Loss")
plt.xlabel("r (m)")
plt.ylabel("t (s)")
plt.show()

fig, ax = plt.subplots(figsize=(6, 4))
val_img = ax.pcolormesh(Rs*ms_r[:,:tmax+1], tmax*ms_t[:,:tmax+1], val_err[:,:tmax+1], cmap='jet')
ax.set_title('Validation Loss Magnitude')
ax.axis([Rs*CVM_r[0], Rs*CVM_r[-1], tmax*CVM_t[0], tmax*CVM_t[-1]])
fig.colorbar(val_img, ax=ax, label='Loss Magnitude')
ax.set_xlabel("r (m)")
ax.set_ylabel("t (s)")
plt.show()

# plot concentration predictions
legends = ['Neural Network', 'CVM Solution']
font = 10
fix, ax = plt.subplots(2,2,figsize=(12, 8))
ax[0,0].plot(CVM_r_data,c_max*ms_c[...,1])
ax[0,0].plot(CVM_r_data,CVM_c_data[...,1])
ax[0,0].legend(legends,fontsize=font)
ax[0,0].set_xlabel("r ($m$)")
ax[0,0].set_ylabel("c ($mol$ $m^{-3}$)")
ax[0,0].set_title('Concentration at t=1s')
ax[0,1].plot(CVM_r_data,c_max*ms_c[...,tmax+1])
ax[0,1].plot(CVM_r_data,CVM_c_data[...,tmax+1])
ax[0,1].legend(legends,fontsize=font)
ax[0,1].set_xlabel("r ($m$)")
ax[0,1].set_ylabel("c ($mol$ $m^{-3}$)")
ax[0,1].set_title(f'Concentration at t={tmax}s')
ax[1,0].plot(CVM_t_data[:tmax+1],c_max*ms_c[0,:tmax+1])
ax[1,0].plot(CVM_t_data[:tmax+1],CVM_c_data[0,:tmax+1])
ax[1,0].legend(legends,fontsize=font)
ax[1,0].set_xlabel("t ($s$)")
ax[1,0].set_ylabel("c ($mol$ $m^{-3}$)")
ax[1,0].set_title('Concentration at r=0m')
ax[1,1].plot(CVM_t_data[:tmax+1],c_max*ms_c[-1,:tmax+1])
ax[1,1].plot(CVM_t_data[:tmax+1],CVM_c_data[-1,:tmax+1])
ax[1,1].legend(legends,fontsize=font)
ax[1,1].set_xlabel("t ($s$)")
ax[1,1].set_ylabel("c ($mol$ $m^{-3}$)")
ax[1,1].set_title('Concentration at r=5e-6m')
plt.tight_layout()
plt.show()

# plot predictions outside of the training domain
fix, ax = plt.subplots(1,2,figsize=(12, 4))
ax[0].plot(CVM_r_data,c_max*ms_c[...,1661])
ax[0].plot(CVM_r_data,CVM_c_data[...,1661])
ax[0].legend(legends,fontsize=font)
ax[0].set_xlabel("r ($m$)")
ax[0].set_ylabel("c ($mol$ $m^{-3}$)")
ax[0].set_title('Concentration at t=1660s')
ax[1].plot(CVM_r_data,c_max*ms_c[...,-1])
ax[1].plot(CVM_r_data,CVM_c_data[...,-1])
ax[1].legend(legends,fontsize=font)
ax[1].set_xlabel("r ($m$)")
ax[1].set_ylabel("c ($mol$ $m^{-3}$)")
ax[1].set_title('Concentration at t=2000s')
plt.tight_layout()
plt.show()