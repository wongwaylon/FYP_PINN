import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset
import matplotlib.pyplot as plt
from timeit import default_timer as timer

import PINN_code as PINN

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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
lr = 1e-3 # adam learning rate
mse_loss_func = nn.MSELoss() # mean squared error loss function
randomseed = 12121 # random seed
layers = 4 # network layers
nodes = 256 # neurons per layer
input_dim = 2 # input dimensions
output_dim = 1 # output dimensiosn
act_func = nn.Tanh # activation function
N = 2000 # training dataset size
batch = 128 # batch size
adam_epochs = 5000 # adam optimiser epochs
lbfgs_epochs = 100 # lbfgs optimiser epochs
data_change = False # adaptive sampling
bf = 't0' # boundary forcing function

# load validation files and create validation dataset
CVM_r_data = np.loadtxt('CVM_r.csv', delimiter=',')
CVM_t_data = np.loadtxt('CVM_t.csv', delimiter=',')
CVM_c_data = np.loadtxt('CVM_c.csv', delimiter=',')
CVM_r = CVM_r_data/Rs
CVM_t = CVM_t_data[:tmax+1]/tmax
CVM_c = CVM_c_data[:,:tmax+1]/c_max
val_r, val_t, val_c = PINN.CVM_data(CVM_r,CVM_t,CVM_c)
val_data = TensorDataset(val_t, val_r, val_c)

# create training data
t0_data,r0_data,r1_data,phy_data = PINN.create_data(N,0,0,I,seed=randomseed)

# load PINN and import variables and data
net = PINN.PINN(input_dim, output_dim, lays=layers, n_units=nodes, 
                act_func=act_func, loss=mse_loss_func, device=device, 
                seed=randomseed, boundary_force=bf).to(device)
net.vals(tmax=tmax,rmax=Rs,cmax=c_max,D=D,theta_norm=theta_norm)
net.load_data(batch,t0_data,r0_data,r1_data,phy_data,val_data)

start = timer() # start time

# train model
adam_losses, lbfgs_losses = net.train_loop(adam_epochs=adam_epochs, adam_lr=lr, 
                                           lbfgs_epochs=lbfgs_epochs, plots=2,
                                           data_change=data_change)

end = timer() # end time
train_time = end-start # training time
print(train_time,'s')

# save model
model = '4x256tanh_b128N2k_a5k_l100_tmax1000_t0'
torch.save(net.state_dict(),f'PINN_state_{model}.pt')

# visualise PDE + calculate PDE and validation losses
pde_val = net.pde_visual()
val_val = net.val_err()

# save losses to excel
col1 = ['Adam Loss', 'Adam t0', 'Adam r0', 'Adam r1', 'Adam PDE', 'Adam Val']
col2 = ['LBFGS Loss', 'LBFGS t0', 'LBFGS r0', 'LBFGS r1', 'LBFGS PDE', 'LBFGS Val']
adam = np.transpose(adam_losses)
lbfgs = np.transpose(lbfgs_losses)
adam = pd.DataFrame(data=adam, columns=col1)
lbfgs = pd.DataFrame(data=lbfgs, columns=col2)
combine = pd.concat([adam, lbfgs], axis=1)
with pd.ExcelWriter('PINN_losses.xlsx', engine='xlsxwriter') as writer:
        combine.to_excel(writer, sheet_name=model)

# compute PINN prediction
net_r, net_t, ms_r, ms_t = PINN.meshgrid_plot(CVM_r_data/Rs, CVM_t_data/tmax)
c = net.predict(net_r, net_t)
c = c.detach().cpu().numpy()
ms_c = c.reshape(ms_r.shape)

# plot prediction surface plot
fig, ax = plt.subplots(figsize=(6, 4), subplot_kw={"projection": "3d"})
ax.plot_surface(Rs*ms_r[:,:1001],tmax*ms_t[:,:1001],c_max*ms_c[:,:1001], cmap='viridis', antialiased=False)
plt.title("Neural Network Prediction")
plt.xlabel("r (m)")
plt.ylabel("t (s)")
plt.show()

# compute and plot error
val_err = np.absolute(ms_c-CVM_c_data/c_max)

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
ax.plot_surface(Rs*ms_r[:,:1001],tmax*ms_t[:,:1001],val_err[:,:1001], cmap='viridis', antialiased=False)
plt.title("Validation Loss")
plt.xlabel("r (m)")
plt.ylabel("t (s)")
plt.show()

fig, ax = plt.subplots(figsize=(6, 4))
val_img = ax.pcolormesh(Rs*ms_r[:,:1001], tmax*ms_t[:,:1001], val_err[:,:1001], cmap='jet')
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
ax[0,1].plot(CVM_r_data,c_max*ms_c[...,1001])
ax[0,1].plot(CVM_r_data,CVM_c_data[...,1001])
ax[0,1].legend(legends,fontsize=font)
ax[0,1].set_xlabel("r ($m$)")
ax[0,1].set_ylabel("c ($mol$ $m^{-3}$)")
ax[0,1].set_title('Concentration at t=1000s')
ax[1,0].plot(CVM_t_data[:1001],c_max*ms_c[0,:1001])
ax[1,0].plot(CVM_t_data[:1001],CVM_c_data[0,:1001])
ax[1,0].legend(legends,fontsize=font)
ax[1,0].set_xlabel("t ($s$)")
ax[1,0].set_ylabel("c ($mol$ $m^{-3}$)")
ax[1,0].set_title('Concentration at r=0m')
ax[1,1].plot(CVM_t_data[:1001],c_max*ms_c[-1,:1001])
ax[1,1].plot(CVM_t_data[:1001],CVM_c_data[-1,:1001])
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

# plot training and validation losses
errs = np.concatenate((adam_losses,lbfgs_losses),axis=1)
fix, ax = plt.subplots(1,2,figsize=(12, 4))
ax[0].plot(np.arange(0,adam_epochs+lbfgs_epochs,1),errs[0])
ax[0].set_xlabel("Epochs")
ax[0].set_ylabel("Loss")
ax[0].set_yscale("log")
ax[0].set_title('Training Loss')
ax[1].plot(np.arange(0,adam_epochs+lbfgs_epochs,1),errs[5])
ax[1].set_xlabel("Epochs")
ax[1].set_ylabel("Loss")
ax[1].set_yscale("log")
ax[1].set_title('Validation Loss')
plt.tight_layout()
plt.show()