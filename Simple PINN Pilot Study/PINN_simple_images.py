import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

import PINN_simple as PINN

# model parameters
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
randomseed = 12121
input_dim = 2
output_dim = 1

# load network
name = 'PINN_simple_e20k_N1000.pt'
net = PINN.PINN(input_dim, output_dim, seed=randomseed).to(device)
net.load_state_dict(torch.load(name))
net.eval()

# create a 100x100 grid over the domain
x=np.linspace(0,1,101)
t=np.linspace(0,1,101)
net_x, net_t, ms_x, ms_t = PINN.meshgrid_plot(x,t)

# PINN prediction
c_pred = net.predict(net_x, net_t)

# solve for analytical solution
c_val = np.zeros((len(net_x),1))
for i in range(len(net_x)):
    c_val[i] = PINN.analy_soln(net_x[i].detach().numpy(),net_t[i].detach().numpy())
c_val = torch.from_numpy(c_val).float().to(device)

# turn into 2D gridpoints
c_val = c_val.detach().cpu().numpy()
c_pred = c_pred.detach().cpu().numpy()
ms_c = c_pred.reshape(ms_x.shape)
val_c = c_val.reshape(ms_x.shape)

# load loss data
loss_data = pd.read_excel('PINN_simple.xlsx', sheet_name='losses_e20k_N1000', index_col=0)
adam_loss = loss_data['Final Loss']
adam_val = loss_data['Validation Loss']

# plot the training and validation losses
fix, ax = plt.subplots(1,2,figsize=(12, 4))
ax[0].plot(np.arange(len(adam_loss)),adam_loss)
ax[0].set_xlabel("Epochs")
ax[0].set_ylabel("Loss")
ax[0].set_title('Training Loss')
ax[0].set_yscale("log")
ax[1].plot(np.arange(len(adam_val)),adam_val)
ax[1].set_xlabel("Epochs")
ax[1].set_ylabel("Loss")
ax[1].set_title('Validation Loss')
ax[1].set_yscale("log")
plt.tight_layout()
plt.show()
fix.savefig('simplePINN_losses.png') # save figure

# plot PINN prediction vs analytical solution
legends = ['Neural Network', 'Analytical Solution']
font = 10
fix, ax = plt.subplots(1,2,figsize=(12, 4))
ax[0].plot(x,ms_c[...,1])
ax[0].plot(x,val_c[...,1])
ax[0].legend(legends,fontsize=font)
ax[0].set_xlabel("x")
ax[0].set_ylabel("c)")
ax[0].set_title('Concentration at t=0.01')
ax[1].plot(x,ms_c[...,-1])
ax[1].plot(x,val_c[...,-1])
ax[1].legend(legends,fontsize=font)
ax[1].set_xlabel("x")
ax[1].set_ylabel("c")
ax[1].set_title('Concentration at t=1')
plt.tight_layout()
plt.show()
fix.savefig('simplePINN_comparision.png') # save figure

# plot 3D surface plot of the PINN prediction
fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, figsize=(6,4))
ax.plot_surface(ms_t,ms_x,ms_c, cmap='viridis', antialiased=False)
plt.title("PINN Concentration Prediction")
plt.xlabel("t")
plt.ylabel("x")
plt.tight_layout()
plt.show()
fig.savefig('simplePINN_surfplot.png')