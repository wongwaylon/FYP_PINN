import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

import PINN_code as PINN

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# load validation files
CVM_r_data = np.loadtxt('CVM_r.csv', delimiter=',')
CVM_t_data = np.loadtxt('CVM_t.csv', delimiter=',')
CVM_c_data = np.loadtxt('CVM_c.csv', delimiter=',')

# load model predictions
# network parameters
c_t1_p1 = pd.read_excel('PINN Models/PINN_net_params.xlsx', sheet_name='t1', index_col=0)
c_t1000_p1 = pd.read_excel('PINN Models/PINN_net_params.xlsx', sheet_name='t1000', index_col=0)
c_t1660_p1 = pd.read_excel('PINN Models/PINN_net_params.xlsx', sheet_name='t1660', index_col=0)
c_t2000_p1 = pd.read_excel('PINN Models/PINN_net_params.xlsx', sheet_name='t2000', index_col=0)
c_r0_p1 = pd.read_excel('PINN Models/PINN_net_params.xlsx', sheet_name='r0', index_col=0)
c_r1_p1 = pd.read_excel('PINN Models/PINN_net_params.xlsx', sheet_name='r1', index_col=0)

# dataset sizes and training epochs
c_t1_p2 = pd.read_excel('PINN Models/PINN_dataset_epochs.xlsx', sheet_name='t1', index_col=0)
c_t1000_p2 = pd.read_excel('PINN Models/PINN_dataset_epochs.xlsx', sheet_name='t1000', index_col=0)
c_t1660_p2 = pd.read_excel('PINN Models/PINN_dataset_epochs.xlsx', sheet_name='t1660', index_col=0)
c_t2000_p2 = pd.read_excel('PINN Models/PINN_dataset_epochs.xlsx', sheet_name='t2000', index_col=0)
c_r0_p2 = pd.read_excel('PINN Models/PINN_dataset_epochs.xlsx', sheet_name='r0', index_col=0)
c_r1_p2 = pd.read_excel('PINN Models/PINN_dataset_epochs.xlsx', sheet_name='r1', index_col=0)

# adaptive sampling
c_t1_p3 = pd.read_excel('PINN Models/PINN_adaptive_sampling.xlsx', sheet_name='t1', index_col=0)
c_t1000_p3 = pd.read_excel('PINN Models/PINN_adaptive_sampling.xlsx', sheet_name='t1000', index_col=0)
c_r0_p3 = pd.read_excel('PINN Models/PINN_adaptive_sampling.xlsx', sheet_name='r0', index_col=0)
c_r1_p3 = pd.read_excel('PINN Models/PINN_adaptive_sampling.xlsx', sheet_name='r1', index_col=0)

# boundary forcing
c_t1_p4 = pd.read_excel('PINN Models/PINN_boundary_forcing.xlsx', sheet_name='t1', index_col=0)
c_t1000_p4 = pd.read_excel('PINN Models/PINN_boundary_forcing.xlsx', sheet_name='t1000', index_col=0)
c_t1660_p4 = pd.read_excel('PINN Models/PINN_boundary_forcing.xlsx', sheet_name='t1660', index_col=0)
c_t2000_p4 = pd.read_excel('PINN Models/PINN_boundary_forcing.xlsx', sheet_name='t2000', index_col=0)
c_r0_p4 = pd.read_excel('PINN Models/PINN_boundary_forcing.xlsx', sheet_name='r0', index_col=0)
c_r1_p4 = pd.read_excel('PINN Models/PINN_boundary_forcing.xlsx', sheet_name='r1', index_col=0)

# legend fontsize
font=10

#%% network sizes
legends = ['3x100 NN', '3x256 NN', '4x100 NN', '4x256 NN', 'CVM Solution']
fig, ax = plt.subplots(2,2,figsize=(12, 8))
ax[0,0].plot(CVM_r_data,c_t1_p1['3x100tanh'])
ax[0,0].plot(CVM_r_data,c_t1_p1['3x256tanh'])
ax[0,0].plot(CVM_r_data,c_t1_p1['4x100tanh'])
ax[0,0].plot(CVM_r_data,c_t1_p1['4x256tanh'])
ax[0,0].plot(CVM_r_data,CVM_c_data[...,1])
ax[0,0].legend(legends,fontsize=font)
ax[0,0].set_xlabel("r ($m$)")
ax[0,0].set_ylabel("c ($mol$ $m^{-3}$)")
ax[0,0].set_title('Concentration at t=1s')
ax[0,1].plot(CVM_r_data,c_t1000_p1['3x100tanh'])
ax[0,1].plot(CVM_r_data,c_t1000_p1['3x256tanh'])
ax[0,1].plot(CVM_r_data,c_t1000_p1['4x100tanh'])
ax[0,1].plot(CVM_r_data,c_t1000_p1['4x256tanh'])
ax[0,1].plot(CVM_r_data,CVM_c_data[...,1001])
ax[0,1].legend(legends,fontsize=font)
ax[0,1].set_xlabel("r ($m$)")
ax[0,1].set_ylabel("c ($mol$ $m^{-3}$)")
ax[0,1].set_title('Concentration at t=1000s')
ax[1,0].plot(CVM_t_data[:1001],c_r0_p1['3x100tanh'])
ax[1,0].plot(CVM_t_data[:1001],c_r0_p1['3x256tanh'])
ax[1,0].plot(CVM_t_data[:1001],c_r0_p1['4x100tanh'])
ax[1,0].plot(CVM_t_data[:1001],c_r0_p1['4x256tanh'])
ax[1,0].plot(CVM_t_data[:1001],CVM_c_data[0,:1001])
ax[1,0].legend(legends,fontsize=font)
ax[1,0].set_xlabel("t ($s$)")
ax[1,0].set_ylabel("c ($mol$ $m^{-3}$)")
ax[1,0].set_title('Concentration at r=0m')
ax[1,1].plot(CVM_t_data[:1001],c_r1_p1['3x100tanh'])
ax[1,1].plot(CVM_t_data[:1001],c_r1_p1['3x256tanh'])
ax[1,1].plot(CVM_t_data[:1001],c_r1_p1['4x100tanh'])
ax[1,1].plot(CVM_t_data[:1001],c_r1_p1['4x256tanh'])
ax[1,1].plot(CVM_t_data[:1001],CVM_c_data[-1,:1001])
ax[1,1].legend(legends,fontsize=font)
ax[1,1].set_xlabel("t ($s$)")
ax[1,1].set_ylabel("c ($mol$ $m^{-3}$)")
ax[1,1].set_title('Concentration at r=5e-6m')
plt.tight_layout()
plt.show()
fig.savefig('Model Plots/networks_predictions.png')

#%% activation function
legends = ['Tanh', 'Sin', 'CVM Solution']
fig, ax = plt.subplots(2,2,figsize=(12, 8))
ax[0,0].plot(CVM_r_data,c_t1_p1['4x256tanh'])
ax[0,0].plot(CVM_r_data,c_t1_p1['4x256sin'])
ax[0,0].plot(CVM_r_data,CVM_c_data[...,1])
ax[0,0].legend(legends,fontsize=font)
ax[0,0].set_xlabel("r ($m$)")
ax[0,0].set_ylabel("c ($mol$ $m^{-3}$)")
ax[0,0].set_title('Concentration at t=1s')
ax[0,1].plot(CVM_r_data,c_t1000_p1['4x256tanh'])
ax[0,1].plot(CVM_r_data,c_t1000_p1['4x256sin'])
ax[0,1].plot(CVM_r_data,CVM_c_data[...,1001])
ax[0,1].legend(legends,fontsize=font)
ax[0,1].set_xlabel("r ($m$)")
ax[0,1].set_ylabel("c ($mol$ $m^{-3}$)")
ax[0,1].set_title('Concentration at t=1000s')
ax[1,0].plot(CVM_t_data[:1001],c_r0_p1['4x256tanh'])
ax[1,0].plot(CVM_t_data[:1001],c_r0_p1['4x256sin'])
ax[1,0].plot(CVM_t_data[:1001],CVM_c_data[0,:1001])
ax[1,0].legend(legends,fontsize=font)
ax[1,0].set_xlabel("t ($s$)")
ax[1,0].set_ylabel("c ($mol$ $m^{-3}$)")
ax[1,0].set_title('Concentration at r=0m')
ax[1,1].plot(CVM_t_data[:1001],c_r1_p1['4x256tanh'])
ax[1,1].plot(CVM_t_data[:1001],c_r1_p1['4x256sin'])
ax[1,1].plot(CVM_t_data[:1001],CVM_c_data[-1,:1001])
ax[1,1].legend(legends,fontsize=font)
ax[1,1].set_xlabel("t ($s$)")
ax[1,1].set_ylabel("c ($mol$ $m^{-3}$)")
ax[1,1].set_title('Concentration at r=5e-6m')
plt.tight_layout()
plt.show()
fig.savefig('Model Plots/activations_predictions.png')

fig, ax = plt.subplots(1,2,figsize=(12, 4))
ax[0].plot(CVM_r_data,c_t1660_p1['4x256tanh']) # 4x256 tanh
ax[0].plot(CVM_r_data,c_t1660_p1['4x256sin']) # 4x256 sin
ax[0].plot(CVM_r_data,CVM_c_data[...,1661])
ax[0].legend(legends,fontsize=font)
ax[0].set_xlabel("r ($m$)")
ax[0].set_ylabel("c ($mol$ $m^{-3}$)")
ax[0].set_title('Concentration at t=1660s')
ax[1].plot(CVM_r_data,c_t2000_p1['4x256tanh']) # 4x256 tanh
ax[1].plot(CVM_r_data,c_t2000_p1['4x256sin']) # 4x256 sin
ax[1].plot(CVM_r_data,CVM_c_data[...,-1])
ax[1].legend(legends,fontsize=font)
ax[1].set_xlabel("r ($m$)")
ax[1].set_ylabel("c ($mol$ $m^{-3}$)")
ax[1].set_title('Concentration at t=2000s')
plt.tight_layout()
plt.show()
fig.savefig('Model Plots/activations_untrained.png')

#%% batch sizes
legends = ['batch=128', 'batch=256', 'batch=512', 'CVM Solution']
fig, ax = plt.subplots(2,2,figsize=(12, 8))
ax[0,0].plot(CVM_r_data,c_t1_p2['b128N1k_a5k_l100'])
ax[0,0].plot(CVM_r_data,c_t1_p2['b256N1k_a5k_l100'])
ax[0,0].plot(CVM_r_data,c_t1_p2['b512N1k_a5k_l100'])
ax[0,0].plot(CVM_r_data,CVM_c_data[...,1])
ax[0,0].legend(legends,fontsize=font)
ax[0,0].set_xlabel("r ($m$)")
ax[0,0].set_ylabel("c ($mol$ $m^{-3}$)")
ax[0,0].set_title('Concentration at t=1s')
ax[0,1].plot(CVM_r_data,c_t1000_p2['b128N1k_a5k_l100'])
ax[0,1].plot(CVM_r_data,c_t1000_p2['b256N1k_a5k_l100'])
ax[0,1].plot(CVM_r_data,c_t1000_p2['b512N1k_a5k_l100'])
ax[0,1].plot(CVM_r_data,CVM_c_data[...,1001])
ax[0,1].legend(legends,fontsize=font)
ax[0,1].set_xlabel("r ($m$)")
ax[0,1].set_ylabel("c ($mol$ $m^{-3}$)")
ax[0,1].set_title('Concentration at t=1000s')
ax[1,0].plot(CVM_t_data[:1001],c_r0_p2['b128N1k_a5k_l100'])
ax[1,0].plot(CVM_t_data[:1001],c_r0_p2['b256N1k_a5k_l100'])
ax[1,0].plot(CVM_t_data[:1001],c_r0_p2['b512N1k_a5k_l100'])
ax[1,0].plot(CVM_t_data[:1001],CVM_c_data[0,:1001])
ax[1,0].legend(legends,fontsize=font)
ax[1,0].set_xlabel("t ($s$)")
ax[1,0].set_ylabel("c ($mol$ $m^{-3}$)")
ax[1,0].set_title('Concentration at r=0m')
ax[1,1].plot(CVM_t_data[:1001],c_r1_p2['b128N1k_a5k_l100'])
ax[1,1].plot(CVM_t_data[:1001],c_r1_p2['b256N1k_a5k_l100'])
ax[1,1].plot(CVM_t_data[:1001],c_r1_p2['b512N1k_a5k_l100'])
ax[1,1].plot(CVM_t_data[:1001],CVM_c_data[-1,:1001])
ax[1,1].legend(legends,fontsize=font)
ax[1,1].set_xlabel("t ($s$)")
ax[1,1].set_ylabel("c ($mol$ $m^{-3}$)")
ax[1,1].set_title('Concentration at r=5e-6m')
plt.tight_layout()
plt.show()
fig.savefig('Model Plots/batches_predictions.png')

#%% dataset sizes
legends = ['N=1000', 'N=2000', 'N=4000', 'CVM Solution']
fig, ax = plt.subplots(2,2,figsize=(12, 8))
ax[0,0].plot(CVM_r_data,c_t1_p2['b128N1k_a5k_l100'])
ax[0,0].plot(CVM_r_data,c_t1_p2['b128N2k_a5k_l100'])
ax[0,0].plot(CVM_r_data,c_t1_p2['b128N4k_a5k_l100'])
ax[0,0].plot(CVM_r_data,CVM_c_data[...,1])
ax[0,0].legend(legends,fontsize=font)
ax[0,0].set_xlabel("r ($m$)")
ax[0,0].set_ylabel("c ($mol$ $m^{-3}$)")
ax[0,0].set_title('Concentration at t=1s')
ax[0,1].plot(CVM_r_data,c_t1000_p2['b128N1k_a5k_l100'])
ax[0,1].plot(CVM_r_data,c_t1000_p2['b128N2k_a5k_l100'])
ax[0,1].plot(CVM_r_data,c_t1000_p2['b128N4k_a5k_l100'])
ax[0,1].plot(CVM_r_data,CVM_c_data[...,1001])
ax[0,1].legend(legends,fontsize=font)
ax[0,1].set_xlabel("r ($m$)")
ax[0,1].set_ylabel("c ($mol$ $m^{-3}$)")
ax[0,1].set_title('Concentration at t=1000s')
ax[1,0].plot(CVM_t_data[:1001],c_r0_p2['b128N1k_a5k_l100'])
ax[1,0].plot(CVM_t_data[:1001],c_r0_p2['b128N2k_a5k_l100'])
ax[1,0].plot(CVM_t_data[:1001],c_r0_p2['b128N4k_a5k_l100'])
ax[1,0].plot(CVM_t_data[:1001],CVM_c_data[0,:1001])
ax[1,0].legend(legends,fontsize=font)
ax[1,0].set_xlabel("t ($s$)")
ax[1,0].set_ylabel("c ($mol$ $m^{-3}$)")
ax[1,0].set_title('Concentration at r=0m')
ax[1,1].plot(CVM_t_data[:1001],c_r1_p2['b128N1k_a5k_l100'])
ax[1,1].plot(CVM_t_data[:1001],c_r1_p2['b128N2k_a5k_l100'])
ax[1,1].plot(CVM_t_data[:1001],c_r1_p2['b128N4k_a5k_l100'])
ax[1,1].plot(CVM_t_data[:1001],CVM_c_data[-1,:1001])
ax[1,1].legend(legends,fontsize=font)
ax[1,1].set_xlabel("t ($s$)")
ax[1,1].set_ylabel("c ($mol$ $m^{-3}$)")
ax[1,1].set_title('Concentration at r=5e-6m')
plt.tight_layout()
plt.show()
fig.savefig('Model Plots/datasets_predictions.png')

fig, ax = plt.subplots(1,2,figsize=(12, 4))
ax[0].plot(CVM_r_data,c_t1660_p2['b128N1k_a5k_l100'])
ax[0].plot(CVM_r_data,c_t1660_p2['b128N2k_a5k_l100'])
ax[0].plot(CVM_r_data,c_t1660_p2['b128N4k_a5k_l100'])
ax[0].plot(CVM_r_data,CVM_c_data[...,1661])
ax[0].legend(legends,fontsize=font)
ax[0].set_xlabel("r ($m$)")
ax[0].set_ylabel("c ($mol$ $m^{-3}$)")
ax[0].set_title('Concentration at t=1660s')
ax[1].plot(CVM_r_data,c_t2000_p2['b128N1k_a5k_l100'])
ax[1].plot(CVM_r_data,c_t2000_p2['b128N2k_a5k_l100'])
ax[1].plot(CVM_r_data,c_t2000_p2['b128N4k_a5k_l100'])
ax[1].plot(CVM_r_data,CVM_c_data[...,-1])
ax[1].legend(legends,fontsize=font)
ax[1].set_xlabel("r ($m$)")
ax[1].set_ylabel("c ($mol$ $m^{-3}$)")
ax[1].set_title('Concentration at t=2000s')
plt.tight_layout()
plt.show()
fig.savefig('Model Plots/datasets_untrained.png')

#%% best solution without boundary forcing / adaptive sampling
legends = ['3000 Adam Epochs', 'CVM Solution']
fig, ax = plt.subplots(2,2,figsize=(12, 8))
ax[0,0].plot(CVM_r_data,c_t1_p2['b128N2k_a3k_l100'])
ax[0,0].plot(CVM_r_data,CVM_c_data[...,1])
ax[0,0].legend(legends,fontsize=font)
ax[0,0].set_xlabel("r ($m$)")
ax[0,0].set_ylabel("c ($mol$ $m^{-3}$)")
ax[0,0].set_title('Concentration at t=1s')
ax[0,1].plot(CVM_r_data,c_t1000_p2['b128N2k_a3k_l100'])
ax[0,1].plot(CVM_r_data,CVM_c_data[...,1001])
ax[0,1].legend(legends,fontsize=font)
ax[0,1].set_xlabel("r ($m$)")
ax[0,1].set_ylabel("c ($mol$ $m^{-3}$)")
ax[0,1].set_title('Concentration at t=1000s')
ax[1,0].plot(CVM_t_data[:1001],c_r0_p2['b128N2k_a3k_l100'])
ax[1,0].plot(CVM_t_data[:1001],CVM_c_data[0,:1001])
ax[1,0].legend(legends,fontsize=font)
ax[1,0].set_xlabel("t ($s$)")
ax[1,0].set_ylabel("c ($mol$ $m^{-3}$)")
ax[1,0].set_title('Concentration at r=0m')
ax[1,1].plot(CVM_t_data[:1001],c_r1_p2['b128N2k_a3k_l100'])
ax[1,1].plot(CVM_t_data[:1001],CVM_c_data[-1,:1001])
ax[1,1].legend(legends,fontsize=font)
ax[1,1].set_xlabel("t ($s$)")
ax[1,1].set_ylabel("c ($mol$ $m^{-3}$)")
ax[1,1].set_title('Concentration at r=5e-6m')
plt.tight_layout()
plt.show()
fig.savefig('Model Plots/epochs_predictions.png')

fig, ax = plt.subplots(1,2,figsize=(12, 4))
ax[0].plot(CVM_r_data,c_t1660_p2['b128N2k_a3k_l100'])
ax[0].plot(CVM_r_data,CVM_c_data[...,1661])
ax[0].legend(legends,fontsize=font)
ax[0].set_xlabel("r ($m$)")
ax[0].set_ylabel("c ($mol$ $m^{-3}$)")
ax[0].set_title('Concentration at t=1660s')
ax[1].plot(CVM_r_data,c_t2000_p2['b128N2k_a3k_l100'])
ax[1].plot(CVM_r_data,CVM_c_data[...,-1])
ax[1].legend(legends,fontsize=font)
ax[1].set_xlabel("r ($m$)")
ax[1].set_ylabel("c ($mol$ $m^{-3}$)")
ax[1].set_title('Concentration at t=2000s')
plt.tight_layout()
plt.show()
fig.savefig('Model Plots/epochs_untrained.png')

#%% adaptive sampling
legends = ['3000 Adam Epochs', 'Adaptive After 1000 Adam Epochs', 'Adaptive After 2000 Adam Epochs', 'CVM Solution']
fig, ax = plt.subplots(2,2,figsize=(12, 8))
ax[0,0].plot(CVM_r_data,c_t1_p3['no_ad'])
ax[0,0].plot(CVM_r_data,c_t1_p3['1000_ad4x500'])
ax[0,0].plot(CVM_r_data,c_t1_p3['2000_ad2x500'])
ax[0,0].plot(CVM_r_data,CVM_c_data[...,1])
ax[0,0].legend(legends,fontsize=font)
ax[0,0].set_xlabel("r ($m$)")
ax[0,0].set_ylabel("c ($mol$ $m^{-3}$)")
ax[0,0].set_title('Concentration at t=1s')
ax[0,1].plot(CVM_r_data,c_t1000_p3['no_ad'])
ax[0,1].plot(CVM_r_data,c_t1000_p3['1000_ad4x500'])
ax[0,1].plot(CVM_r_data,c_t1000_p3['2000_ad2x500'])
ax[0,1].plot(CVM_r_data,CVM_c_data[...,1001])
ax[0,1].legend(legends,fontsize=font)
ax[0,1].set_xlabel("r ($m$)")
ax[0,1].set_ylabel("c ($mol$ $m^{-3}$)")
ax[0,1].set_title('Concentration at t=1000s')
ax[1,0].plot(CVM_t_data[:1001],c_r0_p3['no_ad'])
ax[1,0].plot(CVM_t_data[:1001],c_r0_p3['1000_ad4x500'])
ax[1,0].plot(CVM_t_data[:1001],c_r0_p3['2000_ad2x500'])
ax[1,0].plot(CVM_t_data[:1001],CVM_c_data[0,:1001])
ax[1,0].legend(legends,fontsize=font)
ax[1,0].set_xlabel("t ($s$)")
ax[1,0].set_ylabel("c ($mol$ $m^{-3}$)")
ax[1,0].set_title('Concentration at r=0m')
ax[1,1].plot(CVM_t_data[:1001],c_r1_p3['no_ad'])
ax[1,1].plot(CVM_t_data[:1001],c_r1_p3['1000_ad4x500'])
ax[1,1].plot(CVM_t_data[:1001],c_r1_p3['2000_ad2x500'])
ax[1,1].plot(CVM_t_data[:1001],CVM_c_data[-1,:1001])
ax[1,1].legend(legends,fontsize=font)
ax[1,1].set_xlabel("t ($s$)")
ax[1,1].set_ylabel("c ($mol$ $m^{-3}$)")
ax[1,1].set_title('Concentration at r=5e-6m')
plt.tight_layout()
plt.show()
fig.savefig('Model Plots/adaptive_predictions.png')

#%% boundary forcing
legends = ['No Boundary Forcing', 't0 Boundary Forcing', 'r0 Boundary Forcing', 't0 and r0 Boundary Forcing', 'CVM Solution']
fig, ax = plt.subplots(2,2,figsize=(12, 8))
ax[0,0].plot(CVM_r_data,c_t1_p4['none_3000'])
ax[0,0].plot(CVM_r_data,c_t1_p4['t0_3000'])
ax[0,0].plot(CVM_r_data,c_t1_p4['r0_3000'])
ax[0,0].plot(CVM_r_data,c_t1_p4['t0r0_3000'])
ax[0,0].plot(CVM_r_data,CVM_c_data[...,1])
ax[0,0].legend(legends,fontsize=font)
ax[0,0].set_xlabel("r ($m$)")
ax[0,0].set_ylabel("c ($mol$ $m^{-3}$)")
ax[0,0].set_title('Concentration at t=1s')
ax[0,1].plot(CVM_r_data,c_t1000_p4['none_3000'])
ax[0,1].plot(CVM_r_data,c_t1000_p4['t0_3000'])
ax[0,1].plot(CVM_r_data,c_t1000_p4['r0_3000'])
ax[0,1].plot(CVM_r_data,c_t1000_p4['t0r0_3000'])
ax[0,1].plot(CVM_r_data,CVM_c_data[...,1001])
ax[0,1].legend(legends,fontsize=font)
ax[0,1].set_xlabel("r ($m$)")
ax[0,1].set_ylabel("c ($mol$ $m^{-3}$)")
ax[0,1].set_title('Concentration at t=1000s')
ax[1,0].plot(CVM_t_data[:1001],c_r0_p4['none_3000'])
ax[1,0].plot(CVM_t_data[:1001],c_r0_p4['t0_3000'])
ax[1,0].plot(CVM_t_data[:1001],c_r0_p4['r0_3000'])
ax[1,0].plot(CVM_t_data[:1001],c_r0_p4['t0r0_3000'])
ax[1,0].plot(CVM_t_data[:1001],CVM_c_data[0,:1001])
ax[1,0].legend(legends,fontsize=font)
ax[1,0].set_xlabel("t ($s$)")
ax[1,0].set_ylabel("c ($mol$ $m^{-3}$)")
ax[1,0].set_title('Concentration at r=0m')
ax[1,1].plot(CVM_t_data[:1001],c_r1_p4['none_3000'])
ax[1,1].plot(CVM_t_data[:1001],c_r1_p4['t0_3000'])
ax[1,1].plot(CVM_t_data[:1001],c_r1_p4['r0_3000'])
ax[1,1].plot(CVM_t_data[:1001],c_r1_p4['t0r0_3000'])
ax[1,1].plot(CVM_t_data[:1001],CVM_c_data[-1,:1001])
ax[1,1].legend(legends,fontsize=font)
ax[1,1].set_xlabel("t ($s$)")
ax[1,1].set_ylabel("c ($mol$ $m^{-3}$)")
ax[1,1].set_title('Concentration at r=5e-6m')
plt.tight_layout()
plt.show()
fig.savefig('Model Plots/boundary_predictions.png')

#%% boundary forcing at t0
legends = ['N=2000, 3000 Adam Epochs', 'N=2000, 5000 Adam Epochs', 'CVM Solution']
fig, ax = plt.subplots(2,2,figsize=(12, 8))
ax[0,0].plot(CVM_r_data,c_t1_p4['t0_3000'])
ax[0,0].plot(CVM_r_data,c_t1_p4['t0_5000'])
ax[0,0].plot(CVM_r_data,CVM_c_data[...,1])
ax[0,0].legend(legends,fontsize=font)
ax[0,0].set_xlabel("r ($m$)")
ax[0,0].set_ylabel("c ($mol$ $m^{-3}$)")
ax[0,0].set_title('Concentration at t=1s')
ax[0,1].plot(CVM_r_data,c_t1000_p4['t0_3000'])
ax[0,1].plot(CVM_r_data,c_t1000_p4['t0_5000'])
ax[0,1].plot(CVM_r_data,CVM_c_data[...,1001])
ax[0,1].legend(legends,fontsize=font)
ax[0,1].set_xlabel("r ($m$)")
ax[0,1].set_ylabel("c ($mol$ $m^{-3}$)")
ax[0,1].set_title('Concentration at t=1000s')
ax[1,0].plot(CVM_t_data[:1001],c_r0_p4['t0_3000'])
ax[1,0].plot(CVM_t_data[:1001],c_r0_p4['t0_5000'])
ax[1,0].plot(CVM_t_data[:1001],CVM_c_data[0,:1001])
ax[1,0].legend(legends,fontsize=font)
ax[1,0].set_xlabel("t ($s$)")
ax[1,0].set_ylabel("c ($mol$ $m^{-3}$)")
ax[1,0].set_title('Concentration at r=0m')
ax[1,1].plot(CVM_t_data[:1001],c_r1_p4['t0_3000'])
ax[1,1].plot(CVM_t_data[:1001],c_r1_p4['t0_5000'])
ax[1,1].plot(CVM_t_data[:1001],CVM_c_data[-1,:1001])
ax[1,1].legend(legends,fontsize=font)
ax[1,1].set_xlabel("t ($s$)")
ax[1,1].set_ylabel("c ($mol$ $m^{-3}$)")
ax[1,1].set_title('Concentration at r=5e-6m')
plt.tight_layout()
plt.show()
fig.savefig('Model Plots/t0_predictions.png')

fig, ax = plt.subplots(1,2,figsize=(12, 4))
ax[0].plot(CVM_r_data,c_t1660_p4['t0_3000'])
ax[0].plot(CVM_r_data,c_t1660_p4['t0_5000'])
ax[0].plot(CVM_r_data,CVM_c_data[...,1661])
ax[0].legend(legends,fontsize=font)
ax[0].set_xlabel("r ($m$)")
ax[0].set_ylabel("c ($mol$ $m^{-3}$)")
ax[0].set_title('Concentration at t=1660s')
ax[1].plot(CVM_r_data,c_t2000_p4['t0_3000'])
ax[1].plot(CVM_r_data,c_t2000_p4['t0_5000'])
ax[1].plot(CVM_r_data,CVM_c_data[...,-1])
ax[1].legend(legends,fontsize=font)
ax[1].set_xlabel("r ($m$)")
ax[1].set_ylabel("c ($mol$ $m^{-3}$)")
ax[1].set_title('Concentration at t=2000s')
plt.tight_layout()
plt.show()
fig.savefig('Model Plots/t0_untrained.png')

#%% heatmaps
# parameters from [Zhang et al., JES, 2007]
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
In = -2 # Current density [A/m2], negative for charging
theta = (omega/R/T)*(2*omega*E/9/(1-nu)) # [m3/mol]

# equation normalisation and network parameters
tmax = 1000
I = In*Rs/(D*c_max*F)
theta_norm = theta*c_max
randomseed = 12121
input_dim = 2
output_dim = 1
layers = 4
nodes = 256

# CVM solution values
net_r, net_t, ms_r, ms_t = PINN.meshgrid_plot(CVM_r_data/Rs,CVM_t_data/tmax)

# best without boundary forcing
name = 'PINN Models/PINN_state_4x256tanh_b128N2k_a3k_l100_tmax1000.pt'
net = PINN.PINN(input_dim, output_dim, lays=layers, n_units=nodes, device=device, seed=randomseed).to(device)
net.vals(tmax=tmax,rmax=Rs,cmax=c_max,D=D,theta_norm=theta_norm)
net.load_state_dict(torch.load(name))
net.eval()
c_pred = net.predict(net_r, net_t)
c_pred = c_pred.detach().cpu().numpy()
ms_c = c_pred.reshape(ms_r.shape)
val_err = np.absolute(ms_c-CVM_c_data/c_max)

# plot and save figure
fig, ax = plt.subplots(figsize=(6, 4))
val_img = ax.pcolormesh(Rs*ms_r[:,:1001], tmax*ms_t[:,:1001], val_err[:,:1001], cmap='jet',  vmin=0, vmax=0.01)
ax.set_title('Absolute Validation Loss')
ax.axis([CVM_r_data[0], CVM_r_data[-1], CVM_t_data[0], CVM_t_data[1001]])
fig.colorbar(val_img, ax=ax, label='Loss Magnitude')
ax.set_xlabel("r ($m$)")
ax.set_ylabel("t ($s$)")
plt.show()
fig.savefig('Model Plots/epochs_heatmap.png')

# with boundary forcing
name = 'PINN_state_4x256tanh_b128N2k_a5k_l100_tmax1000_t0.pt'
bf = 't0'
net = PINN.PINN(input_dim, output_dim, lays=layers, n_units=nodes, boundary_force=bf, device=device, seed=randomseed).to(device)
net.vals(tmax=tmax,rmax=Rs,cmax=c_max,D=D,theta_norm=theta_norm)
net.load_state_dict(torch.load(name))
net.eval()
c_pred = net.predict(net_r, net_t)
c_pred = c_pred.detach().cpu().numpy()
ms_c = c_pred.reshape(ms_r.shape)
val_err = np.absolute(ms_c-CVM_c_data/c_max)

# plot and save figure
fig, ax = plt.subplots(figsize=(6, 4))
val_img = ax.pcolormesh(Rs*ms_r[:,:1001], tmax*ms_t[:,:1001], val_err[:,:1001], cmap='jet',  vmin=0, vmax=0.01)
ax.set_title('Absolute Validation Loss')
ax.axis([CVM_r_data[0], CVM_r_data[-1], CVM_t_data[0], CVM_t_data[1001]])
fig.colorbar(val_img, ax=ax, label='Loss Magnitude')
ax.set_xlabel("r ($m$)")
ax.set_ylabel("t ($s$)")
plt.show()
fig.savefig('Model Plots/t0_heatmap.png')

fig = plt.figure(0, figsize=(6,4))
plt.plot(CVM_t_data[:1661], val_err[-1,:1661])
plt.vlines(x=1000, ymin=0, ymax=0.022, colors='k', linestyles='dashed')
plt.title('Validation Loss Magnitude at r=5e-6')
plt.text(200, 0.0175, 'Inside Training Domain', fontsize=font)
plt.text(1050, 0.0175, 'Outside Training Domain', fontsize=font)
plt.ylim([0,0.022])
plt.legend(['Surface Loss Magnitude'])
plt.xlabel('Time (Seconds)')
plt.ylabel('Loss Magnitude')
plt.tight_layout()
plt.show()
fig.savefig("Model Plots/t0_lossplot.png")
