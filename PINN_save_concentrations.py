import numpy as np
import pandas as pd
import torch

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
In = -2 # Current density [A/m2], negative for charging
theta = (omega/R/T)*(2*omega*E/9/(1-nu)) # [m3/mol]

# Equation normalisation and network parameters
tmax = 1000
I = In*Rs/(D*c_max*F)
theta_norm = theta*c_max
randomseed = 12121
input_dim = 2
output_dim = 1

# Load validation files
CVM_r_data = np.loadtxt('CVM_r.csv', delimiter=',')
CVM_t_data = np.loadtxt('CVM_t.csv', delimiter=',')

#%% Network parameters
# Initialise dataframes
c_t1_p1 = pd.DataFrame()
c_t1000_p1 = pd.DataFrame()
c_t1660_p1 = pd.DataFrame()
c_t2000_p1 = pd.DataFrame()
c_r0_p1 = pd.DataFrame()
c_r1_p1 = pd.DataFrame()

# Network sizes
file1 = 'PINN Models/PINN_state_3x100tanh_b128N1k_a5k_l100_tmax1000.pt'
file2 = 'PINN Models/PINN_state_3x256tanh_b128N1k_a5k_l100_tmax1000.pt'
file3 = 'PINN Models/PINN_state_4x100tanh_b128N1k_a5k_l100_tmax1000.pt'
file4 = 'PINN Models/PINN_state_4x256tanh_b128N1k_a5k_l100_tmax1000.pt'
files = [file1, file2, file3, file4]
layers = [3, 3, 4, 4]
nodes = [100, 256, 100, 256]
for i in range(len(files)):
    net = PINN.PINN(input_dim, output_dim, lays=layers[i], n_units=nodes[i], device=device, seed=randomseed).to(device)
    net.vals(tmax=tmax,rmax=Rs,cmax=c_max,D=D,theta_norm=theta_norm)
    net.load_state_dict(torch.load(files[i]))
    net.eval()

    net_r, net_t, ms_r, ms_t = PINN.meshgrid_plot(CVM_r_data/Rs,CVM_t_data/tmax)
    c_pred = net.predict(net_r, net_t)
    c_pred = c_pred.detach().cpu().numpy()
    ms_c = c_pred.reshape(ms_r.shape)
    
    df_t1 = pd.DataFrame(data=c_max*ms_c[...,1], columns=[f'{layers[i]}x{nodes[i]}tanh'])
    df_t1000 = pd.DataFrame(data=c_max*ms_c[...,1001], columns=[f'{layers[i]}x{nodes[i]}tanh'])
    df_t1660 = pd.DataFrame(data=c_max*ms_c[...,1661], columns=[f'{layers[i]}x{nodes[i]}tanh'])
    df_t2000 = pd.DataFrame(data=c_max*ms_c[...,-1], columns=[f'{layers[i]}x{nodes[i]}tanh'])
    df_r0 = pd.DataFrame(data=c_max*ms_c[0,:1001], columns=[f'{layers[i]}x{nodes[i]}tanh'])
    df_r1 = pd.DataFrame(data=c_max*ms_c[-1,:1001], columns=[f'{layers[i]}x{nodes[i]}tanh'])
    c_t1_p1 = pd.concat([c_t1_p1,df_t1], axis=1)
    c_t1000_p1 = pd.concat([c_t1000_p1,df_t1000], axis=1)
    c_t1660_p1 = pd.concat([c_t1660_p1,df_t1660], axis=1)
    c_t2000_p1 = pd.concat([c_t2000_p1,df_t2000], axis=1)
    c_r0_p1 = pd.concat([c_r0_p1,df_r0], axis=1)
    c_r1_p1 = pd.concat([c_r1_p1,df_r1], axis=1)

# Sin activation function
file = 'PINN Models/PINN_state_4x256sin_b128N1k_a5k_l100_tmax1000.pt'
net = PINN.PINN(input_dim, output_dim, lays=4, n_units=256, act_func=PINN.Sin, device=device, seed=randomseed).to(device)
net.vals(tmax=tmax,rmax=Rs,cmax=c_max,D=D,theta_norm=theta_norm)
net.load_state_dict(torch.load(file))
net.eval()

net_r, net_t, ms_r, ms_t = PINN.meshgrid_plot(CVM_r_data/Rs,CVM_t_data/tmax)
c_pred = net.predict(net_r, net_t)
c_pred = c_pred.detach().cpu().numpy()
ms_c = c_pred.reshape(ms_r.shape)

df_t1 = pd.DataFrame(data=c_max*ms_c[...,1], columns=['4x256sin'])
df_t1000 = pd.DataFrame(data=c_max*ms_c[...,1001], columns=['4x256sin'])
df_t1660 = pd.DataFrame(data=c_max*ms_c[...,1661], columns=['4x256sin'])
df_t2000 = pd.DataFrame(data=c_max*ms_c[...,-1], columns=['4x256sin'])
df_r0 = pd.DataFrame(data=c_max*ms_c[0,:1001], columns=['4x256sin'])
df_r1 = pd.DataFrame(data=c_max*ms_c[-1,:1001], columns=['4x256sin'])
c_t1_p1 = pd.concat([c_t1_p1,df_t1], axis=1)
c_t1000_p1 = pd.concat([c_t1000_p1,df_t1000], axis=1)
c_t1660_p1 = pd.concat([c_t1660_p1,df_t1660], axis=1)
c_t2000_p1 = pd.concat([c_t2000_p1,df_t2000], axis=1)
c_r0_p1 = pd.concat([c_r0_p1,df_r0], axis=1)
c_r1_p1 = pd.concat([c_r1_p1,df_r1], axis=1)

# Save to excel
with pd.ExcelWriter('PINN Models/PINN_net_params.xlsx', engine='xlsxwriter') as writer:
    c_t1_p1.to_excel(writer, sheet_name='t1')
    c_t1000_p1.to_excel(writer, sheet_name='t1000')
    c_t1660_p1.to_excel(writer, sheet_name='t1660')
    c_t2000_p1.to_excel(writer, sheet_name='t2000')
    c_r0_p1.to_excel(writer, sheet_name='r0')
    c_r1_p1.to_excel(writer, sheet_name='r1')

# Final network sizes
layers = 4
neurons = 256

#%% datasets and epochs
# Initialise dataframes
c_t1_p2 = pd.DataFrame()
c_t1000_p2 = pd.DataFrame()
c_t1660_p2 = pd.DataFrame()
c_t2000_p2 = pd.DataFrame()
c_r0_p2 = pd.DataFrame()
c_r1_p2 = pd.DataFrame()

df_t1 = pd.DataFrame(data=c_t1_p1['4x256tanh'])
df_t1000 = pd.DataFrame(data=c_t1000_p1['4x256tanh'])
df_t1660 = pd.DataFrame(data=c_t1660_p1['4x256tanh'])
df_t2000 = pd.DataFrame(data=c_t2000_p1['4x256tanh'])
df_r0 = pd.DataFrame(data=c_r0_p1['4x256tanh'])
df_r1 = pd.DataFrame(data=c_r1_p1['4x256tanh'])
df_t1.rename(columns={'4x256tanh':'b128N1k_a5k_l100'}, inplace=True)
df_t1000.rename(columns={'4x256tanh':'b128N1k_a5k_l100'}, inplace=True)
df_t1660.rename(columns={'4x256tanh':'b128N1k_a5k_l100'}, inplace=True)
df_t2000.rename(columns={'4x256tanh':'b128N1k_a5k_l100'}, inplace=True)
df_r0.rename(columns={'4x256tanh':'b128N1k_a5k_l100'}, inplace=True)
df_r1.rename(columns={'4x256tanh':'b128N1k_a5k_l100'}, inplace=True)
c_t1_p2 = pd.concat([c_t1_p2,df_t1], axis=1)
c_t1000_p2 = pd.concat([c_t1000_p2,df_t1000], axis=1)
c_t1660_p2 = pd.concat([c_t1660_p2,df_t1660], axis=1)
c_t2000_p2 = pd.concat([c_t2000_p2,df_t2000], axis=1)
c_r0_p2 = pd.concat([c_r0_p2,df_r0], axis=1)
c_r1_p2 = pd.concat([c_r1_p2,df_r1], axis=1)

# Batch sizes
file1 = 'PINN Models/PINN_state_4x256tanh_b256N1k_a5k_l100_tmax1000.pt'
file2 = 'PINN Models/PINN_state_4x256tanh_b512N1k_a5k_l100_tmax1000.pt'
files = [file1, file2]
batch = [256, 512]
for i in range(len(files)):
    net = PINN.PINN(input_dim, output_dim, lays=layers, n_units=neurons, device=device, seed=randomseed).to(device)
    net.vals(tmax=tmax,rmax=Rs,cmax=c_max,D=D,theta_norm=theta_norm)
    net.load_state_dict(torch.load(files[i]))
    net.eval()

    net_r, net_t, ms_r, ms_t = PINN.meshgrid_plot(CVM_r_data/Rs,CVM_t_data/tmax)
    c_pred = net.predict(net_r, net_t)
    c_pred = c_pred.detach().cpu().numpy()
    ms_c = c_pred.reshape(ms_r.shape)
    
    df_t1 = pd.DataFrame(data=c_max*ms_c[...,1], columns=[f'b{batch[i]}N1k_a5k_l100'])
    df_t1000 = pd.DataFrame(data=c_max*ms_c[...,1001], columns=[f'b{batch[i]}N1k_a5k_l100'])
    df_t1660 = pd.DataFrame(data=c_max*ms_c[...,1661], columns=[f'b{batch[i]}N1k_a5k_l100'])
    df_t2000 = pd.DataFrame(data=c_max*ms_c[...,-1], columns=[f'b{batch[i]}N1k_a5k_l100'])
    df_r0 = pd.DataFrame(data=c_max*ms_c[0,:1001], columns=[f'b{batch[i]}N1k_a5k_l100'])
    df_r1 = pd.DataFrame(data=c_max*ms_c[-1,:1001], columns=[f'b{batch[i]}N1k_a5k_l100'])
    c_t1_p2 = pd.concat([c_t1_p2,df_t1], axis=1)
    c_t1000_p2 = pd.concat([c_t1000_p2,df_t1000], axis=1)
    c_t1660_p2 = pd.concat([c_t1660_p2,df_t1660], axis=1)
    c_t2000_p2 = pd.concat([c_t2000_p2,df_t2000], axis=1)
    c_r0_p2 = pd.concat([c_r0_p2,df_r0], axis=1)
    c_r1_p2 = pd.concat([c_r1_p2,df_r1], axis=1)

# Dataset sizes
file1 = 'PINN Models/PINN_state_4x256tanh_b128N2k_a5k_l100_tmax1000.pt'
file2 = 'PINN Models/PINN_state_4x256tanh_b128N4k_a5k_l100_tmax1000.pt'
files = [file1, file2]
N = [2, 4]
for i in range(len(files)):
    net = PINN.PINN(input_dim, output_dim, lays=layers, n_units=neurons, device=device, seed=randomseed).to(device)
    net.vals(tmax=tmax,rmax=Rs,cmax=c_max,D=D,theta_norm=theta_norm)
    net.load_state_dict(torch.load(files[i]))
    net.eval()

    net_r, net_t, ms_r, ms_t = PINN.meshgrid_plot(CVM_r_data/Rs,CVM_t_data/tmax)
    c_pred = net.predict(net_r, net_t)
    c_pred = c_pred.detach().cpu().numpy()
    ms_c = c_pred.reshape(ms_r.shape)
    
    df_t1 = pd.DataFrame(data=c_max*ms_c[...,1], columns=[f'b128N{N[i]}k_a5k_l100'])
    df_t1000 = pd.DataFrame(data=c_max*ms_c[...,1001], columns=[f'b128N{N[i]}k_a5k_l100'])
    df_t1660 = pd.DataFrame(data=c_max*ms_c[...,1661], columns=[f'b128N{N[i]}k_a5k_l100'])
    df_t2000 = pd.DataFrame(data=c_max*ms_c[...,-1], columns=[f'b128N{N[i]}k_a5k_l100'])
    df_r0 = pd.DataFrame(data=c_max*ms_c[0,:1001], columns=[f'b128N{N[i]}k_a5k_l100'])
    df_r1 = pd.DataFrame(data=c_max*ms_c[-1,:1001], columns=[f'b128N{N[i]}k_a5k_l100'])
    c_t1_p2 = pd.concat([c_t1_p2,df_t1], axis=1)
    c_t1000_p2 = pd.concat([c_t1000_p2,df_t1000], axis=1)
    c_t1660_p2 = pd.concat([c_t1660_p2,df_t1660], axis=1)
    c_t2000_p2 = pd.concat([c_t2000_p2,df_t2000], axis=1)
    c_r0_p2 = pd.concat([c_r0_p2,df_r0], axis=1)
    c_r1_p2 = pd.concat([c_r1_p2,df_r1], axis=1)

# 3000 adam epochs
file = 'PINN Models/PINN_state_4x256tanh_b128N2k_a3k_l100_tmax1000.pt'
net = PINN.PINN(input_dim, output_dim, lays=layers, n_units=neurons, device=device, seed=randomseed).to(device)
net.vals(tmax=tmax,rmax=Rs,cmax=c_max,D=D,theta_norm=theta_norm)
net.load_state_dict(torch.load(file))
net.eval()

net_r, net_t, ms_r, ms_t = PINN.meshgrid_plot(CVM_r_data/Rs,CVM_t_data/tmax)
c_pred = net.predict(net_r, net_t)
c_pred = c_pred.detach().cpu().numpy()
ms_c = c_pred.reshape(ms_r.shape)

df_t1 = pd.DataFrame(data=c_max*ms_c[...,1], columns=['b128N2k_a3k_l100'])
df_t1000 = pd.DataFrame(data=c_max*ms_c[...,1001], columns=['b128N2k_a3k_l100'])
df_t1660 = pd.DataFrame(data=c_max*ms_c[...,1661], columns=['b128N2k_a3k_l100'])
df_t2000 = pd.DataFrame(data=c_max*ms_c[...,-1], columns=['b128N2k_a3k_l100'])
df_r0 = pd.DataFrame(data=c_max*ms_c[0,:1001], columns=['b128N2k_a3k_l100'])
df_r1 = pd.DataFrame(data=c_max*ms_c[-1,:1001], columns=['b128N2k_a3k_l100'])
c_t1_p2 = pd.concat([c_t1_p2,df_t1], axis=1)
c_t1000_p2 = pd.concat([c_t1000_p2,df_t1000], axis=1)
c_t1660_p2 = pd.concat([c_t1660_p2,df_t1660], axis=1)
c_t2000_p2 = pd.concat([c_t2000_p2,df_t2000], axis=1)
c_r0_p2 = pd.concat([c_r0_p2,df_r0], axis=1)
c_r1_p2 = pd.concat([c_r1_p2,df_r1], axis=1)

# Save to excel
with pd.ExcelWriter('PINN Models/PINN_dataset_epochs.xlsx', engine='xlsxwriter') as writer:
    c_t1_p2.to_excel(writer, sheet_name='t1')
    c_t1000_p2.to_excel(writer, sheet_name='t1000')
    c_t1660_p2.to_excel(writer, sheet_name='t1660')
    c_t2000_p2.to_excel(writer, sheet_name='t2000')
    c_r0_p2.to_excel(writer, sheet_name='r0')
    c_r1_p2.to_excel(writer, sheet_name='r1')

#%% Adaptive sampling
# Initialise dataframes
c_t1_p3 = pd.DataFrame()
c_t1000_p3 = pd.DataFrame()
c_r0_p3 = pd.DataFrame()
c_r1_p3 = pd.DataFrame()

df_t1 = pd.DataFrame(data=c_t1_p2['b128N2k_a3k_l100'])
df_t1000 = pd.DataFrame(data=c_t1000_p2['b128N2k_a3k_l100'])
df_r0 = pd.DataFrame(data=c_r0_p2['b128N2k_a3k_l100'])
df_r1 = pd.DataFrame(data=c_r1_p2['b128N2k_a3k_l100'])
df_t1.rename(columns={'b128N2k_a3k_l100':'no_ad'}, inplace=True)
df_t1000.rename(columns={'b128N2k_a3k_l100':'no_ad'}, inplace=True)
df_r0.rename(columns={'b128N2k_a3k_l100':'no_ad'}, inplace=True)
df_r1.rename(columns={'b128N2k_a3k_l100':'no_ad'}, inplace=True)
c_t1_p3 = pd.concat([c_t1_p3,df_t1], axis=1)
c_t1000_p3 = pd.concat([c_t1000_p3,df_t1000], axis=1)
c_r0_p3 = pd.concat([c_r0_p3,df_r0], axis=1)
c_r1_p3 = pd.concat([c_r1_p3,df_r1], axis=1)

# Adaptive sampling files
file1 = 'PINN Models/PINN_state_4x256tanh_b128N2k_a1k_l100_tmax1000_ad500a_4runs.pt'
file2 = 'PINN Models/PINN_state_4x256tanh_b128N2k_a2k_l100_tmax1000_ad500a_2runs.pt'
files = [file1, file2]
ad = ['1000_ad4x500', '2000_ad2x500']
for i in range(len(files)):
    
    net = PINN.PINN(input_dim, output_dim, lays=layers, n_units=neurons, device=device, seed=randomseed).to(device)
    net.vals(tmax=tmax,rmax=Rs,cmax=c_max,D=D,theta_norm=theta_norm)
    net.load_state_dict(torch.load(files[i]))
    net.eval()

    net_r, net_t, ms_r, ms_t = PINN.meshgrid_plot(CVM_r_data/Rs,CVM_t_data/tmax)
    c_pred = net.predict(net_r, net_t)
    c_pred = c_pred.detach().cpu().numpy()
    ms_c = c_pred.reshape(ms_r.shape)
    
    df_t1 = pd.DataFrame(data=c_max*ms_c[...,1], columns=[ad[i]])
    df_t1000 = pd.DataFrame(data=c_max*ms_c[...,1001], columns=[ad[i]])
    df_r0 = pd.DataFrame(data=c_max*ms_c[0,:1001], columns=[ad[i]])
    df_r1 = pd.DataFrame(data=c_max*ms_c[-1,:1001], columns=[ad[i]])
    c_t1_p3 = pd.concat([c_t1_p3,df_t1], axis=1)
    c_t1000_p3 = pd.concat([c_t1000_p3,df_t1000], axis=1)
    c_r0_p3 = pd.concat([c_r0_p3,df_r0], axis=1)
    c_r1_p3 = pd.concat([c_r1_p3,df_r1], axis=1)

# Save to excel
with pd.ExcelWriter('PINN Models/PINN_adaptive_sampling.xlsx', engine='xlsxwriter') as writer:
    c_t1_p3.to_excel(writer, sheet_name='t1')
    c_t1000_p3.to_excel(writer, sheet_name='t1000')
    c_r0_p3.to_excel(writer, sheet_name='r0')
    c_r1_p3.to_excel(writer, sheet_name='r1')

#%% Boundary forcing
# Initialise dataframes
c_t1_p4 = pd.DataFrame()
c_t1000_p4 = pd.DataFrame()
c_t1660_p4 = pd.DataFrame()
c_t2000_p4 = pd.DataFrame()
c_r0_p4 = pd.DataFrame()
c_r1_p4 = pd.DataFrame()

df_t1 = pd.DataFrame(data=c_t1_p2['b128N2k_a3k_l100'])
df_t1000 = pd.DataFrame(data=c_t1000_p2['b128N2k_a3k_l100'])
df_t1660 = pd.DataFrame(data=c_t1660_p2['b128N2k_a3k_l100'])
df_t2000 = pd.DataFrame(data=c_t2000_p2['b128N2k_a3k_l100'])
df_r0 = pd.DataFrame(data=c_r0_p2['b128N2k_a3k_l100'])
df_r1 = pd.DataFrame(data=c_r1_p2['b128N2k_a3k_l100'])
df_t1.rename(columns={'b128N2k_a3k_l100':'none_3000'}, inplace=True)
df_t1000.rename(columns={'b128N2k_a3k_l100':'none_3000'}, inplace=True)
df_t1660.rename(columns={'b128N2k_a3k_l100':'none_3000'}, inplace=True)
df_t2000.rename(columns={'b128N2k_a3k_l100':'none_3000'}, inplace=True)
df_r0.rename(columns={'b128N2k_a3k_l100':'none_3000'}, inplace=True)
df_r1.rename(columns={'b128N2k_a3k_l100':'none_3000'}, inplace=True)
c_t1_p4 = pd.concat([c_t1_p4,df_t1], axis=1)
c_t1000_p4 = pd.concat([c_t1000_p4,df_t1000], axis=1)
c_t1660_p4 = pd.concat([c_t1660_p4,df_t1660], axis=1)
c_t2000_p4 = pd.concat([c_t2000_p4,df_t2000], axis=1)
c_r0_p4 = pd.concat([c_r0_p4,df_r0], axis=1)
c_r1_p4 = pd.concat([c_r1_p4,df_r1], axis=1)

# Boundary forcing files
file1 = 'PINN Models/PINN_state_4x256tanh_b128N2k_a3k_l100_tmax1000_t0.pt'
file2 = 'PINN Models/PINN_state_4x256tanh_b128N2k_a3k_l100_tmax1000_r0.pt'
file3 = 'PINN Models/PINN_state_4x256tanh_b128N2k_a3k_l100_tmax1000_t0r0.pt'
files = [file1, file2, file3]
bf = ['t0', 'r0', 't0r0']
for i in range(len(files)):
    
    net = PINN.PINN(input_dim, output_dim, lays=layers, n_units=neurons, boundary_force=bf[i], device=device, seed=randomseed).to(device)
    net.vals(tmax=tmax,rmax=Rs,cmax=c_max,D=D,theta_norm=theta_norm)
    net.load_state_dict(torch.load(files[i]))
    net.eval()

    net_r, net_t, ms_r, ms_t = PINN.meshgrid_plot(CVM_r_data/Rs,CVM_t_data/tmax)
    c_pred = net.predict(net_r, net_t)
    c_pred = c_pred.detach().cpu().numpy()
    ms_c = c_pred.reshape(ms_r.shape)
    
    df_t1 = pd.DataFrame(data=c_max*ms_c[...,1], columns=[f'{bf[i]}_3000'])
    df_t1000 = pd.DataFrame(data=c_max*ms_c[...,1001], columns=[f'{bf[i]}_3000'])
    df_t1660 = pd.DataFrame(data=c_max*ms_c[...,1661], columns=[f'{bf[i]}_3000'])
    df_t2000 = pd.DataFrame(data=c_max*ms_c[...,-1], columns=[f'{bf[i]}_3000'])
    df_r0 = pd.DataFrame(data=c_max*ms_c[0,:1001], columns=[f'{bf[i]}_3000'])
    df_r1 = pd.DataFrame(data=c_max*ms_c[-1,:1001], columns=[f'{bf[i]}_3000'])
    c_t1_p4 = pd.concat([c_t1_p4,df_t1], axis=1)
    c_t1000_p4 = pd.concat([c_t1000_p4,df_t1000], axis=1)
    c_t1660_p4 = pd.concat([c_t1660_p4,df_t1660], axis=1)
    c_t2000_p4 = pd.concat([c_t2000_p4,df_t2000], axis=1)
    c_r0_p4 = pd.concat([c_r0_p4,df_r0], axis=1)
    c_r1_p4 = pd.concat([c_r1_p4,df_r1], axis=1)

# t0 boundary function for 5k adam epochs
file = 'PINN_state_4x256tanh_b128N2k_a5k_l100_tmax1000_t0.pt'
net = PINN.PINN(input_dim, output_dim, lays=layers, n_units=neurons, boundary_force='t0',device=device, seed=randomseed).to(device)
net.vals(tmax=tmax,rmax=Rs,cmax=c_max,D=D,theta_norm=theta_norm)
net.load_state_dict(torch.load(file))
net.eval()

net_r, net_t, ms_r, ms_t = PINN.meshgrid_plot(CVM_r_data/Rs,CVM_t_data/tmax)
c_pred = net.predict(net_r, net_t)
c_pred = c_pred.detach().cpu().numpy()
ms_c = c_pred.reshape(ms_r.shape)

df_t1 = pd.DataFrame(data=c_max*ms_c[...,1], columns=['t0_5000'])
df_t1000 = pd.DataFrame(data=c_max*ms_c[...,1001], columns=['t0_5000'])
df_t1660 = pd.DataFrame(data=c_max*ms_c[...,1661], columns=['t0_5000'])
df_t2000 = pd.DataFrame(data=c_max*ms_c[...,-1], columns=['t0_5000'])
df_r0 = pd.DataFrame(data=c_max*ms_c[0,:1001], columns=['t0_5000'])
df_r1 = pd.DataFrame(data=c_max*ms_c[-1,:1001], columns=['t0_5000'])
c_t1_p4 = pd.concat([c_t1_p4,df_t1], axis=1)
c_t1000_p4 = pd.concat([c_t1000_p4,df_t1000], axis=1)
c_t1660_p4 = pd.concat([c_t1660_p4,df_t1660], axis=1)
c_t2000_p4 = pd.concat([c_t2000_p4,df_t2000], axis=1)
c_r0_p4 = pd.concat([c_r0_p4,df_r0], axis=1)
c_r1_p4 = pd.concat([c_r1_p4,df_r1], axis=1)

# Save to excel
with pd.ExcelWriter('PINN Models/PINN_boundary_forcing.xlsx', engine='xlsxwriter') as writer:
    c_t1_p4.to_excel(writer, sheet_name='t1')
    c_t1000_p4.to_excel(writer, sheet_name='t1000')
    c_t1660_p4.to_excel(writer, sheet_name='t1660')
    c_t2000_p4.to_excel(writer, sheet_name='t2000')
    c_r0_p4.to_excel(writer, sheet_name='r0')
    c_r1_p4.to_excel(writer, sheet_name='r1')