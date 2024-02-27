# -*- coding: utf-8 -*-
"""
Created on Thu Feb  8 14:36:42 2024

@author: Waylon
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import math
# from torch.autograd import Variable
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter


# eq: r^2 dc/dt = d(Ds*r^2 * dc/dr)/dr
#  dc/dr at r=0 = 0
# dc/dr at r=R = -j/Ds

# inputs = r and t (j and Ds??) or is j and Ds dependent on r and t from an eq
# varying r=r0 bc too? like In/F
# outputs = 1 (concentration c)

def meshgrid_plot(r, t):
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ms_r, ms_t = np.meshgrid(r, t)
    r = np.ravel(ms_r).reshape(-1,1)
    t = np.ravel(ms_t).reshape(-1,1)
    pt_r = torch.from_numpy(r).float().requires_grad_(True).to(device)
    pt_t = torch.from_numpy(t).float().requires_grad_(True).to(device)
    return pt_r, pt_t, ms_r, ms_t

def dydx(y,x):
    dy_dx = torch.autograd.grad(y, x, grad_outputs=torch.ones_like(y), create_graph=True)[0]
    return dy_dx

class Net(nn.Module):
    def __init__(
        self,
        input_dim, # number of inputs
        output_dim, # number of outputs
        lays=5, # number of hidden layers
        n_units=128, # number per layer dim
        epochs=10000,
        loss=nn.MSELoss(), # neural network loss
        lr=1e-3, # learning rate
        device="cpu"
    ) -> None:
        super(Net,self).__init__()

        self.epochs = epochs
        self.loss = loss
        self.lr = lr
        self.n_units = n_units
        self.device = device
        
        # activation function
        act_func = nn.Tanh
        # act_func = nn.ReLU
        
        # self.input_layer = nn.Sequential(nn.Linear(input_dim, n_units), act_func())
        # self.hidden_layers = nn.Sequential(*[nn.Sequential(*[nn.Linear(n_units, n_units), act_func()]) for i in range(lays)])
        # self.output_layer = nn.Linear(n_units, output_dim)
        
        
        # input layer
        in_layer = nn.Linear(input_dim, n_units)
        nn.init.xavier_normal_(in_layer.weight)
        nn.init.zeros_(in_layer.bias)
        self.input_layer = nn.Sequential(in_layer, act_func())
        
        # hidden layers
        modules = []
        for i in range(lays-1):
            mid_lay = nn.Linear(n_units,n_units)
            nn.init.xavier_normal_(mid_lay.weight)
            nn.init.zeros_(mid_lay.bias)
            modules.append(nn.Sequential(mid_lay, act_func()))
            
        self.hidden_layers = nn.Sequential(*modules)
        
        # output layer
        self.output_layer = nn.Linear(n_units, output_dim)
        nn.init.xavier_normal_(self.output_layer.weight)
        nn.init.zeros_(self.output_layer.bias)
        # print(self.output_layer.weight)
        # print(self.output_layer.bias)
    
    def forward(self, r,t):
        x = torch.cat([r,t],axis=1) # combined two arrays of 1 columns each to one array of 2 columns
        ins = self.input_layer(x)
        mids = self.hidden_layers(ins)
        outs = self.output_layer(mids)
        return outs
    
    def train_loop(self, N, rmax, tmax, cmax, c_ini, c_rmax,  theta_norm, D):
        # torch.manual_seed(123)
        optimiser = optim.Adam(self.parameters(), lr=self.lr) # + lvbgs
        self.train()
        losses = []
        # c_ini = 0
        c_rmin = 0
        # c_rmax = 1
        # initial conditions
        t_t0 = torch.zeros(N,1).requires_grad_(True).to(device)
        c_t0 = torch.full((N,1),float(c_ini)).requires_grad_(True).to(device)
        
        # r=0 BC
        r_r0 = torch.zeros(N,1).requires_grad_(True).to(device)
        c_r0 = torch.full((N,1),float(c_rmin)).requires_grad_(True).to(device)
        
        # r=1 BC
        # r_r1 = torch.full((N,1),float(rmax)).requires_grad_(True).to(device)
        r_r1 = torch.ones(N,1).requires_grad_(True).to(device)
        c_r1 = torch.full((N,1),float(c_rmax)).requires_grad_(True).to(device)
        
        # pde loss
        c_phy = torch.zeros(N,1).requires_grad_(True).to(device)
        
        # skew r rand towards end (more points to the surface)
        # Adam optimiser then use lbfgs
        
        # adaptive sampling method for more at t=0 - read
        
        for epoch in range(self.epochs):
            optimiser.zero_grad()
            
            r_t0 = torch.rand(N,1).requires_grad_(True).to(device)
            guess_t0 = self.forward(r_t0,t_t0)
            mse_t0 = self.loss(guess_t0,c_t0)
            
            t_r0 = torch.rand(N,1).requires_grad_(True).to(device)
            guess_r0 = self.forward(r_r0,t_r0)
            dcdx_r0 = dydx(guess_r0,r_r0)
            mse_r0 = self.loss(dcdx_r0,c_r0)
            
            t_r1 = torch.rand(N,1).requires_grad_(True).to(device)
            bc1 = self.rmax_cbounds(r_r1, t_r1, theta_norm)
            # dcdx_r1 = torch.autograd.grad(guess_r0, r_r0, grad_outputs=torch.ones_like(guess_r0), create_graph=True)[0]
            mse_r1 = self.loss(bc1,c_r1)
            # mse_r1 = self.loss(dcdx_r1,c_r1)
            
            r_phy = torch.rand(N,1).requires_grad_(True).to(device)
            t_phy = torch.rand(N,1).requires_grad_(True).to(device)
            pde = self.phy_loss(r_phy,t_phy,theta_norm,D,rmax,tmax)
            mse_phy = self.loss(pde,c_phy)
            
            # loss = mse_t0 + mse_r0 + mse_r1 + mse_phy
            loss = mse_t0 + mse_r0 + mse_r1 + mse_phy
            # can play around with weights for loss func
            loss.backward()
            # loss.backward(retain_graph=True) # idk why need retain_graph=True
            # ^ but it basically acts like backward()
            optimiser.step()
            losses.append(loss.item())
            
            with torch.autograd.no_grad():
            	 print(epoch,"Traning Loss:",loss.data, mse_t0.data, mse_r0.data, mse_r1.data, mse_phy.data)
                # print(epoch,"Traning Loss:",loss.data)
                
            if epoch % 5000 == 0:

                fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
                r=np.arange(0,1,0.02) # 100 points
                t=np.arange(0,1,0.02)
                net_r, net_t, ms_r, ms_t = meshgrid_plot(r,t)
                c = self.predict(net_r, net_t)
                ms_c = c.reshape(ms_r.shape)
                ax.plot_surface(tmax*ms_t,ms_r*rmax,ms_c*cmax, cmap='viridis', antialiased=False)
                plt.title(f"Training Step {epoch}")
                plt.xlabel("t")
                plt.ylabel("x")
                plt.show()
        
        return losses
            
    def predict(self, r, t):
        self.eval()
        c = self.forward(r, t)
        return c.detach().cpu().numpy()
    
    # PDE as a loss function f
    def phy_loss(self,r,t,theta_norm,D,r0,tmax):
        c = self.forward(r,t) # concentration c is given by the network
        dcdr = dydx(c,r)
        d2cdr2 = dydx(dcdr,r)
        dcdt = dydx(c,t)
        pde = r*dcdt*(r0**2)/(D*tmax) - (1+theta_norm*c)*(r*d2cdr2 + dcdr*2) - r*theta_norm*dcdr**2
        return pde
# can also minimise d/error to solution
    
    def rmax_cbounds(self,r,t,theta_norm):
        c = self.forward(r,t)
        dcdr = dydx(c,r)
        bc1_val = -(1+theta_norm*c)*dcdr
        return bc1_val



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

# Normalisation
tmax = 1000 # norm t=1
tmax_norm = 1
dt = 1
dr = dt/tmax
I = In*Rs/(D*c_max*F)
theta_norm = theta*c_max
# t_norm_max = tmax*D/(Rs**2)
c_ini_norm = c_0/c_max
r_norm_max = 1
r_norm_min = 0
dcdt_norm = Rs**2 / (tmax*D)
bc0 = 0
bc1 = I
N = 100
lr = 1e-3
epochs = 20000
mse_loss_func = nn.MSELoss() # Mean squared error
# theta_norm=0
## model
input_dim = 2
output_dim = 1
net = Net(input_dim,output_dim,epochs=epochs,lr=lr,loss=mse_loss_func).to(device)
print(net)

# %% run sim
losses = net.train_loop(N,Rs,tmax,c_max,c_ini_norm,I,theta_norm,D)
# visualise losses v epochs

# lambda1 = 1e-2
# lambda2 = 1e-2

print("PINN's state_dict:")
for param_tensor in net.state_dict():
    print(param_tensor, "\t", net.state_dict()[param_tensor].size())

# # Print optimiser's state_dict
# print("Optimiser's state_dict:")
# for var_name in optimiser.state_dict():
#     print(var_name, "\t", optimiser.state_dict()[var_name])

torch.save(net.state_dict(),"PINN_statedict.pt")

net.load_state_dict(torch.load("PINN_statedict.pt"))
net.eval()
# %% final visualisation
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
r=np.arange(0,1,1e-2) # 100 points in r
t=np.arange(0,1,1e-2) # 100 points in t
net_r, net_t, ms_r, ms_t = meshgrid_plot(r,t)
c = net.predict(net_r, net_t)
ms_c = c.reshape(ms_r.shape)
ax.plot_surface(tmax*ms_t,ms_r*Rs,ms_c*c_max, cmap='viridis', antialiased=False)
plt.title(f"Training Step {epochs-1}")
plt.xlabel("t")
plt.ylabel("r")
plt.show()

# CVM solution files from other python code
CVM_r = np.loadtxt('CVM_r.csv', delimiter=',')
CVM_t = np.loadtxt('CVM_t.csv', delimiter=',')
CVM_c = np.loadtxt('CVM_c.csv', delimiter=',')

# CVM_r = np.loadtxt('CVM_r_theta0.csv', delimiter=',')
# CVM_t = np.loadtxt('CVM_t_theta0.csv', delimiter=',')
# CVM_c = np.loadtxt('CVM_c_theta0.csv', delimiter=',')

plt.figure(0)
plt.plot(Rs*ms_r[1],c_max*ms_c[1], label = "neural_network")
plt.plot(CVM_r,CVM_c[...,1], label = 'CVM_soln')
plt.legend()
plt.xlabel("r")
plt.ylabel("c at t0")

plt.figure(1)
plt.plot(Rs*ms_r[-1],c_max*ms_c[-1], label = "neural_network")
plt.plot(CVM_r,CVM_c[...,-1], label = 'CVM_soln')
plt.legend()
plt.xlabel("r")
plt.ylabel("c at tmax")

plt.figure(2)
plt.plot(tmax*ms_t[...,1],c_max*ms_c[...,1], label = "neural_network")
plt.plot(CVM_t,CVM_c[1], label = 'CVM_soln')
plt.legend()
plt.xlabel("t")
plt.ylabel("c at rmin")

plt.figure(3)
plt.plot(tmax*ms_t[...,-1],c_max*ms_c[...,-1], label = "neural_network")
plt.plot(CVM_t,CVM_c[-1], label = 'CVM_soln')
plt.legend()
plt.xlabel("t")
plt.ylabel("c at rmax")

plt.figure(4)
plt.plot(np.arange(100,epochs,1),losses[100:])
plt.xlabel("epochs")
plt.ylabel("loss")

plt.show()

# %% animation

# import matplotlib.animation as ani
# fig = plt.figure(4)
# plt.xlabel('r (m)')
# plt.ylabel('c (mol/m3)')
# plt.ylim(0,15000) # max = cmax
# plt.xlim(0,5e-6) # max = Rs
# txt_title = fig.set_title('')
# line1, = fig.plot([], [])     
# # ax.plot returns a list of 2D line objects
# line2, = fig.plot([], [])
# fig.legend(['neural_network','CVM_soln'])
# # plt.plot(Rs*ms_r[0],c_max*ms_c[0], label = "neural_network")
# def animate(i):
#     # fig.plot(Rs*ms_r[i],c_max*ms_c[i], label = "neural_network") 
#     # fig.plot(CVM_r,CVM_c[...,i], label = 'CVM_soln')
#     x1 = Rs*ms_r[i]
#     x2 = CVM_r
#     y1 = c_max*ms_c[i]
#     y2 = CVM_c[...,i]
#     line1.set_data(x1, y1)
#     line2.set_data(x2, y2)
#     txt_title.set_text('Frame = {0:4d}'.format(i))
#     return (line1,line2)


# animator = ani.FuncAnimation(fig, animate, frames=len(ms_c[0]))

# plt.show()

# fig, ax = plt.subplots()

# plt.xlabel('r (m)')
# plt.ylabel('c (mol/m3)')
# plt.ylim(-200,5000) # max = cmax
# plt.xlim(0,5e-6) # max = Rs

# # x = np.arange(0, 2*np.pi, 0.01)
# line1, = ax.plot(Rs*ms_r[0],c_max*ms_c[0], label='neural_network')
# line2, = ax.plot(CVM_r,CVM_c[...,0], label='CVM_soln')

# def animate(i):
#     line1.set_ydata(c_max*ms_c[i])  # update the data.
#     line2.set_ydata(CVM_c[...,i*20])
#     return line1, line2,


# vid = ani.FuncAnimation(fig, animate, frames=10)

# plt.show