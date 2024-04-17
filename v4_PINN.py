# -*- coding: utf-8 -*-
"""
Created on Thu Feb  8 14:36:42 2024

@author: Waylon
"""

# this code includes PINN + adaptie sampling + distance functions for ic and bc

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, ConcatDataset, DataLoader
import math
# from torch.autograd import Variable
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

from timeit import default_timer as timer
start = timer()
# eq: r^2 dc/dt = d(Ds*r^2 * dc/dr)/dr
#  dc/dr at r=0 = 0
# dc/dr at r=R = -j/Ds

# inputs = r and t (j and Ds??) or is j and Ds dependent on r and t from an eq
# varying r=r0 bc too? like In/F
# outputs = 1 (concentration c)

def create_data(N, c_t0, c_r0, c_r1, grid=False,seed=None):
    if seed != None:
        torch.manual_seed(seed)
    t0_t = torch.zeros(N,1)
    t0_r = torch.rand(N,1)
    t0_c = torch.full((N,1),float(c_t0))
    r0_t = torch.rand(N,1)
    r0_r = torch.zeros(N,1)
    r0_c = torch.full((N,1),float(c_r0))
    r1_t = torch.rand(N,1)
    r1_r = torch.ones(N,1)
    r1_c = torch.full((N,1),float(I))
    pde_t = torch.rand(N,1)
    pde_r = torch.rand(N,1)
    if grid == True:
        side = int(np.sqrt(N))
        if side*side == N:
            t = np.linspace(0,1,side)
            r = np.linspace(0,1,side)
            pde_r, pde_t, ms_r, ms_t = meshgrid_plot(r,t)
        else:
            print("N is not a perfect square, using random PDE data")
    pde_c = torch.zeros(N,1)
    
    t0_data = TensorDataset(t0_t, t0_r, t0_c)
    r0_data = TensorDataset(r0_t, r0_r, r0_c)
    r1_data = TensorDataset(r1_t, r1_r, r1_c)
    pde_data = TensorDataset(pde_t, pde_r, pde_c)
    
    return t0_data, r0_data, r1_data, pde_data

def meshgrid_plot(r, t):
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ms_r, ms_t = np.meshgrid(r, t, indexing='ij')
    r = np.ravel(ms_r).reshape(-1,1)
    t = np.ravel(ms_t).reshape(-1,1)
    pt_r = torch.from_numpy(r).float().requires_grad_(True).to(device)
    pt_t = torch.from_numpy(t).float().requires_grad_(True).to(device)
    return pt_r, pt_t, ms_r, ms_t

def dydx(y,x):
    dy_dx = torch.autograd.grad(y, x, grad_outputs=torch.ones_like(y), create_graph=True)[0]
    return dy_dx

def CVM_data(r,t,c):
    net_r, net_t, grid_r, grid_t = meshgrid_plot(r,t)
    val_c = np.ravel(c).reshape(-1,1)
    net_c = torch.from_numpy(val_c).float().requires_grad_(True).to(device)
    return net_r, net_t, net_c

# class Sine(torch.nn.Module):
# class Sine(torch.nn.Module):
    # def __init__(self, w0=1.):
    #     super(self).__init__()
    #     # return
    # def forward(self, x):
    #     return torch.sin(self.w0 * x)
    # def forward(self, input: Tensor) -> Tensor:
    #     return torch.sin(input)
class Sin(torch.nn.Module):
  # def __init__(self):
  #   pass
  def forward(self, x):
    return torch.sin(x)

# PINN
class PINN(nn.Module):
    def __init__(
        self,
        input_dim, # number of inputs
        output_dim, # number of outputs
        lays=3, # number of hidden layers
        n_units=256, # number per layer dim
        loss=nn.MSELoss(), # neural network loss
        act_func = nn.Tanh,
        device="cpu",
        seed=None
    ) -> None:
        super(PINN,self).__init__()
        
        if seed != None:
            torch.manual_seed(seed)
            
        self.loss_func = loss
        self.n_units = n_units
        self.device = device
          
        # activation function
        # act_func = nn.Tanh
        # act_func = Sin
        
        # input layer
        in_layer = nn.Linear(input_dim, n_units)
        nn.init.xavier_normal_(in_layer.weight)
        nn.init.zeros_(in_layer.bias)
        # in_layer = act_func(in_layer)
        self.input_layer = nn.Sequential(in_layer, act_func())
        # self.input_layer = nn.Sequential(in_layer,act_func)
        # self.input_layer = nn.Sequential(in_layer)
        
        # hidden layers
        modules = []
        for i in range(lays-1):
            mid_lay = nn.Linear(n_units,n_units)
            nn.init.xavier_normal_(mid_lay.weight)
            nn.init.zeros_(mid_lay.bias)
            modules.append(nn.Sequential(mid_lay, act_func()))
            # modules.append(nn.Sequential(mid_lay, act_func))
            
        self.hidden_layers = nn.Sequential(*modules)
        
        # output layer
        self.output_layer = nn.Linear(n_units, output_dim)
        # nn.init.xavier_normal_(self.output_layer.weight)
        # nn.init.zeros_(self.output_layer.bias)
        nn.init.xavier_normal_(self.output_layer.weight)
        nn.init.zeros_(self.output_layer.bias)
        # print(self.output_layer.weight)
        # print(self.output_layer.bias)
        
        # # adam optimiser
        self.lr = 1e-3 # adam learning rate
        self.adam_optim = optim.Adam(self.parameters(), lr=self.lr)
        
        # lbfgs optimiser
        self.lbfgs_optim = optim.LBFGS(
            self.parameters(),
            max_iter=100,
            history_size=100,
            tolerance_grad=1.0 * np.finfo(float).eps, 
            tolerance_change=1.0 * np.finfo(float).eps,
            line_search_fn='strong_wolfe')
        
    def forward(self, r,t):
        x = torch.cat([r,t],axis=1) # combined two arrays of 1 columns each to one array of 2 columns
        ins = self.input_layer(x)
        mids = self.hidden_layers(ins)
        outs = self.output_layer(mids)
        return outs
    
    def guess_c(self, r, t):
        c = self.forward(r,t)
        # c = t*c
        dcdr = dydx(c,r)
        # print("one",dcdr)
        # c = c + r*dcdr + c*r**2
        # dctdr = dydx(c*t,r)
        # c = t*c + t*r*(r*c + dctdr)
        c = (c+r*dcdr)*t/(t+r**2) + c*t*r**2
        # c_guess = torch.pow(t,0.5)*c
        return c
    
    def vals(self, tmax=1, rmax=1, cmax=1, D=1, theta_norm=0):
        self.tmax = tmax
        self.rmax = rmax
        self.cmax = cmax
        self.D = D
        self.theta_norm = theta_norm
    
    def load_data(self, batch, t0_data, r0_data, r1_data, phy_data, val_data):
        # if run with dataloader
        self.batch = batch
        self.phy_data = phy_data
        self.phy_N = len(phy_data)
        self.val_data = val_data
        self.val_loader = DataLoader(dataset=val_data, batch_size=batch, shuffle=True)
        self.t0_loader = DataLoader(dataset=t0_data, batch_size=batch, shuffle=True)
        self.r0_loader = DataLoader(dataset=r0_data, batch_size=batch, shuffle=True)
        self.r1_loader = DataLoader(dataset=r1_data, batch_size=batch, shuffle=True)
        self.phy_loader = DataLoader(dataset=phy_data, batch_size=batch, shuffle=True)
    
    def change_data(self,seed=None):
        if seed != None:
            torch.manual_seed(seed)
            
        self.pde_visual()
        
        N = self.phy_N
        new_t = torch.rand(N,1)
        # print(new_t)
        new_r = torch.rand(N,1)
        new_c = torch.zeros(N,1)
        new_dataset = TensorDataset(new_t,new_r,new_c)
        old_dataset = self.phy_data
        new_data = ConcatDataset([old_dataset,new_dataset])
        dataload = DataLoader(dataset=new_data,batch_size=len(new_dataset)+self.phy_N)
        
        t, r, c = next(iter(dataload))
        t = t.requires_grad_(True).to(self.device)
        r = r.requires_grad_(True).to(self.device)
        c = c.to(self.device)
        # guess_c = self.forward(r,t)
        # guess_c = t*guess_c
        # guess_c = torch.sqrt(t)*guess_c
        guess_c = self.predict(r,t)
        pde = self.phy_loss(r,t,guess_c)
        k1 = 2
        k2 = 0
        err_eq = torch.pow(pde, k1) / torch.pow(pde, k1).mean() + k2 # mean square error
        # err_eq_norm = (err_eq / sum(err_eq))
        
        # replace original values with new rand values
        err_norm = err_eq / err_eq.sum()
        err_norm = err_norm.transpose(0,1)
        IDs = torch.multinomial(err_norm,self.phy_N,replacement=False)
        t_data = torch.squeeze(t[IDs],0)
        r_data = torch.squeeze(r[IDs],0)
        c_data = torch.squeeze(c[IDs],0)
        new_dataset = TensorDataset(t_data.detach(),r_data.detach(),c_data.detach())
        self.phy_data = new_dataset
        new_dataloader = DataLoader(dataset=new_dataset, batch_size=self.batch, shuffle=True)
        self.phy_loader = new_dataloader
        
        plt.scatter(t_data.detach(), r_data.detach())
        plt.title("PDE new data")
        plt.xlabel("t")
        plt.ylabel("r")
        plt.xlim(-0.05, 1.05)
        plt.ylim(-0.05, 1.05)
        plt.show()
        
    def pinn_loss(self):
        
        # initial condition
        t_t0, r_t0, c_t0 = next(iter(self.t0_loader))
        r_t0 = r_t0.requires_grad_(True).to(self.device)
        t_t0 = t_t0.requires_grad_(True).to(self.device)
        c_t0 = c_t0.to(self.device)
        # guess_t0 = self.forward(r_t0,t_t0)
        guess_t0 = self.guess_c(r_t0,t_t0)
        mse_t0 = self.loss_func(guess_t0,c_t0)
        # dydx_t0 = dydx(mse_t0,t_t0)
        
        # r=0 condition
        t_r0, r_r0, c_r0 = next(iter(self.r0_loader))
        r_r0 = r_r0.requires_grad_(True).to(self.device)
        t_r0 = t_r0.requires_grad_(True).to(self.device)
        c_r0 = c_r0.to(self.device)
        # guess_r0 = self.forward(r_r0,t_r0)
        guess_r0 = self.guess_c(r_r0,t_r0)
        dcdx_r0 = dydx(guess_r0,r_r0)
        mse_r0 = self.loss_func(dcdx_r0,c_r0)
        
        # r=1 condition
        t_r1, r_r1, c_r1 = next(iter(self.r1_loader))
        r_r1 = r_r1.requires_grad_(True).to(self.device)
        t_r1 = t_r1.requires_grad_(True).to(self.device)
        c_r1 = c_r1.to(self.device)
        # guess_r1 = self.forward(r_r1,t_r1)
        guess_r1 = self.guess_c(r_r1,t_r1)
        bc1 = self.rmax_cbounds(r_r1, t_r1, guess_r1)
        mse_r1 = self.loss_func(bc1,c_r1)
        
            
        t_phy, r_phy, c_phy = next(iter(self.phy_loader))
        r_phy = r_phy.requires_grad_(True).to(self.device)
        t_phy = t_phy.requires_grad_(True).to(self.device)
        c_phy = c_phy.to(self.device)
        # guess_pde = self.forward(r_phy,t_phy)
        guess_pde = self.guess_c(r_phy,t_phy)
        pde = self.phy_loss(r_phy,t_phy,guess_pde)
        mse_phy = self.loss_func(pde,c_phy) # only gives one loss value
        # also calc d(pde)/dc i guess
        # dpdedc = dydx(pde,guess_pde)
        # mse_dphy = self.loss_func(dpdedc,c_phy)
        
        # train_loss = mse_t0 + mse_r0 + mse_r1 + mse_phy + mse_dphy
        # train_loss = mse_t0 + mse_r0 + mse_r1 + mse_phy
        train_loss = mse_r1 + mse_phy
        self.train_loss = train_loss
        self.t0_loss = mse_t0
        self.r0_loss = mse_r0
        self.r1_loss = mse_r1
        self.pde_loss = mse_phy
        # self.dpde_loss = mse_dphy
        return train_loss
    
    # PDE as a loss function f
    def phy_loss(self,r,t,c):
        # c = self.forward(r,t) # concentration c is given by the network
        dcdr = dydx(c,r)
        d2cdr2 = dydx(dcdr,r)
        dcdt = dydx(c,t)
        pde = r*dcdt*(self.rmax**2)/(self.D*self.tmax) - (1+self.theta_norm*c)*(r*d2cdr2 + dcdr*2) - r*self.theta_norm*dcdr**2
        # print(pde)
        return pde
        # can also minimise d/error to solution
    
    def rmax_cbounds(self,r,t,c):
        # c = self.forward(r,t)
        dcdr = dydx(c,r)
        bc1_val = -(1+self.theta_norm*c)*dcdr
        return bc1_val
    
    def val_loss(self):
        # self.eval()
        r, t, c = next(iter(self.val_loader))
        # for r, t, c in self.val_loader:
        guess_c = self.predict(r,t)
        val_losses = self.loss_func(guess_c, c)
        return val_losses          
   
    def closure(self):
        self.lbfgs_optim.zero_grad()
        loss = self.pinn_loss() # change 1 and 2 to a var to change
        loss.backward()
        return loss
    
    def train_loop(self, adam_epochs=10000, adam_lr=1e-3, lbfgs_epochs=100, runs=1, data_change=False, plots=1,seed=None):
        adam_loss = []
        adam_t0 = []
        adam_r0 = []
        adam_r1 = []
        adam_pde = []
        adam_val = []
        lbfgs_loss = []
        lbfgs_t0 = []
        lbfgs_r0 = []
        lbfgs_r1 = []
        lbfgs_pde = []
        lbfgs_val = []
        
        for g in self.adam_optim.param_groups:
            g['lr'] = lr
        
        for i in range(runs):
            if data_change == True:
                self.change_data(seed=seed)
                
            for epoch in range(adam_epochs):
                self.train()
                for j, data in enumerate(self.phy_loader):
                    self.adam_optim.zero_grad()
                
                    loss = self.pinn_loss()
                    # can play around with weights for loss func
                    loss.backward()
                    # loss.backward(retain_graph=True) # idk why need retain_graph=True
                    # ^ but it basically acts like backward()
                    self.adam_optim.step()
                # losses.append(loss.item())
                    
                    with torch.autograd.no_grad():
                        print("Run",i,"Epoch:",epoch,"Batch:",j+1,"\nAdam Loss:",loss.data, self.t0_loss.data, self.r0_loss.data, self.r1_loss.data, self.pde_loss.data)
                    # print(epoch,"Traning Loss:",loss.data)
                    
                with torch.autograd.no_grad():
                    adam_loss.append(self.train_loss.data)
                    adam_t0.append(self.t0_loss.data)
                    adam_r0.append(self.r0_loss.data)
                    adam_r1.append(self.r1_loss.data)
                    adam_pde.append(self.pde_loss.data)
                    
                val_losses = self.val_loss()
                adam_val.append(val_losses.data)
                
                if epoch % (adam_epochs/plots) == 0:
                    self.visualisation(epoch)
                    # plt.title(f"Adam Training Step {epoch}")
                
            for epoch in range(lbfgs_epochs):
                self.train()
                for j, data in enumerate(self.phy_loader):
                    self.lbfgs_optim.step(self.closure)
                    
                    with torch.autograd.no_grad():
                        # print(epoch,"Training Loss:",self.train_loss.data)
                        print("Run",i,"Epoch:",epoch,"Batch:",j+1,"\nLBFGS Loss:",self.train_loss.data, self.t0_loss.data, self.r0_loss.data, self.r1_loss.data, self.pde_loss.data)
                
                with torch.autograd.no_grad():    
                    lbfgs_loss.append(self.train_loss.data)
                    lbfgs_t0.append(self.t0_loss.data)
                    lbfgs_r0.append(self.r0_loss.data)
                    lbfgs_r1.append(self.r1_loss.data)
                    lbfgs_pde.append(self.pde_loss.data)
                
                
                val_error = self.val_loss()
                lbfgs_val.append(val_error.data)
                
                if epoch % lbfgs_epochs == 0:
                    self.visualisation(epoch)
                    # plt.title(f"LBFGS Training Step {epoch}")

        # losses: loss, t0, r0, r1, pde, val
        adam_losses = [adam_loss,adam_t0,adam_r0,adam_r1,adam_pde,adam_val]
        lbfgs_losses = [lbfgs_loss,lbfgs_t0,lbfgs_r0,lbfgs_r1,lbfgs_pde,lbfgs_val]
        return adam_losses, lbfgs_losses
            
    def predict(self, r, t):
        self.eval()
        # c = self.forward(r, t)
        # return c.detach().cpu().numpy()
        c = self.guess_c(r,t)
        return c      
    
    def visualisation(self, epoch):
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        r=np.linspace(0,1,101) # 100 points
        t=np.linspace(0,1,1001) # 1000 points
        net_r, net_t, ms_r, ms_t = meshgrid_plot(r,t)
        c = self.predict(net_r, net_t)
        c = c.detach().cpu().numpy() 
        ms_c = c.reshape(ms_r.shape)
        ax.plot_surface(ms_r*self.rmax,self.tmax*ms_t,ms_c*self.cmax, cmap='viridis', antialiased=False)
        plt.title(f"Training Step {epoch}")
        plt.xlabel("r")
        plt.ylabel("t")
        plt.show()
        
        plt.plot(Rs*ms_r[...,1],c_max*ms_c[...,1], label = "neural_network")
        # plt.plot(Rs*CVM_r,c_max*CVM_c[...,1], label = 'CVM_soln')
        plt.legend()
        plt.xlabel("r")
        plt.ylabel("c at t0")
        plt.show()
        
    def pde_visual(self):
        r=np.linspace(0,1,101)[1:] # 100 points
        t=np.linspace(0,1,101)[1:] # 100 points
        # r = np.rand(100,1)
        # t = np.rand(100,1)
        net_r, net_t, ms_r, ms_t = meshgrid_plot(r,t)
        # net_c = self.forward(net_r,net_t)
        # net_c = net_t*net_c
        net_c = self.guess_c(net_r,net_t)
        c_pde = self.phy_loss(net_r, net_t, net_c)
        vals = torch.zeros(len(c_pde),1)
        # pde_err = torch.pow(c_pde,2) / torch.pow(c_pde,2).mean()
        pde_err = torch.sqrt(torch.pow(c_pde,2))
        pde_loss = self.loss_func(c_pde,vals)
        pde_err = pde_err.detach().cpu().numpy()
        grid_pde_err = pde_err.reshape(ms_r.shape)
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        ax.plot_surface(ms_r,ms_t,grid_pde_err, cmap='viridis', antialiased=False)
        plt.title("pde loss")
        plt.xlabel("r")
        plt.ylabel("t")
        plt.show()
        print("PDE Loss:", pde_loss.detach().cpu().numpy())
        return(pde_loss.detach().cpu().numpy())
    
    def val_err(self,r,t):
        # val_data = self.val_data
        dataload = DataLoader(dataset=val_data,batch_size=len(val_data))
        t, r, c = next(iter(dataload))
        guess_c = self.predict(r, t)
        err_vals = torch.sqrt(torch.pow((guess_c - c),2))
        return err_vals.detach().cpu().numpy()
# %% code starts here

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
# current density -1 to -3 as new input?
theta = (omega/R/T)*(2*omega*E/9/(1-nu)) # [m3/mol]

# Normalisation
CVM_tmax = 1000 # norm t=1
# tmax = CVM_tmax * Rs**2 / D
tmax = CVM_tmax
dt = 1
dr = dt/tmax
I = In*Rs/(D*c_max*F)
theta_norm = theta*c_max
c_ini_norm = c_0/c_max
# dcdt_norm = Rs**2 / (tmax*D)
# theta_norm=0


# model stuff
lr = 1e-3
adam_epochs = 5000
lbfgs_epochs = 100
change=True
mse_loss_func = nn.MSELoss() # Mean squared error function
input_dim = 2
output_dim = 1
layers = 5
per_layer = 256
actfunc = nn.Tanh
# actfunc = Sin
randomseed = 12121

# data loading
N = 5000 # training data
batch = 1024 # batchsize


# load validation files
CVM_r_data = np.loadtxt('CVM_r.csv', delimiter=',')
CVM_t_data = np.loadtxt('CVM_t.csv', delimiter=',')
CVM_c_data = np.loadtxt('CVM_c.csv', delimiter=',')

CVM_r = CVM_r_data/Rs
CVM_t = CVM_t_data[:1001]/tmax
CVM_c = CVM_c_data[:,:1001]/c_max

val_r, val_t, val_c = CVM_data(CVM_r,CVM_t,CVM_c)
val_data = TensorDataset(val_t, val_r, val_c)

net = PINN(input_dim, output_dim, lays=layers, n_units=per_layer, act_func=actfunc, loss=mse_loss_func, device=device, seed=randomseed).to(device)
# print(net)

net.vals(tmax=tmax,rmax=Rs,cmax=c_max,D=D,theta_norm=theta_norm)
t0_data,r0_data,r1_data,phy_data = create_data(N,0,0,I,seed=randomseed)
net.load_data(batch,t0_data,r0_data,r1_data,phy_data,val_data)

phyload = DataLoader(dataset=phy_data,batch_size=N)

phy_t, phy_r, phy_c = next(iter(phyload))
plt.scatter(phy_t, phy_r)
plt.title("PDE data")
plt.xlabel("t")
plt.ylabel("r")
plt.xlim(-0.05, 1.05)
plt.ylim(-0.05, 1.05)
plt.show()


# %%
adam_losses, lbfgs_losses = net.train_loop(adam_epochs=adam_epochs, adam_lr=lr, lbfgs_epochs=lbfgs_epochs, plots=5)

#%%
# adaptive_epochs = 2000
# runs = 4
# adam_adaptive, lbfgs_adaptive = net.train_loop(adam_epochs=adaptive_epochs, adam_lr=lr, lbfgs_epochs=50, runs=runs, data_change=True)

print("PINN")
print(net)

print("PINN's state_dict:")
for param_tensor in net.state_dict():
    print(param_tensor, "\t", net.state_dict()[param_tensor].size())

# Print optimiser's state_dict
# print("Optimiser's state_dict:")
# for var_name in optimiser.state_dict():
#     print(var_name, "\t", optimiser.state_dict()[var_name])

torch.save(net.state_dict(),"PINN_state_5x256_tanh_N5000_b1024_5000epoch_t0r0force.pt")

end = timer()
print("Time taken:", end-start,"s")

# net.load_state_dict(torch.load("PINN_state_3x100_tanh_N1000_b128_5000epoch.pt"))
# net.eval()

# r = torch.zeros(10,1).requires_grad_(True)
# t = torch.rand(10,1).requires_grad_(True)
# c = torch.rand(10,1).requires_grad_(True)
# dcdr = dydx(t,r)
# print(dcdr)

# %% final visualisation

net.visualisation(adam_epochs+lbfgs_epochs)

net.pde_visual()

net_r, net_t, ms_r, ms_t = meshgrid_plot(CVM_r,CVM_t)
# err_val = net.val_visual(net_r,net_t)
# err_grid = err_val.

# fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
# r=np.linspace(0,1,100) # 100 points in r
# t=np.linspace(0,1,1001) # 1001 points in t

c = net.predict(net_r, net_t)
# fac = net.factor(net_r,net_t)
# c = fac*c
c_pde = net.phy_loss(net_r, net_t, c)
pde_err = torch.pow(c_pde,2) / torch.pow(c_pde,2).mean()
err = torch.pow((val_c - c),2) / torch.pow((val_c*c_max - c),2).mean()
err_vals = torch.pow((val_c - c),2)
val_err = net.loss_func(val_c,c)
c = c.detach().cpu().numpy()
ms_c = c.reshape(ms_r.shape)
print("Validation Loss:", val_err.detach().cpu().numpy())
# ax.plot_surface(ms_r*Rs,tmax*ms_t,ms_c*c_max, cmap='viridis', antialiased=False)
# plt.title(f"Training Step {adam_epochs+lbfgs_epochs}")
# plt.xlabel("r")
# plt.ylabel("t")
# plt.show()

err_np = err.detach().cpu().numpy()
err_vals = err_vals.detach().cpu().numpy()
err_grid = err_np.reshape(ms_r.shape)
err_valgrid = err_vals.reshape(ms_r.shape)
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
ax.plot_surface(ms_r,ms_t,err_valgrid, cmap='viridis', antialiased=False)
plt.title("val loss")
plt.xlabel("r")
plt.ylabel("t")
plt.show()

# pde_err = torch.pow(c_pde,2)
# pde_err = torch.sqrt(pde_err)
# c_pde = c_pde.detach().cpu().numpy()
# pde_err = pde_err.detach().cpu().numpy()
# c_gridpde = c_pde.reshape(ms_r.shape)
# grid_pde_err = pde_err.reshape(ms_r.shape)
# fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
# ax.plot_surface(ms_r,ms_t,grid_pde_err, cmap='viridis', antialiased=False)
# plt.title("pde loss")
# plt.xlabel("r")
# plt.ylabel("t")
# plt.show()
# %% line graphs
plt.figure(0)
plt.plot(Rs*ms_r[...,1],c_max*ms_c[...,1], label = "neural_network")
plt.plot(Rs*CVM_r,c_max*CVM_c[...,1], label = 'CVM_soln')
plt.legend()
plt.xlabel("r")
plt.ylabel("c at t0")
# plt.ylim(-300,1000)
L_1 = mse = ((ms_c[...,1] - CVM_c[...,1]/c_max)**2).mean()
print("Loss at 1s:",L_1)

plt.figure(1)
plt.plot(Rs*ms_r[...,501],c_max*ms_c[...,501], label = "neural_network")
plt.plot(Rs*CVM_r,c_max*CVM_c[...,501], label = 'CVM_soln')
plt.legend()
plt.xlabel("r")
plt.ylabel("c at t=500s")
L_500 = mse = ((ms_c[...,501] - CVM_c[...,501])**2).mean()
print("Loss at 500s:",L_500)
plt.figure(2)
plt.plot(Rs*ms_r[...,-1],c_max*ms_c[...,-1], label = "neural_network")
plt.plot(Rs*CVM_r,c_max*CVM_c[...,-1], label = 'CVM_soln')
plt.legend()
plt.xlabel("r")
plt.ylabel("c at tmax")
L_1000 = mse = ((ms_c[...,-1] - CVM_c[...,-1])**2).mean()
print("Loss at 1000s:",L_1000)

plt.figure(3)
plt.plot(tmax*ms_t[1],c_max*ms_c[1], label = "neural_network")
plt.plot(tmax*CVM_t,c_max*CVM_c[1], label = 'CVM_soln')
plt.legend()
plt.xlabel("t")
plt.ylabel("c at rmin")

plt.figure(4)
plt.plot(tmax*ms_t[50],c_max*ms_c[50], label = "neural_network")
plt.plot(tmax*CVM_t,c_max*CVM_c[50], label = 'CVM_soln')
plt.legend()
plt.xlabel("t")
plt.ylabel("c at r=mid")

plt.figure(5)
plt.plot(tmax*ms_t[-1],c_max*ms_c[-1], label = "neural_network")
plt.plot(tmax*CVM_t,c_max*CVM_c[-1], label = 'CVM_soln')
plt.legend()
plt.xlabel("t")
plt.ylabel("c at rmax")

# plt.figure(6)
# plt.plot(np.arange(0,adam_epochs,1),adam_losses[0])
# plt.xlabel("epochs")
# plt.ylabel("loss")
# # plt.ylim(0,0.1)

# plt.figure(7)
# plt.plot(np.arange(0,adam_epochs,1),adam_losses[5])
# plt.xlabel("epochs")
# plt.ylabel("validation loss")
# plt.ylim(0,0.02)

# plt.figure(8)
# plt.plot(np.arange(0,lbfgs_epochs,1),lbfgs_losses[0])
# plt.xlabel("epochs")
# plt.ylabel("loss")


# plt.figure(9)
# plt.plot(np.arange(0,lbfgs_epochs,1),lbfgs_losses[5])
# plt.xlabel("epochs")
# plt.ylabel("validation loss")
# plt.show()
# %%
plt.figure(10)
r = np.ravel(CVM_r_data/Rs).reshape(-1,1)
r = torch.from_numpy(r).float().requires_grad_(True).to(device)
t = torch.full((len(CVM_r_data),1),float(CVM_t_data[1661]/tmax)).requires_grad_(True).to(device)
c = net.predict(r,t)
c = c.detach().cpu().numpy()
# print(c)
plt.plot(CVM_r_data,c_max*c, label = "neural_network")
plt.plot(CVM_r_data,CVM_c_data[...,1661], label = 'CVM_soln')
plt.legend()
plt.xlabel("r")
plt.ylabel("c at t=1660")
L_1660 = mse = ((c - CVM_c_data[...,1661]/c_max)**2).mean()
print("Loss at 1660s:",L_1660)

plt.figure(11)
t = torch.full((len(CVM_r_data),1),float(CVM_t_data[-1]/tmax)).requires_grad_(True).to(device)
c = net.predict(r,t)
c = c.detach().cpu().numpy()
# print(c)
plt.plot(CVM_r_data,c_max*c, label = "neural_network")
plt.plot(CVM_r_data,CVM_c_data[...,-1], label = 'CVM_soln')
plt.legend()
plt.xlabel("r")
plt.ylabel("c at t=2000")
L_2000 = mse = ((c - CVM_c_data[...,-1]/c_max)**2).mean()
print("Loss at 2000s:",L_2000)

#%% subplots
# c outputs
fix, ax = plt.subplots(2,2,figsize=(8, 6))
ax[0,0].plot(Rs*ms_r[...,1],c_max*ms_c[...,1])
ax[0,0].plot(Rs*CVM_r,c_max*CVM_c[...,1])
ax[0,0].legend(['neural_network','CVM_soln'])
ax[0,0].set_xlabel("r")
ax[0,0].set_ylabel("c")
ax[0,0].set_title('At t=1s')

ax[0,1].plot(Rs*ms_r[...,-1],c_max*ms_c[...,-1])
ax[0,1].plot(Rs*CVM_r,c_max*CVM_c[...,-1])
ax[0,1].legend(['neural_network','CVM_soln'])
ax[0,1].set_xlabel("r")
ax[0,1].set_ylabel("c")
ax[0,1].set_title('At t=1000s')

ax[1,0].plot(tmax*ms_t[0],c_max*ms_c[0])
ax[1,0].plot(tmax*CVM_t,c_max*CVM_c[0])
ax[1,0].legend(['neural_network','CVM_soln'])
ax[1,0].set_xlabel("t")
ax[1,0].set_ylabel("c")
ax[1,0].set_title('At r=0')

ax[1,1].plot(tmax*ms_t[-1],c_max*ms_c[-1])
ax[1,1].plot(tmax*CVM_t,c_max*CVM_c[-1])
ax[1,1].legend(['neural_network','CVM_soln'])
ax[1,1].set_xlabel("t")
ax[1,1].set_ylabel("c")
ax[1,1].set_title('At r=5e-6')

plt.tight_layout()
plt.show()

# training and validation loss
fix, ax = plt.subplots(2,2,figsize=(8, 6))
ax[0,0].plot(np.arange(0,adam_epochs,1),adam_losses[0])
ax[0,0].set_xlabel("Epochs")
ax[0,0].set_ylabel("Loss")
ax[0,0].set_yscale("log")
ax[0,0].set_title('Adam Training Loss')

ax[0,1].plot(np.arange(0,adam_epochs,1),adam_losses[5])
ax[0,1].set_xlabel("Epochs")
ax[0,1].set_ylabel("Loss")
ax[0,1].set_yscale("log")
ax[0,1].set_title('Adam Validation Loss')

ax[1,0].plot(np.arange(0,lbfgs_epochs,1),lbfgs_losses[0])
ax[1,0].set_xlabel("Epochs")
ax[1,0].set_ylabel("Loss")
ax[1,0].set_yscale("log")
ax[1,0].set_title('LBFGS Training Loss')

ax[1,1].plot(np.arange(0,lbfgs_epochs,1),lbfgs_losses[5])
ax[1,1].set_xlabel("Epochs")
ax[1,1].set_ylabel("Loss")
ax[1,1].set_yscale("log")
ax[1,1].set_title('LBFGS Validation Loss')

plt.tight_layout()
plt.show()

# adam
fix, ax = plt.subplots(2,2,figsize=(8, 6))
ax[0,0].plot(np.arange(0,adam_epochs,1),adam_losses[1])
ax[0,0].set_xlabel("Epochs")
ax[0,0].set_ylabel("Loss")
ax[0,0].set_yscale("log")
ax[0,0].set_title('Adam t0 Loss')

ax[0,1].plot(np.arange(0,adam_epochs,1),adam_losses[2])
ax[0,1].set_xlabel("Epochs")
ax[0,1].set_ylabel("Loss")
ax[0,1].set_yscale("log")
ax[0,1].set_title('Adam r0 Loss')

ax[1,0].plot(np.arange(0,adam_epochs,1),adam_losses[3])
ax[1,0].set_xlabel("Epochs")
ax[1,0].set_ylabel("Loss")
ax[1,0].set_yscale("log")
ax[1,0].set_title('Adam r1 Loss')

ax[1,1].plot(np.arange(0,adam_epochs,1),adam_losses[4])
ax[1,1].set_xlabel("Epochs")
ax[1,1].set_ylabel("Loss")
ax[1,1].set_yscale("log")
ax[1,1].set_title('Adam PDE Loss')

plt.tight_layout()
plt.show()

# lbfgs
fix, ax = plt.subplots(2,2,figsize=(8, 6))
ax[0,0].plot(np.arange(0,lbfgs_epochs,1),lbfgs_losses[1])
ax[0,0].set_xlabel("Epochs")
ax[0,0].set_ylabel("Loss")
ax[0,0].set_yscale("log")
ax[0,0].set_title('LBFGS t0 Loss')

ax[0,1].plot(np.arange(0,lbfgs_epochs,1),lbfgs_losses[2])
ax[0,1].set_xlabel("Epochs")
ax[0,1].set_ylabel("Loss")
ax[0,1].set_yscale("log")
ax[0,1].set_title('LBFGS r0 Loss')

ax[1,0].plot(np.arange(0,lbfgs_epochs,1),lbfgs_losses[3])
ax[1,0].set_xlabel("Epochs")
ax[1,0].set_ylabel("Loss")
ax[1,0].set_yscale("log")
ax[1,0].set_title('LBFGS r1 Loss')

ax[1,1].plot(np.arange(0,lbfgs_epochs,1),lbfgs_losses[4])
ax[1,1].set_xlabel("Epochs")
ax[1,1].set_ylabel("Loss")
ax[1,1].set_yscale("log")
ax[1,1].set_title('LBFGS PDE Loss')

plt.tight_layout()
plt.show()

errs = np.concatenate((adam_losses,lbfgs_losses),axis=1)

fix, ax = plt.subplots(2,2,figsize=(8, 6))
ax[0,0].plot(np.arange(0,adam_epochs+lbfgs_epochs,1),errs[1])
ax[0,0].set_xlabel("Epochs")
ax[0,0].set_ylabel("Loss")
ax[0,0].set_yscale("log")
ax[0,0].set_title('t0 Loss')

ax[0,1].plot(np.arange(0,adam_epochs+lbfgs_epochs,1),errs[2])
ax[0,1].set_xlabel("Epochs")
ax[0,1].set_ylabel("Loss")
ax[0,1].set_yscale("log")
ax[0,1].set_title('r0 Loss')

ax[1,0].plot(np.arange(0,adam_epochs+lbfgs_epochs,1),errs[3])
ax[1,0].set_xlabel("Epochs")
ax[1,0].set_ylabel("Loss")
ax[1,0].set_yscale("log")
ax[1,0].set_title('r1 Loss')

ax[1,1].plot(np.arange(0,adam_epochs+lbfgs_epochs,1),errs[4])
ax[1,1].set_xlabel("Epochs")
ax[1,1].set_ylabel("Loss")
ax[1,1].set_yscale("log")
ax[1,1].set_title('PDE Loss')

plt.tight_layout()
plt.show()


# %% animation stuff
# import matplotlib.animation as ani

# # plot t as moving
# fig, ax = plt.subplots()
# plt.xlabel('r (m)')
# plt.ylabel('c (mol/m3)')
# plt.ylim(-100,15000) # max = cmax
# plt.xlim(0,Rs) # max = Rs
# line1, = ax.plot(Rs*CVM_r,c_max*CVM_c[...,0])
# line2, = ax.plot(Rs*ms_r[...,0],c_max*ms_c[...,0])
# fig.legend(['CVM_soln','neural_network'])
# def animate(i):
#     line1.set_ydata(c_max*CVM_c[...,i])
#     line2.set_ydata(c_max*ms_c[...,i])  # update the data.
#     plt.title(f"t = {i}s")
#     return line1, line2,

# vid = ani.FuncAnimation(fig, animate, frames=1000, interval=5)
# writervideo = ani.FFMpegWriter(fps=60)
# vid.save('v3_moving_t.gif',writer='ffmpeg',fps=60)
# plt.show

# # plot r as moving
# fig, ax = plt.subplots()
# plt.xlabel('t (s)')
# plt.ylabel('c (mol/m3)')
# plt.ylim(-100,15000) # max = cmax
# plt.xlim(0,tmax) # max = Rs
# line1, = ax.plot(tmax*CVM_t,CVM_c[0])
# line2, = ax.plot(tmax*ms_t[0],c_max*ms_c[0])
# fig.legend(['CVM_soln','neural_network'])

# def animate(i):
#     line1.set_ydata(c_max*CVM_c[i])
#     line2.set_ydata(c_max*ms_c[i])  # update the data.
#     plt.title(f"r = {round(i*1e-2,2)}")
#     return line1, line2,

# vid = ani.FuncAnimation(fig, animate, frames=100, interval=50)
# writervideo = ani.FFMpegWriter(fps=60)
# vid.save('v3_moving_r.gif',writer='ffmpeg',fps=60)
# plt.show