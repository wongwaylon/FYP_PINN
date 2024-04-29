import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import math
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# create the datasets required for PINN training
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
    r1_c = torch.full((N,1),float(c_r1))
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

# turn 1D axes into a 2D array and inputs into the PINN
def meshgrid_plot(r, t):
    ms_r, ms_t = np.meshgrid(r, t, indexing='ij')
    r = np.ravel(ms_r).reshape(-1,1)
    t = np.ravel(ms_t).reshape(-1,1)
    pt_r = torch.from_numpy(r).float().requires_grad_(True).to(device)
    pt_t = torch.from_numpy(t).float().requires_grad_(True).to(device)
    return pt_r, pt_t, ms_r, ms_t

# automatic differentiation
def dydx(y,x):
    dy_dx = torch.autograd.grad(y, x, grad_outputs=torch.ones_like(y), create_graph=True)[0]
    return dy_dx

# analytical solution
def analy_soln(x,t):
    trans = 0
    for i in range(1000):
        k = i+1
        trans += (((-1)**k)/k) * math.exp(-k**2 * math.pi**2 *t) * math.sin(k*math.pi*x)
    c = x + (2/math.pi)*trans;
    return c

# PINN class
class PINN(nn.Module):
    def __init__(
        self,
        input_dim, # number of inputs
        output_dim, # number of outputs
        lays=3, # number of hidden layers
        n_units=100, # neurons per layer
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
        
        # initialise neural network
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
        
        # adam optimiser
        self.lr = 1e-3 # adam learning rate
        self.adam_optim = optim.Adam(self.parameters(), lr=self.lr)
    
    # prediction through the neural network
    def forward(self, r,t):
        x = torch.cat([r,t],axis=1) # from 2 arrays of 1 columns each to 1 array of 2 columns
        ins = self.input_layer(x)
        mids = self.hidden_layers(ins)
        outs = self.output_layer(mids)
        return outs
    
    # put training data into the dataloader
    def load_data(self, batch, t0_data, r0_data, r1_data, phy_data):
        self.batch = batch
        self.t0_loader = DataLoader(dataset=t0_data, batch_size=batch, shuffle=False)
        self.r0_loader = DataLoader(dataset=r0_data, batch_size=batch, shuffle=False)
        self.r1_loader = DataLoader(dataset=r1_data, batch_size=batch, shuffle=False)
        self.phy_loader = DataLoader(dataset=phy_data, batch_size=batch, shuffle=False)
        
        # create validation data
        t, x, c = next(iter(self.phy_loader))
        c_val = np.zeros([len(c),1])
        for i in range(len(c)):
            c_val[i] = analy_soln(x[i].detach().numpy(),t[i].detach().numpy())
        c_val = torch.from_numpy(c_val).float().requires_grad_(True).to(device)
        val_data = TensorDataset(t, x, c_val)
        self.val_loader = DataLoader(dataset=val_data, batch_size=batch, shuffle=False)
    
    # compute loss function    
    def pinn_loss(self):
        
        # initial condition
        t_t0, r_t0, c_t0 = next(iter(self.t0_loader))
        r_t0 = r_t0.to(self.device)
        t_t0 = t_t0.to(self.device)
        c_t0 = c_t0.to(self.device)
        guess_t0 = self.forward(r_t0,t_t0)
        mse_t0 = self.loss_func(guess_t0,c_t0)
        
        # r=0 condition
        t_r0, r_r0, c_r0 = next(iter(self.r0_loader))
        r_r0 = r_r0.to(self.device)
        t_r0 = t_r0.to(self.device)
        c_r0 = c_r0.to(self.device)
        guess_r0 = self.forward(r_r0,t_r0)
        mse_r0 = self.loss_func(guess_r0,c_r0)
        
        # r=1 condition
        t_r1, r_r1, c_r1 = next(iter(self.r1_loader))
        r_r1 = r_r1.to(self.device)
        t_r1 = t_r1.to(self.device)
        c_r1 = c_r1.to(self.device)
        guess_r1 = self.forward(r_r1,t_r1)
        mse_r1 = self.loss_func(guess_r1,c_r1)
        
            
        t_phy, r_phy, c_phy = next(iter(self.phy_loader))
        r_phy = r_phy.requires_grad_(True).to(self.device)
        t_phy = t_phy.requires_grad_(True).to(self.device)
        c_phy = c_phy.to(self.device)
        guess_pde = self.forward(r_phy,t_phy)
        pde = self.phy_loss(r_phy,t_phy,guess_pde)
        mse_phy = self.loss_func(pde,c_phy)
        
        train_loss = mse_t0 + mse_r0 + mse_r1 + mse_phy

        self.train_loss = train_loss
        self.t0_loss = mse_t0
        self.r0_loss = mse_r0
        self.r1_loss = mse_r1
        self.pde_loss = mse_phy
        return train_loss
    
    # PDE as a loss function f
    def phy_loss(self,r,t,c):
        dcdr = dydx(c,r)
        d2cdr2 = dydx(dcdr,r)
        dcdt = dydx(c,t)
        pde = dcdt - d2cdr2
        return pde
    
    # calculate validation loss
    def val_loss(self):
        # self.eval()
        r, t, c = next(iter(self.val_loader))
        # for r, t, c in self.val_loader:
        guess_c = self.predict(r,t)
        val_losses = self.loss_func(guess_c, c)
        return val_losses 
    
    # training loop
    def train_loop(self, adam_epochs=10000, adam_lr=1e-3, plots=1, seed=None):
        adam_loss = []
        adam_t0 = []
        adam_r0 = []
        adam_r1 = []
        adam_pde = []
        adam_val = []

        for g in self.adam_optim.param_groups:
            g['lr'] = adam_lr
            
        # start training loop
        for epoch in range(adam_epochs):
            self.train()
            self.adam_optim.zero_grad()
            loss = self.pinn_loss()
            loss.backward()
            self.adam_optim.step()
                
            with torch.autograd.no_grad():
                print("Epoch:",epoch,"\nAdam Loss:",loss.data, self.t0_loss.data, self.r0_loss.data, self.r1_loss.data, self.pde_loss.data)
                
            with torch.autograd.no_grad():
                # save data
                adam_loss.append(self.train_loss.data)
                adam_t0.append(self.t0_loss.data)
                adam_r0.append(self.r0_loss.data)
                adam_r1.append(self.r1_loss.data)
                adam_pde.append(self.pde_loss.data)
                
            val_losses = self.val_loss()
            adam_val.append(val_losses.data)
            
            # training visualisation
            if epoch % (adam_epochs/plots) == 0:
                self.visualisation(epoch)    

        # losses: loss, t0, r0, r1, pde, val
        adam_losses = [adam_loss,adam_t0,adam_r0,adam_r1,adam_pde,adam_val]
        return adam_losses
    
    # compute c without the gradients
    def predict(self, r, t):
        self.eval()
        c = self.forward(r, t)
        return c

    # create 3D surface plot of the PINN prediction
    def visualisation(self, epoch):
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        r=np.linspace(0,1,101) # 100 points
        t=np.linspace(0,1,101) # 100 points
        net_r, net_t, ms_r, ms_t = meshgrid_plot(r,t)
        c = self.predict(net_r, net_t)
        c = c.detach().cpu().numpy()
        ms_c = c.reshape(ms_r.shape)
        ax.plot_surface(ms_r,ms_t,ms_c, cmap='viridis', antialiased=False)
        plt.title(f"Training Step {epoch}")
        plt.xlabel("r")
        plt.ylabel("t")
        plt.show()