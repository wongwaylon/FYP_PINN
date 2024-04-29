import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, ConcatDataset, DataLoader
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

# create validation data from the CVM solution imported
def CVM_data(r,t,c):
    net_r, net_t, grid_r, grid_t = meshgrid_plot(r,t)
    val_c = np.ravel(c).reshape(-1,1)
    net_c = torch.from_numpy(val_c).float().requires_grad_(True).to(device)
    return net_r, net_t, net_c

# sine activation function module
class Sin(torch.nn.Module):
  def forward(self, x):
    return torch.sin(x)

# PINN class
class PINN(nn.Module):
    def __init__(
        self,
        input_dim, # number of inputs
        output_dim, # number of outputs
        lays=3, # number of hidden layers
        n_units=256, # neurons per layer
        loss=nn.MSELoss(), # neural network loss
        act_func = nn.Tanh, # activation function
        device="cpu", # CPU/GPU
        seed=None, # random seed
        boundary_force=None # boundary forcing function
    ) -> None:
        super(PINN,self).__init__()
        
        # load basic terms
        if seed != None:
            torch.manual_seed(seed)  
        self.seed = seed
        self.loss_func = loss
        self.n_units = n_units
        self.device = device
        self.boundary_force = boundary_force
        
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
        # add sigmoid act func?
        
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
    
    # prediction through the neural network
    def forward(self, r,t):
        x = torch.cat([r,t],axis=1) # from 2 arrays of 1 columns each to 1 array of 2 columns
        ins = self.input_layer(x)
        mids = self.hidden_layers(ins)
        outs = self.output_layer(mids)
        return outs
    
    # modify PINN output given boundary function
    def guess_c(self, r, t):
        c = self.forward(r,t)
        if self.boundary_force == 't0':
            c = t*c
        elif self.boundary_force == 'r0':
            dcdr = dydx(c,r)
            c = c - r*dcdr + c*r**2
        elif self.boundary_force == 't0r0':
            dcdr = dydx(c,r)
            c = (c-r*dcdr)*t/(t+r**2) + c*t*r**2
        return c
    
    # values required for PINN model (PDE, etc.)
    def vals(self, tmax=1, rmax=1, cmax=1, D=1, theta_norm=0):
        self.tmax = tmax
        self.rmax = rmax
        self.cmax = cmax
        self.D = D
        self.theta_norm = theta_norm
    
    # put the training and validation data into the data loader
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
    
    # adaptive sampling function
    def change_data(self):
        if self.seed != None:
            torch.manual_seed(self.seed)   
        self.pde_visual() # visualise current PDE residuals
        
        # create random dataset
        N = self.phy_N
        new_t = torch.rand(N,1)
        new_r = torch.rand(N,1)
        new_c = torch.zeros(N,1)
        new_dataset = TensorDataset(new_t,new_r,new_c)
        old_dataset = self.phy_data
        new_data = ConcatDataset([old_dataset,new_dataset])
        dataload = DataLoader(dataset=new_data,batch_size=len(new_dataset)+self.phy_N)
        
        # compute PDE residuals with new dataset
        t, r, c = next(iter(dataload))
        t = t.requires_grad_(True).to(self.device)
        r = r.requires_grad_(True).to(self.device)
        c = c.to(self.device)
        # guess_c = self.forward(r,t)
        guess_c = self.predict(r,t)
        pde = self.phy_loss(r,t,guess_c)
        k1 = 2
        k2 = 0
        err_eq = torch.pow(pde, k1) / torch.pow(pde, k1).mean() + k2 # calculate MSE
        
        # replace original dataset with new dataset with probability function
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
        
        # visualise new PDE dataset
        plt.figure(0,figsize=(6,4))
        plt.scatter(t_data.detach(), r_data.detach())
        plt.title("New PDE Dataset")
        plt.xlabel("t")
        plt.ylabel("r")
        plt.xlim(-0.05, 1.05)
        plt.ylim(-0.05, 1.05)
        plt.show()
    
    # compute loss function
    def pinn_loss(self):
        
        # initial condition
        t_t0, r_t0, c_t0 = next(iter(self.t0_loader))
        r_t0 = r_t0.requires_grad_(True).to(self.device)
        t_t0 = t_t0.requires_grad_(True).to(self.device)
        c_t0 = c_t0.to(self.device)
        guess_t0 = self.guess_c(r_t0,t_t0)
        mse_t0 = self.loss_func(guess_t0,c_t0)
        
        # r=0 condition
        t_r0, r_r0, c_r0 = next(iter(self.r0_loader))
        r_r0 = r_r0.requires_grad_(True).to(self.device)
        t_r0 = t_r0.requires_grad_(True).to(self.device)
        c_r0 = c_r0.to(self.device)
        guess_r0 = self.guess_c(r_r0,t_r0)
        dcdx_r0 = dydx(guess_r0,r_r0)
        mse_r0 = self.loss_func(dcdx_r0,c_r0)
        
        # r=1 condition
        t_r1, r_r1, c_r1 = next(iter(self.r1_loader))
        r_r1 = r_r1.requires_grad_(True).to(self.device)
        t_r1 = t_r1.requires_grad_(True).to(self.device)
        c_r1 = c_r1.to(self.device)
        guess_r1 = self.guess_c(r_r1,t_r1)
        bc1 = self.rmax_cbounds(r_r1, t_r1, guess_r1)
        mse_r1 = self.loss_func(bc1,c_r1)
        
        # PDE
        t_phy, r_phy, c_phy = next(iter(self.phy_loader))
        r_phy = r_phy.requires_grad_(True).to(self.device)
        t_phy = t_phy.requires_grad_(True).to(self.device)
        c_phy = c_phy.to(self.device)
        guess_pde = self.guess_c(r_phy,t_phy)
        pde = self.phy_loss(r_phy,t_phy,guess_pde)
        mse_phy = self.loss_func(pde,c_phy)
        
        # loss depending on boundary forcing function
        if self.boundary_force == None:
            train_loss = mse_t0 + mse_r0 + mse_r1 + mse_phy
        elif self.boundary_force == 't0':
            train_loss = mse_r0 + mse_r1 + mse_phy
        elif self.boundary_force == 't0':
            train_loss = mse_t0 + mse_r1 + mse_phy
        elif self.boundary_force == 't0r0':
            train_loss = mse_r1 + mse_phy
        
        # save losses
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
        pde = r*dcdt*(self.rmax**2)/(self.D*self.tmax) - (1+self.theta_norm*c)*(r*d2cdr2 + dcdr*2) - r*self.theta_norm*dcdr**2
        return pde
    
    # boundary condition at r=1
    def rmax_cbounds(self,r,t,c):
        dcdr = dydx(c,r)
        bc1_val = -(1+self.theta_norm*c)*dcdr
        return bc1_val
    
    # calculate validation loss
    def val_loss(self):
        r, t, c = next(iter(self.val_loader))
        guess_c = self.predict(r,t)
        val_losses = self.loss_func(guess_c, c)
        return val_losses
    
    # closure function for L-BFGS optimiser
    def closure(self):
        self.lbfgs_optim.zero_grad()
        loss = self.pinn_loss()
        loss.backward()
        return loss
    
    # training loop
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
            g['lr'] = adam_lr
        
        # loop if have adaptive sampling
        for i in range(runs):
            if data_change == True:
                self.change_data(seed=seed)
            
            # start Adam training loop
            for epoch in range(adam_epochs):
                self.train()
                for j, data in enumerate(self.phy_loader):
                    self.adam_optim.zero_grad()
                
                    loss = self.pinn_loss()
                    loss.backward()
                    self.adam_optim.step()
                    
                    with torch.autograd.no_grad():
                        print("Run",i,"Epoch:",epoch,"Batch:",j+1,"\nAdam Loss:",loss.data, self.t0_loss.data, self.r0_loss.data, self.r1_loss.data, self.pde_loss.data)
                    
                with torch.autograd.no_grad():
                    # save data
                    adam_loss.append(self.train_loss.data)
                    adam_t0.append(self.t0_loss.data)
                    adam_r0.append(self.r0_loss.data)
                    adam_r1.append(self.r1_loss.data)
                    adam_pde.append(self.pde_loss.data)
                    
                val_losses = self.val_loss()
                adam_val.append(val_losses.data)
                
                # visualise PINN prediction
                if epoch % (adam_epochs/plots) == 0:
                    self.visualisation(epoch)
            
            # start L-BFGS training loop
            for epoch in range(lbfgs_epochs):
                self.train()
                for j, data in enumerate(self.phy_loader):
                    self.lbfgs_optim.step(self.closure)
                    
                    with torch.autograd.no_grad():
                        print("Run",i,"Epoch:",epoch,"Batch:",j+1,"\nLBFGS Loss:",self.train_loss.data, self.t0_loss.data, self.r0_loss.data, self.r1_loss.data, self.pde_loss.data)
                
                with torch.autograd.no_grad(): 
                    # save data
                    lbfgs_loss.append(self.train_loss.data)
                    lbfgs_t0.append(self.t0_loss.data)
                    lbfgs_r0.append(self.r0_loss.data)
                    lbfgs_r1.append(self.r1_loss.data)
                    lbfgs_pde.append(self.pde_loss.data)
                
                val_error = self.val_loss()
                lbfgs_val.append(val_error.data)
                
                # visualise PINN prediction
                if epoch % lbfgs_epochs == 0:
                    self.visualisation(epoch)

        # losses: loss, t0, r0, r1, pde, val
        adam_losses = [adam_loss,adam_t0,adam_r0,adam_r1,adam_pde,adam_val]
        lbfgs_losses = [lbfgs_loss,lbfgs_t0,lbfgs_r0,lbfgs_r1,lbfgs_pde,lbfgs_val]
        return adam_losses, lbfgs_losses
    
    # compute c without gradients        
    def predict(self, r, t):
        self.eval()
        c = self.guess_c(r,t)
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
        ax.plot_surface(ms_r*self.rmax,self.tmax*ms_t,ms_c*self.cmax, cmap='viridis', antialiased=False)
        plt.title(f"Training Step {epoch}")
        plt.xlabel("r")
        plt.ylabel("t")
        plt.show()
    
    # create 3D surface plot of PDE residuals
    def pde_visual(self):
        r=np.linspace(0,1,101)[1:] # 100 points
        t=np.linspace(0,1,101)[1:] # 100 points
        net_r, net_t, ms_r, ms_t = meshgrid_plot(r,t)
        net_c = self.guess_c(net_r,net_t)
        c_pde = self.phy_loss(net_r, net_t, net_c)
        vals = torch.zeros(len(c_pde),1)
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
        return pde_loss.detach().cpu().numpy()
    
    # compute validation loss for the whole validation dataset
    def val_err(self):
        val_data = self.val_data
        dataload = DataLoader(dataset=val_data,batch_size=len(val_data))
        t, r, c = next(iter(dataload))
        guess_c = self.predict(r, t)
        val_err = self.loss_func(guess_c,c)
        return val_err.detach().cpu().numpy()