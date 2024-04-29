import numpy as np
import pandas as pd
import torch
import PINN_simple as PINN
from timeit import default_timer as timer

start = timer() # time code runtime
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# model parameters
randomseed = 12121
input_dim = 2
output_dim = 1
N = 1000
batch = 1000
epochs = 20000

# create training data
t0_data,r0_data,r1_data,phy_data = PINN.create_data(N,0,0,1,seed=randomseed)

# initialise neural network
net = PINN.PINN(input_dim, output_dim, seed=randomseed).to(device)

# load training data into the PINN
net.load_data(batch,t0_data,r0_data,r1_data,phy_data)

# train model
losses = net.train_loop(adam_epochs=epochs, plots=5, seed=randomseed)

# code runtime
end = timer()

# save model
name = "PINN_simple_e20k_N1000.pt"
torch.save(net.state_dict(),name)

# create dataframe for data
colnames = ['Final Loss', 't0 Loss', 'x0 Loss', 'x1 Loss', 'PDE Loss', 'Validation Loss']
new_losses = np.transpose(losses)
df1 = pd.DataFrame(data=new_losses, columns=colnames)

data = {'Final Loss': losses[0][-1].numpy(),
        'PDE Loss': losses[-2][-1].numpy(),
        'Validation Loss': losses[-1][-1].numpy(),
        't0 Loss': losses[1][-1].numpy(),
        'x0 Loss': losses[2][-1].numpy(),
        'x1 Loss': losses[3][-1].numpy(),
        'Runtime (s)': end-start}
df2 = pd.DataFrame(data=data, index=['e20k_N1000'])

# save data as excel
with pd.ExcelWriter('PINN_simple.xlsx', engine='xlsxwriter') as writer:
    df1.to_excel(writer, sheet_name='losses_e20k_N1000')
    df2.to_excel(writer, sheet_name='final_data')