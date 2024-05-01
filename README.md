# FYP_PINN
University of Bath Final Year Project: Lithium-Ion Battery State Predictions Using Physics-Informed Neural Networks (PINNs)

# Abstract
As the world moves towards a net zero environment, projects to electrify transportation and the energy sector have been implemented to reduce the impacts of climate change, leading to a rise in Lithium-ion (Li-ion) battery demand. To anticipate the soar in battery demand, a deep understanding of battery dynamics is necessary to develop large and reliable Li-ion batteries. One way to predict the state of charge of a Li-ion battery is through the partial differential equation (PDE) of Fick’s law of diffusion in a single particle model. Currently, most of these models are computed through numerical methods, which use complex numerical algorithms and can require high computational powers for complex cases.  
  
Given the rise of neural network usage over the years, research has been conducted to incorporate physical governing equations into neural networks to solve PDEs, creating Physics-Informed Neural Networks (PINNs). The advantage of PINNs lies in their ability to solve PDEs without the need for complex numerical algorithms and can be solved for all values in the domain, similar to analytical solutions.  
  
This project aims to develop a PINN code to solve a spherical diffusion equation in the form of a PDE to predict the Li-ion concentrations in battery cathodes and anodes. The model predictions will be compared with concentration calculated using the control volume method (CVM) to determine its accuracy.
A PINN model was created using the PyTorch library in Python, where a range of network and training parameters were tested to reach a result with an optimal balance between network accuracy and training time. The resultant network is a 4x256 neural network with tanh as the activation function, the output of the network is integrated with a distance function to ensure the initial condition is satisfied. The network was trained using a 2000 data point dataset for each loss term in batch sizes of 128 over 5000 epochs in the Adam optimiser followed by 100 epochs through the L-BFGS optimiser.  
  
The PINN was trained for 1 hour and achieved a training loss of 1.38E-03 and a validation loss of 2.51E-06, with the execution time of the trained model being under 1 second. It is believed that in the long run, the training time can be offset by the short code execution time the more occasions the network is used to predict Li-ion concentrations. The accuracies of the model prediction were also on par with existing battery PINN models.

# Details
| File | Description |
| --- | --- |
| `PINN_code.py` | PINN model |
| `PINN_train.py` | PINN training file |
| `SolveDiffusionStress.py` | Compute spherical diffusion using the control volume method |
| `CVM_c.csv` | Control volume method concentration solution |
| `CVM_c.csv` | Control volume method concentration solution for $\theta =0$ |
| `CVM_r.csv` | Control volume method spatial steps |
| `CVM_t.csv` | Control volume method time steps |
| `PINN_save_concentrations.py` | Save concentration predictions from different PINN models into Excel files |
| `PINN_save_images.py` | Plot and save PINN prediction graphs |
| `PINN_state_4x256tanh_b128N2k_a5k_l100_tmax1000_t0.pt` | Most successful trained PINN utilising a boundary forcing function at $t=0$ |
| `PINN Models` | Folder for trained PINNs and concentration prediction data |
| `Model Plots` | Folder including concentration prediction plots using different network/training parameters |
|`Simple PINN Pilot Study` | Pilot study for PINN coding using the diffusion equation |

# Trained PINN Naming Convention
`PINN_state_{layers}_{neurons per layer}{activation function}_b{batch size}N{training dataset size}_a{adam epochs}_l{lbfgs epochs}_tmax{max t value}+{boundary forcing / adaptive sampling}`
