# CausalRL_TSSRN
A simple implementation of the method described in Causal Reinforcement Learning for Train Scheduling on Single-track Railway Networks

# Summary
This project implements the algorithm described in the paper Causal Reinforcement Learning for Train Scheduling on Single-track Railway Networks, and it performs computations on one of the case studies. The reward progression of all agents during training (evaluated every 25 iterations) is shown in the figure below.
![tf_scr_st1](https://github.com/user-attachments/assets/69aec488-0843-4e87-9b11-bb56d42dfa21)
![tf_scr_st2](https://github.com/user-attachments/assets/759d0cd0-b70a-4366-bd64-1ed9a98a4985)
This project serves as a rapid evaluation version of the algorithm presented in the paper. Users can adjust hyperparameters to explore more complex scenarios. 
The environment in this project adheres to the Gymnasium environment interface, and the algorithm is implemented using PyTorch.

# Requirements
- gym==0.26.2
- gym-notices==0.0.8
- gymnasium==0.28.1
- gymnasium-notices==0.0.1
- matplotlib==3.8.3
- tensorboard==2.16.2
- tensorboard-data-server==0.7.2
- tensorboard-plugin-wit
- tensorboardX==2.6.2.2
- tensorflow==2.16.1
- tensorflow-intel==2.16.1
- tensorflow-io-gcs-filesystem==0.31.0
- torch==2.2.1+cu121
- torch_geometric==2.5.1
- torchaudio==2.2.1+cu121
- torchvision==0.17.1+cu121
- 
# Usage
To run this case:
```python
  python run_clc_MAA2C.py
```

# Acknowledgments
The reinforcement learning part in this project is inspired by Chenglong Chen's pytorch-DRL project:
https://github.com/ChenglongChen/pytorch-DRL

# License
Mit License
