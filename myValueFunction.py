# MIT License

# Copyright (c) 2024 Feiyu Yang 杨飞宇

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
import torch
import torch.nn as nn
class CausalInferencedCritic(nn.Module):
    def __init__(self, obs_dim, hidden_dim):
        super(CausalInferencedCritic, self).__init__()

        self.agent_specific_layers = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim), 
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim), 
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, observationWithz):
 
        return self.agent_specific_layers(observationWithz)
