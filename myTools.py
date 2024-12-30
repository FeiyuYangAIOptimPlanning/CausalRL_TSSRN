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
import torch.nn.functional as F
# from torch.utils.tensorboard import SummaryWriter
import numpy as np
from plot_tasks import plot_vectors,plot_combined_error_boxplots,plot_combined_error_violinplots

global_update_cnt_list=[0]
global_iter=[0]
codes=[
        torch.tensor([0,0,0,0],dtype=torch.float32),
        torch.tensor([0,0,0,1],dtype=torch.float32),
        torch.tensor([0,0,1,0],dtype=torch.float32),
        torch.tensor([0,0,1,1],dtype=torch.float32),
        torch.tensor([0,1,0,0],dtype=torch.float32),
        torch.tensor([0,1,0,1],dtype=torch.float32),
        torch.tensor([0,1,1,0],dtype=torch.float32),
        torch.tensor([0,1,1,1],dtype=torch.float32),
        torch.tensor([1,0,0,0],dtype=torch.float32),
        torch.tensor([1,0,0,1],dtype=torch.float32),
        torch.tensor([1,0,1,0],dtype=torch.float32),
        torch.tensor([1,0,1,1],dtype=torch.float32),
        torch.tensor([1,1,0,0],dtype=torch.float32),
        torch.tensor([1,1,0,1],dtype=torch.float32),
        torch.tensor([1,1,1,0],dtype=torch.float32),
        torch.tensor([1,1,1,1],dtype=torch.float32),
        ]
def action2code(action):
    assert action>=0 and action<16,'action error'
    return codes[action]

class SignPreservingLogScale(nn.Module):
    def forward(self, x):
        scaled_x = torch.zeros_like(x)
        pos_mask = x > 0
        neg_mask = x < 0
        scaled_x[pos_mask] = torch.log1p(x[pos_mask]) 
        scaled_x[neg_mask] = -torch.log1p(-x[neg_mask]) 
        return scaled_x

class MSLELoss(nn.Module):
    def __init__(self):
        super(MSLELoss, self).__init__()
        self.sign_preserving_log_scale = SignPreservingLogScale()

    def forward(self, y_pred, y_true):
        log_y_pred = self.sign_preserving_log_scale(y_pred)
        log_y_true = self.sign_preserving_log_scale(y_true)
        return F.mse_loss(log_y_pred,log_y_true,reduction='mean')
    
def flatten_list(nested_list):
        """
        """
        flat_list = []
        for item in nested_list:
            if isinstance(item, list):
                flat_list.extend(flatten_list(item))  
            else:
                flat_list.append(item)
        return flat_list