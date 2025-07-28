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

# This code is based on the project pytorch-DRL by Chenglong Chen (2017).
# Modifications by Feiyu Yang (2024).
# 
# Licensed under the MIT License.

# MIT License

# Copyright (c) 2017 Chenglong Chen

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

import torch as th
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
#3,entF=0.5
#5,entF=2
#6,entF=1000
#7,entF=1
#8,0.01
#9,0.05
writer = SummaryWriter('runs\\MkedEntPcy_trainActSamp_EntF8en2_criticStzsDetached9')
globalEpisode=[0]

def addScalarFcn(tag,value,globalstep):
    writer.add_scalar(tag,value,global_step=globalstep)

def addTextFcn(tag,value,globalstep):
    writer.add_text(tag,value,global_step=globalstep)

def identity(x):
    return x


# def entropy(p):
#     return -th.sum(p * th.log(p+1e-5), -1)

def masked_entropy(action_log_probs, mask):
    p=th.exp(action_log_probs)
    # 确保 mask 中不止一个位置为 1 的行参与计算
    valid_rows = (mask.sum(dim=-1) > 1)


    # 计算熵，避免 log(0)
    entropy = -th.sum(p * th.log(p + 1e-5), dim=-1)

    # 归一化熵
    entropy = th.sum(entropy)/valid_rows.sum().item()

    return entropy


def kl_log_probs(log_p1, log_p2):
    return -th.sum(th.exp(log_p1)*(log_p2 - log_p1), 1)


def index_to_one_hot(index, dim):
    if isinstance(index, np.int) or isinstance(index, np.int64):
        one_hot = np.zeros(dim)
        one_hot[index] = 1.
    else:
        one_hot = np.zeros((len(index), dim))
        one_hot[np.arange(len(index)), index] = 1.
    return one_hot

    
def action2code(action,isNumpy=False,iscuda=False):
    codes=[
        th.tensor([0,0,0,0],dtype=th.float32),
        th.tensor([0,0,0,1],dtype=th.float32),
        th.tensor([0,0,1,0],dtype=th.float32),
        th.tensor([0,0,1,1],dtype=th.float32),
        th.tensor([0,1,0,0],dtype=th.float32),
        th.tensor([0,1,0,1],dtype=th.float32),
        th.tensor([0,1,1,0],dtype=th.float32),
        th.tensor([0,1,1,1],dtype=th.float32),
        th.tensor([1,0,0,0],dtype=th.float32),
        th.tensor([1,0,0,1],dtype=th.float32),
        th.tensor([1,0,1,0],dtype=th.float32),
        th.tensor([1,0,1,1],dtype=th.float32),
        th.tensor([1,1,0,0],dtype=th.float32),
        th.tensor([1,1,0,1],dtype=th.float32),
        th.tensor([1,1,1,0],dtype=th.float32),
        th.tensor([1,1,1,1],dtype=th.float32),
        ]
    assert action>=0 and action<16,'action error'
    if isNumpy:
            return codes[action].numpy()
    else:
        if iscuda:
            return codes[action].cuda()
        else:
            return codes[action]

def binary_to_one_hot(binary_tensor, action_dim,use_gpu=True):
    """
    Convert a batch of binary tensor actions to one-hot encoded actions.

    Args:
    - binary_tensor (torch.Tensor): A tensor of shape [batch, 4] with binary values.
    - action_dim (int): The dimension of the action space (must be <= 16).

    Returns:
    - torch.Tensor: A tensor of shape [batch, action_dim] with one-hot encoded actions.
    """
    # Ensure the action dimension is valid
    assert action_dim <= 16, "actionDim must be less than or equal to 16"

    # Calculate the decimal values from the binary tensor
    # Example: [0, 0, 1, 1] --> 3
    device='cuda'
    if not use_gpu:
        device='cpu'
    multipliers = th.tensor([8, 4, 2, 1], dtype=th.float32,device=device)
    
    decimal_values = th.matmul(binary_tensor, multipliers).to(th.int64)

    # Generate one-hot encoded vectors
    one_hot_encoded = th.nn.functional.one_hot(decimal_values, num_classes=action_dim)

    return one_hot_encoded

def to_tensor_var(x, use_cuda=True, dtype="float"):
    LongTensor = th.cuda.LongTensor if use_cuda else th.LongTensor
    ByteTensor = th.cuda.ByteTensor if use_cuda else th.ByteTensor
    if dtype == "float":
        x = np.array(x, dtype=np.float32).tolist()
        if use_cuda:
            return Variable(th.tensor(x,dtype=th.float32)).cuda()
        else:
            return Variable(th.tensor(x,dtype=th.float32))
    elif dtype == "long":
        x = np.array(x, dtype=np.long).tolist()
        return Variable(LongTensor(x))
    elif dtype == "byte":
        x = np.array(x, dtype=np.byte).tolist()
        return Variable(ByteTensor(x))
    else:
        x = np.array(x, dtype=np.float32).tolist()
        if use_cuda:
            return Variable(th.tensor(x,dtype=th.float32)).cuda()
        else:
            return Variable(th.tensor(x,dtype=th.float32))

def npABCDToTensorList(ABCD_batch,use_cuda=True):
    FloatTensor = th.cuda.FloatTensor if use_cuda else th.FloatTensor
    LongTensor = th.cuda.LongTensor if use_cuda else th.LongTensor
    ByteTensor = th.cuda.ByteTensor if use_cuda else th.ByteTensor
    A_batch=[[] for _ in range(18)]
    B_batch=[]
    C_batch=[[] for _ in range(18)]
    D_batch=[]
    E_batch=[[] for _ in range(18)]
    if isinstance(ABCD_batch[0],list):#interact 中单个state来获取action
        a,b,c,d,e=ABCD_batch
        [A_batch[idx].append(Variable(FloatTensor(a[idx]))) for idx in range(len(a))]
        B_batch.append(Variable(LongTensor(b)).unsqueeze(0))
        [C_batch[idx].append(Variable(FloatTensor(c[idx]))) for idx in range(len(c))]
        D_batch.append(Variable(FloatTensor(d)).unsqueeze(0))
        [E_batch[idx].append(Variable(LongTensor(e[idx]))) for idx in range(len(e))]
    else:
        for abcde in ABCD_batch:#train 中batch的state来训练
            a,b,c,d,e=abcde
            [A_batch[idx].append(Variable(FloatTensor(a[idx]))) for idx in range(len(a))]
            B_batch.append(Variable(LongTensor(b)).unsqueeze(0))
            [C_batch[idx].append(Variable(FloatTensor(c[idx]))) for idx in range(len(c))]
            D_batch.append(Variable(FloatTensor(d)).unsqueeze(0))
            [E_batch[idx].append(Variable(LongTensor(e[idx]))) for idx in range(len(e))]

    A_batch=[th.stack(ts,dim=0).squeeze(1).squeeze(1) for ts in A_batch]
    B_batch=th.stack(B_batch,dim=0).squeeze(1).squeeze(1)
    C_batch=[th.stack(ts,dim=0).squeeze(1).squeeze(1) for ts in C_batch]
    D_batch=th.stack(D_batch,dim=0).squeeze(1).squeeze(1)
    E_batch=[th.stack(ts,dim=0).squeeze(1).squeeze(1) for ts in E_batch]
    return (A_batch,B_batch,C_batch,D_batch,E_batch) # ( list18->[batchsize,featuredim],[batchsize,18,4],list18->[batchsize,featuredim],[batchsize,18,1] )
    
# def vae_loss_function(recon_x, x, mu, log_var, pi):
#     ##全小误差 MSE 34、MAE 48 #中等误差 MSE 21、MAE34 #有大误差MSE 14 MAE 22
#     lossMSE = F.mse_loss(recon_x, x, reduction='mean')#mse对小幅度的回归loss不敏感8e-5 0.0001 0.0001 0.0052
#     lossMAE = F.l1_loss(recon_x, x, reduction='mean') #0.0007 0.0007 0.0069 #没效果
#     beta=0.0005
#     beta2=0.1
    
#     # 对于三个分量的混合高斯，先验均值分别为 0.0 、 0.5 和 1.0
#     mu_prior = th.zeros_like(mu).to(mu.device)
#     mu_prior[:,0,:]=th.ones_like(mu[:,0,:],device=mu.device)*0.0
#     mu_prior[:,1,:]=th.ones_like(mu[:,1,:],device=mu.device)*0.5
#     mu_prior[:,2,:]=th.ones_like(mu[:,2,:],device=mu.device)*1.0
    
#     # 计算每个分量的KLD
#     kld1= 1 + log_var[:, 0, :] - mu[:, 0, :].pow(2) - log_var[:, 0, :].exp()
#     lossKLD_1 = -0.5 * th.mean(kld1,dim=(0,1))

#     kld2= 1 + log_var[:, 1, :] - (mu[:, 1, :] - mu_prior[:, 1, :]).pow(2) - log_var[:, 1, :].exp()
#     lossKLD_2 = -0.5 * th.mean(kld2,dim=(0,1))

#     kld3= 1 + log_var[:, 2, :] - (mu[:, 2, :] - mu_prior[:, 2, :]).pow(2) - log_var[:, 2, :].exp()
#     lossKLD_3 = -0.5 * th.mean(kld3,dim=(0,1))
    
#     # 加权求和
#     lossKLD = pi[:, 0] * lossKLD_1 + pi[:, 1] * lossKLD_2 + pi[:,2]*lossKLD_3

#     rtn=(1-beta2)*lossMSE + beta2*lossMAE + beta*(lossKLD.mean())#init:0.0147+0.0817+0.0019
#     assert not th.isinf(rtn).any(),'?'
#     assert not th.isnan(rtn).any(),'?'
#     return rtn

#先验方差为sigma_prior
def vae_loss_function(recon_x, x, mu, log_var, pi):
    # 计算重构误差
    lossMSE = F.mse_loss(recon_x, x, reduction='mean')
    lossMAE = F.l1_loss(recon_x, x, reduction='mean')

    sigma_prior=th.ones_like(mu)*0.5#先验的标准差0.5，方差0.25
    beta = 0.005
    beta2 = 0.1
    
    # 对于三个分量的混合高斯，先验均值分别为 0.0、0.5 和 1.0
    mu_prior = th.zeros_like(mu).to(mu.device)
    mu_prior[:, 0, :] = th.ones_like(mu[:, 0, :], device=mu.device) * 0.0
    mu_prior[:, 1, :] = th.ones_like(mu[:, 1, :], device=mu.device) * 0.5
    mu_prior[:, 2, :] = th.ones_like(mu[:, 2, :], device=mu.device) * 1.0
    
    # sigma_prior 参数化为输入参数
    sigma_prior = sigma_prior.to(mu.device)
    
    # 计算每个分量的KLD
    kld1 = th.log(sigma_prior[:, 0, :] / th.exp(0.5 * log_var[:, 0, :])) + \
           (th.exp(log_var[:, 0, :]) + (mu[:, 0, :] - mu_prior[:, 0, :]) ** 2) / (2 * sigma_prior[:, 0, :] ** 2) - 0.5
    lossKLD_1 = th.mean(kld1, dim=(0, 1))
    
    kld2 = th.log(sigma_prior[:, 1, :] / th.exp(0.5 * log_var[:, 1, :])) + \
           (th.exp(log_var[:, 1, :]) + (mu[:, 1, :] - mu_prior[:, 1, :]) ** 2) / (2 * sigma_prior[:, 1, :] ** 2) - 0.5
    lossKLD_2 = th.mean(kld2, dim=(0, 1))
    
    kld3 = th.log(sigma_prior[:, 2, :] / th.exp(0.5 * log_var[:, 2, :])) + \
           (th.exp(log_var[:, 2, :]) + (mu[:, 2, :] - mu_prior[:, 2, :]) ** 2) / (2 * sigma_prior[:, 2, :] ** 2) - 0.5
    lossKLD_3 = th.mean(kld3, dim=(0, 1))
    
    # 加权求和
    lossKLD = pi[:, 0].unsqueeze(-1) * lossKLD_1 + pi[:, 1].unsqueeze(-1) * lossKLD_2 + pi[:, 2].unsqueeze(-1) * lossKLD_3
    
    # 计算总损失
    rtn = (1 - beta2) * lossMSE + beta2 * lossMAE + beta * lossKLD.mean()
    
    assert not th.isinf(rtn).any(), '?'
    assert not th.isnan(rtn).any(), '?'
    
    return rtn
       


def agg_double_list(l):
    # l: [ [...], [...], [...] ]
    # l_i: result of each step in the i-th episode
    s = [np.sum(np.array(l_i), 0) for l_i in l]
    s_mu = np.mean(np.array(s), 0)#np.array(s)->[10,18,1]
    s_std = np.std(np.array(s), 0)
    return s_mu, s_std

def add_noise_to_logits(logits, std_dev=0.5):
    noise = th.randn_like(logits) * std_dev
    noisy_logits = logits + noise
    return noisy_logits