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
from plot_tasks import plot_vectors,plot_combined_error_boxplots,plot_combined_error_violinplots

class MixtureGaussianVAE(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, z_dim,output_dim,action_code_dim,):
        super(MixtureGaussianVAE, self).__init__()
        self.actionCodeDim=action_code_dim
        self.multiDistCnt=3

        self.vae_fc1 = nn.Linear(input_dim, hidden_dim)
        self.vae_fc_mu = nn.Linear(hidden_dim, z_dim * self.multiDistCnt) 
        self.vae_fc_log_var = nn.Linear(hidden_dim, z_dim * self.multiDistCnt) 
        self.vae_fc_pi = nn.Linear(hidden_dim, self.multiDistCnt)

        self.vae_decoder_fc1 = nn.Linear(z_dim, hidden_dim)
        self.vae_decoder_output = nn.Linear(hidden_dim, output_dim)

        self.fc_predict=nn.Linear(z_dim+self.actionCodeDim,64)
        self.fc_predict_output=nn.Linear(64, output_dim)

        self.fc_reward_output=nn.Linear(64, 1)
        
    def vae_encode(self, x):
        batch_size=x.size(0)
        h = F.leaky_relu(self.vae_fc1(x))
        mu = F.sigmoid(self.vae_fc_mu(h)).view(batch_size, self.multiDistCnt, -1)
        log_var = F.tanh(self.vae_fc_log_var(h)).view(batch_size, self.multiDistCnt, -1)
        pi_logits = F.tanh(self.vae_fc_pi(h))
        pi = F.softmax(pi_logits, dim=1)
        return mu, log_var, pi
    
    def reparameterize(self, mu, log_var, pi):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        z = eps.mul(std).add_(mu)
        idx = torch.multinomial(pi, 1).squeeze(-1)
        selected_z = z[torch.arange(z.size(0)), idx]
        return selected_z
    def vae_decode(self,z):
        h = F.relu(self.vae_decoder_fc1(z))
        rtn= self.vae_decoder_output(h)  
        if (rtn > 1e7).any():
            print('large value error')
        if (rtn<-1e7).any():
            print('low value error')
        return rtn
    
    def predictor(self,z,preActCode):
        batch_size=z.size(0)
        x=torch.concat((z,preActCode),dim=-1)
        h=F.relu(self.fc_predict(x))

        out=self.fc_predict_output(h)

        out_r=self.fc_reward_output(h)
        return out,out_r
    
    def forward(self, preSt,preActCode,st):
    
        mu, log_var, pi = self.vae_encode(preSt)
        z = self.reparameterize(mu, log_var, pi)
        preSt_recon=self.vae_decode(z=z)
  
        predictor_st_out,predictor_r_out=self.predictor(z,preActCode)
        
        return preSt_recon, mu, log_var, pi, predictor_st_out,predictor_r_out,(z,st)
