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
from myGCN import GCNWithResidual
from myMixedGuassianVae import MixtureGaussianVAE
import torch.nn.functional as F
from plot_tasks import plot_vectors,plot_combined_error_boxplots,plot_combined_error_violinplots

class CausalInferenceModule(torch.nn.Module):
    def __init__(self, edge_index, node_cnt, device, index, z_dim) -> None:
        super(CausalInferenceModule, self).__init__()
        self.node_cnt=node_cnt
        self.edge_index=edge_index
        self.feature_padding_to=64
        self.gcn_hid_dim=self.feature_padding_to
        self.gcn_out_dim=self.feature_padding_to
        self.feature_raw_dims=[58,58,58,82,82,82,82,58,58,58,74,58,58,58,58,58,58,74]
   
        self.gcn=GCNWithResidual(self.feature_padding_to,self.gcn_hid_dim,self.gcn_out_dim,self.node_cnt,self.feature_padding_to,device).to(device)
        self.edge_index_tensor=self.edge_index.to(device)

        self.index=index
        self.maxActionDim=16
        self.ActionEncodeLen=4
        self.vae=MixtureGaussianVAE(self.gcn_out_dim, 64, z_dim,self.feature_raw_dims[self.index],self.ActionEncodeLen).to(device)
        

        self.vae_loss_factor=0.8
        self.predictor_loss_factor=1.0-self.vae_loss_factor

    def forward(self,feature_list_byAgentIdx,preActs,st_list_byAgentIdx,rs):
        batchsize=preActs.shape[0]
        gcn_out=self.gcn(feature_list_byAgentIdx, self.edge_index_tensor)
        assert not torch.isnan(gcn_out).any(), "Input to the layer contains NaN"
      
        gcn_out_reshaped = gcn_out.view(batchsize, self.node_cnt, self.gcn_out_dim)

        nodeIndex =self.index
        node_features = gcn_out_reshaped[:, nodeIndex, :]
        preSt_recon, mu, log_var, pi, predictor_st_out,predictor_r_out, (z,st)=self.vae(node_features,preActs[:,nodeIndex,:],st_list_byAgentIdx[nodeIndex].squeeze(-2))
        assert not torch.isnan(log_var).any(), "Input to the layer contains NaN"
        if st.dim()==1:
            st=st.unsqueeze(0)                           
        
        return torch.concat((z,st),dim=-1),torch.concat((z,st),dim=-1),(preSt_recon,feature_list_byAgentIdx[nodeIndex].squeeze(1),mu,log_var,pi),(predictor_st_out,st_list_byAgentIdx[nodeIndex].squeeze(1)),(predictor_r_out,rs[:,nodeIndex,:])
    


