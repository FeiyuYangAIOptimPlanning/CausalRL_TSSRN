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
# import os
# os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data,Batch
from torch_geometric.utils import to_dense_batch

class GCNWithResidual(torch.nn.Module):
    def __init__(self, num_feat, num_hid,num_out,node_cnt,feature_padding_dim,device):
        super(GCNWithResidual, self).__init__()
        self.node_cnt=node_cnt
        self.feature_raw_dims=[58,58,58,82,82,82,82,58,58,58,74,58,58,58,58,58,58,74]
        self.featurePadding_layers_byNodes = nn.ModuleList([ nn.Linear(self.feature_raw_dims[node_idx], feature_padding_dim).to(device) for node_idx in range(node_cnt) ])

        self.conv1 = GCNConv(num_feat, num_hid).to(device)
        self.conv2 = GCNConv(num_hid, num_out).to(device)

    def forward(self, feature_list_byAgentIdx, edge_index):
        for linearlayer in self.featurePadding_layers_byNodes:
            assert not torch.isnan(linearlayer.weight.data).any(), "Input to the layer contains NaN"
        xs=torch.stack([F.relu(self.featurePadding_layers_byNodes[node_idx](feature_list_byAgentIdx[node_idx])) for node_idx in range(self.node_cnt)],dim=-2).squeeze(1)
        for linearlayer in self.featurePadding_layers_byNodes:
            assert not torch.isnan(linearlayer.weight.data).any(), "Input to the layer contains NaN"

        data_list = []
        for i in range(xs.size(0)):
            data_single_graph = Data(x=xs[i], edge_index=edge_index)
            data_list.append(data_single_graph)
        batch_data = Batch.from_data_list(data_list)
        x, edge_index, batch = batch_data.x, batch_data.edge_index, batch_data.batch

        x1 = F.relu(self.conv1(x, edge_index))
        assert not torch.isnan(x1).any(), "Input to the layer contains NaN"

        x2 = F.relu(self.conv2(x1+x, edge_index))
        assert not torch.isnan(x2).any(), "Input to the layer contains NaN"
        x_out=x2+x1
        x_dense, _ = to_dense_batch(x_out, batch)

        return x_dense


