"""
@Author: Xinyuan Li
@Contact: xinyuanli@whu.edu.cn
@File: model.py
@Time: 2025/2/24 6:35 PM
"""

import os
import sys
project_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_path not in sys.path:
    sys.path.append(project_path)
import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import fps, global_max_pool, radius
from torch_geometric.nn import MLP, PointNetConv,GATConv,PointTransformerConv
from torch_geometric.utils import scatter
from torch import nn


class Synapformer(torch.nn.Module):
    def __init__(self, num_nodes, dataset_num_features, hidden_channels,represent_features, heads, dataset_num_classes): 
        super().__init__()
        self.conv1 = GATConv(dataset_num_features, hidden_channels, heads, dropout=0.6) 
        self.conv2 = GATConv(hidden_channels * heads, represent_features, heads=1,
                             concat=False, dropout=0.6)
        self.x = nn.Parameter(torch.Tensor(num_nodes, dataset_num_features))
        torch.nn.init.xavier_uniform(self.x, gain=1)
        self.cache_x = None
        
        #TODO
        
        self.synapse_encoder=synapse_former()
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=3)
        self.classmodel=MLP([represent_features+512+512, 512,dataset_num_classes])


    def forward(self,edge_index,edge_attr,synapse,synapse_index,device,scatter_size,reset_cache=True):
        
        ### Connectome Encoder

        x = F.dropout(self.x, p=0.6, training=self.training)
        x = F.elu(self.conv1(x, edge_index,edge_attr))
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, edge_index,edge_attr)

        ### Synapses Encoder
        
        x_point=self.synapse_encoder(synapse,synapse_index)
        x_point=x_point.to(device=device)

        ### Output Layer

        left_node=scatter(x_point,edge_index[0],dim_size=scatter_size,reduce='max')
        right_node=scatter(x_point,edge_index[1],dim_size=scatter_size,reduce='max')
        final_node_vector=torch.cat([x,left_node,right_node],dim=1)
        pred =self.classmodel(final_node_vector)
        return  pred

class synapse_former(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.sa1_module = SAModule(0.5, 0.2, MLP([6, 64, 64, 128]))
        self.sa2_module = SAModule_Transformer(0.25, 0.4, MLP([128 , 128, 128, 256]),MLP([256 + 6, 256, 512, 1024]))
        self.mlp = MLP([1024, 512, 512], dropout=0.5, norm=None)

    def forward(self,synapse_cordi,batch_index):
        #synapse_cordi:torch.Size([497282, 6]) ,batch_index:Size([497282,])
        sa0_out = (None, synapse_cordi, batch_index)
        sa1_out = self.sa1_module(*sa0_out)
        sa2_out = self.sa2_module(*sa1_out)
        x, pos, batch = sa2_out
        return self.mlp(x)

class SAModule(torch.nn.Module):
    def __init__(self, ratio, r, nn):
        super().__init__()
        self.ratio = ratio
        self.r = r
        self.conv = PointNetConv(nn, add_self_loops=False)

    def forward(self, x, pos, batch):
        idx = fps(pos, batch, ratio=self.ratio)
        row, col = radius(pos, pos[idx], self.r, batch, batch[idx],
                          max_num_neighbors=64)
        edge_index = torch.stack([col, row], dim=0)
        x_dst = None if x is None else x[idx]
        x = self.conv((x, x_dst), (pos, pos[idx]), edge_index)
        pos, batch = pos[idx], batch[idx]
        return x, pos, batch
        


class SAModule_Transformer(torch.nn.Module):
    def __init__(self, ratio, r, nn1,nn2):
        super().__init__()
        self.ratio = ratio
        self.r = r
        self.hidden_dim=90
        self.nn2 = nn2
        in_channels = nn1.channel_list[0]  
        out_channels = nn1.channel_list[-1]  
        
        self.conv = PointTransformerConv(
            in_channels=in_channels, 
            out_channels=out_channels,  
            pos_nn = MLP([6, 64, out_channels], norm=None, plain_last=False),
            attn_nn=Linear(out_channels, out_channels)  
        )   

    def forward(self, x, pos, batch):
        idx = fps(pos, batch, ratio=self.ratio)
        row, col = radius(pos, pos[idx], self.r, batch, batch[idx], max_num_neighbors=64)
        edge_index = torch.stack([col, row], dim=0)
        x_dst = None if x is None else x[idx]
        x= self.conv(x, pos, edge_index)
        x = self.nn2(torch.cat([x, pos], dim=1))
        x = global_max_pool(x, batch)
        pos = pos.new_zeros((x.size(0), 6))
        batch = torch.arange(x.size(0), device=batch.device)
        return x, pos, batch

