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
from torch_geometric.nn import MLP, PointNetConv,GATConv,PointTransformerConv,PPFConv,XConv
from torch_geometric.utils import scatter
from torch import nn
from src.utils.pointtransformerconv import PointTransformerConv_fix
# from mamba_ssm.models.mixer_seq_simple import Mamba


class Synapformer(torch.nn.Module):
    def __init__(self, num_nodes, dataset_num_features, hidden_channels,represent_features, heads, dataset_num_classes): 
        super().__init__()
        self.conv1 = GATConv(dataset_num_features, hidden_channels, heads, dropout=0.6) 
        self.conv2 = GATConv(hidden_channels * heads, represent_features, heads=1,
                             concat=False, dropout=0.6)
        self.x = nn.Parameter(torch.Tensor(num_nodes, dataset_num_features))
        torch.nn.init.xavier_uniform(self.x, gain=1)
        self.cache_x = None
        
        
        self.synapse_encoder=synapse_former(represent_features)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=3)
        self.classmodel=MLP([represent_features+represent_features+represent_features, 512,dataset_num_classes])


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
        attention_weights = self.synapse_encoder.sa2_module.conv.last_attention_weights
        point_attn = self.synapse_encoder.sa2_module.conv.get_point_attention(reduce='sum') 
        pos_weights = self.synapse_encoder.sa2_module.pos
        x_out= self.synapse_encoder.sa2_module.conv

        return  pred,point_attn,pos_weights
    


class Synapformer2(torch.nn.Module):
    def __init__(self, num_nodes, dataset_num_features, hidden_channels,represent_features, heads, dataset_num_classes): 
        super().__init__()
        self.conv1 = GATConv(dataset_num_features, hidden_channels, heads, dropout=0.6) 
        self.conv2 = GATConv(hidden_channels * heads, represent_features, heads=1,
                             concat=False, dropout=0.6)
        self.x = nn.Parameter(torch.Tensor(num_nodes, dataset_num_features))
        torch.nn.init.xavier_uniform(self.x, gain=1)
        self.cache_x = None
        
        #TODO
        
        self.synapse_encoder=synapse_former2()
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
        return  pred,pred,pred

class Synapformer3(torch.nn.Module):
    def __init__(self, num_nodes, dataset_num_features, hidden_channels,represent_features, heads, dataset_num_classes): 
        super().__init__()
        self.conv1 = GATConv(dataset_num_features, hidden_channels, heads, dropout=0.6) 
        self.conv2 = GATConv(hidden_channels * heads, represent_features, heads=1,
                             concat=False, dropout=0.6)
        self.x = nn.Parameter(torch.Tensor(num_nodes, dataset_num_features))
        torch.nn.init.xavier_uniform(self.x, gain=1)
        self.cache_x = None
        
        #TODO
        
        self.synapse_encoder=synapse_former3()
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
        
class synapse_former2(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.sa1_module = SAModule(0.5, 0.2, MLP([3, 64, 64, 128]))
        self.sa2_module = SAModule_Transformer_3d(0.25, 0.4, MLP([128 , 128, 128, 256]),MLP([256 + 3, 256, 512, 1024]))
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
    def infer(self,x, pos, batch):
        idx=torch.arange(len(pos), device=pos.device)
        row, col = radius(pos, pos[idx], self.r, batch, batch[idx],
                          max_num_neighbors=64)
        edge_index = torch.stack([col, row], dim=0)
        x_dst = None if x is None else x[idx]
        x = self.conv((x, x_dst), (pos, pos[idx]), edge_index)
        #pos, batch = pos[idx], batch[idx]
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
        self.pos=None
        
        self.conv = PointTransformerConv(
            in_channels=in_channels, 
            out_channels=out_channels,  
            pos_nn = MLP([6, 64, out_channels], norm=None, plain_last=False),
            # attn_nn=Linear(out_channels, out_channels)  
            attn_nn=Linear(out_channels, 1)  
        )   

    def forward(self, x, pos, batch):
        self.pos=pos
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
        
    def get_SOE(self,x,pos,batch):
        self.pos=pos
        idx = fps(pos, batch, ratio=self.ratio)
        row, col = radius(pos, pos[idx], self.r, batch, batch[idx], max_num_neighbors=64)
        edge_index = torch.stack([col, row], dim=0)
        x_dst = None if x is None else x[idx]
        x= self.conv(x, pos, edge_index)
        
        return x, pos, batch, idx
    
    def infer(self, x, pos, batch):
        self.pos=pos
        idx = torch.arange(len(pos), device=pos.device)
        row, col = radius(pos, pos[idx], self.r, batch, batch[idx], max_num_neighbors=64)
        edge_index = torch.stack([col, row], dim=0)
        x_dst = None if x is None else x[idx]
        x= self.conv(x, pos, edge_index)
        x = self.nn2(torch.cat([x, pos], dim=1))
       
        return x, pos, batch

class SAModule_Transformer_3d(torch.nn.Module):
    def __init__(self, ratio, r, nn1,nn2):
        super().__init__()
        self.ratio = ratio
        self.r = r
        self.hidden_dim=90
        self.nn2 = nn2
        in_channels = nn1.channel_list[0]  
        out_channels = nn1.channel_list[-1]  
        self.pos=None
        
        self.conv = PointTransformerConv(
            in_channels=in_channels, 
            out_channels=out_channels,  
            pos_nn = MLP([3, 64, out_channels], norm=None, plain_last=False),
            # attn_nn=Linear(out_channels, out_channels)  
            attn_nn=Linear(out_channels, 1)  
        )   

    def forward(self, x, pos, batch):
        self.pos=pos
        idx = fps(pos, batch, ratio=self.ratio)
        row, col = radius(pos, pos[idx], self.r, batch, batch[idx], max_num_neighbors=64)
        edge_index = torch.stack([col, row], dim=0)
        x_dst = None if x is None else x[idx]
        x= self.conv(x, pos, edge_index)
        x = self.nn2(torch.cat([x, pos], dim=1))
        x = global_max_pool(x, batch)
        pos = pos.new_zeros((x.size(0), 3))
        batch = torch.arange(x.size(0), device=batch.device)
       
        return x, pos, batch

class SAModule_PPF(torch.nn.Module):
    def __init__(self, ratio, r, nn):
        super().__init__()
        self.ratio = ratio
        self.r = r
        self.conv = PPFConv(nn, add_self_loops=False)

    def forward(self, x, pos, batch):
        idx = fps(pos, batch, ratio=self.ratio)
        row, col = radius(pos, pos[idx], self.r, batch, batch[idx],
                          max_num_neighbors=64)
        edge_index = torch.stack([col, row], dim=0)
        x_dst = None if x is None else x[idx]
        x = self.conv((x, x_dst), (pos, pos[idx]), edge_index)
        pos, batch = pos[idx], batch[idx]
        return x, pos, batch

class SAModule_Pointcnn(torch.nn.Module):
    def __init__(self, ratio, r, nn):
        super().__init__()
        self.ratio = ratio
        self.r = r
        in_channels = nn[0]  
        out_channels = nn[-1]  

        self.conv = XConv(
            in_channels=in_channels,          # 输入特征维度
            out_channels=out_channels,        # 输出特征维度
            dim=6,                   # 3D点云
            kernel_size=16,          # 每个点聚合16个邻居
            hidden_channels=32,      # 隐藏层维度（默认是64//4=16，此处设为32）
            dilation=1,              # 无膨胀采样
            bias=True,               # 启用偏置
            num_workers=1            # 并行计算邻居
        )

    def forward(self, x, pos, batch):
        idx = fps(pos, batch, ratio=self.ratio)
        row, col = radius(pos, pos[idx], self.r, batch, batch[idx],
                          max_num_neighbors=64)
        edge_index = torch.stack([col, row], dim=0)
        x_dst = None if x is None else x[idx]
        x = self.conv(x_dst, pos[idx], edge_index)
        pos, batch = pos[idx], batch[idx]
        return x, pos, batch

class synapse_former(torch.nn.Module):
    def __init__(self,represent_features):
        super().__init__()

        self.sa1_module = SAModule(0.5, 0.2, MLP([6, 64, 64, 128]))
        self.sa2_module = SAModule_Transformer(0.25, 0.4, MLP([128 , 128, 128, 256]),MLP([256 + 6, 256, 512, 1024]))
        self.mlp = MLP([1024, 1024, represent_features], dropout=0.5, norm=None)

    def forward(self,synapse_cordi,batch_index):
        sa0_out = (None, synapse_cordi, batch_index)
        sa1_out = self.sa1_module(*sa0_out)#0.5
        sa2_out = self.sa2_module(*sa1_out)#0.2
        x, pos, batch = sa2_out
        return self.mlp(x)
    
    def infer(self,synapse_cordi,batch_index):
        sa0_out = (None, synapse_cordi, batch_index)
        sa1_out = self.sa1_module.infer(*sa0_out)#0.5
        sa2_out = self.sa2_module.infer(*sa1_out)#0.2
        return sa2_out
        
    
class synapse_former3(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.sa1_module = SAModule(0.5, 0.2, MLP([6, 64, 64, 128]))
        self.sa2_module = SAModule_Transformer(0.25, 0.4, MLP([128 , 128, 128, 256]),MLP([256 + 6, 256, 512, 1024]))
        self.sa3_module = SAModule_Pointcnn(0.25, 0.4,[1024,1024])
        self.mlp = MLP([1024, 512, 512], dropout=0.5, norm=None)

    def forward(self,synapse_cordi,batch_index):
        #synapse_cordi:torch.Size([497282, 6]) ,batch_index:Size([497282,])
        sa0_out = (None, synapse_cordi, batch_index)
        sa1_out = self.sa1_module(*sa0_out)
        sa2_out = self.sa2_module(*sa1_out)
        sa3_out = self.sa3_module(*sa2_out)
        x, pos, batch = sa3_out
        return self.mlp(x)
    

class MambaSynapseEncoder(nn.Module):
    def __init__(self, input_dim=6, hidden_dim=512):
        super().__init__()
        self.linear_in = nn.Linear(input_dim, hidden_dim)
        self.mamba = Mamba(d_model=hidden_dim)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.linear_out = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, synapse, synapse_index):
        batch_size = synapse_index.max().item() + 1
        sequences = []
        for i in range(batch_size):
            mask = (synapse_index == i)
            seq = synapse[mask]  # [num_points_i, 3]
            if seq.shape[0] == 0:
                seq = torch.zeros(1, synapse.shape[1], device=synapse.device)
            sequences.append(seq)

        outputs = []
        for seq in sequences:
            x = self.linear_in(seq.unsqueeze(0))  # [1, L, d]
            x = self.mamba(x)                     # [1, L, d]
            x = x.transpose(1, 2)                 # [1, d, L]
            pooled = self.pool(x).squeeze()       # [d]
            out = self.linear_out(pooled)         # [d]
            outputs.append(out)

        return torch.stack(outputs, dim=0)  # [batch_size, hidden_dim]

class Synapformer_mamba(torch.nn.Module):
    def __init__(self, num_nodes, dataset_num_features, hidden_channels,represent_features, heads, dataset_num_classes): 
        super().__init__()
        self.conv1 = GATConv(dataset_num_features, hidden_channels, heads, dropout=0.6) 
        self.conv2 = GATConv(hidden_channels * heads, represent_features, heads=1,
                             concat=False, dropout=0.6)
        self.x = nn.Parameter(torch.Tensor(num_nodes, dataset_num_features))
        torch.nn.init.xavier_uniform(self.x, gain=1)
        self.cache_x = None
    
        
        self.synapse_encoder = MambaSynapseEncoder(input_dim=6, hidden_dim=512)
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

class synapformer_transimitter(nn.Module):
    def __init__(self,represent_features, dataset_num_classes): 
        super().__init__()
        
        
        self.synapse_encoder=synapse_former_tran(represent_features)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=3)
        self.classmodel=MLP([represent_features, 512,dataset_num_classes])


    def forward(self,synapse,synapse_index):
        
        ### Synapses Encoder
        
        x_point=self.synapse_encoder(synapse,synapse_index)

        pred=self.classmodel(x_point)
        return  pred
    
class synapse_former_tran(torch.nn.Module):
    def __init__(self,represent_features):
        super().__init__()

        self.sa1_module = SAModule(0.5, 0.2, MLP([3, 64, 64, 128]))
        self.sa2_module = SAModule_Transformer_tran(0.25, 0.4, MLP([128 , 128, 128, 256]),MLP([256 + 3, 256, 512, 1024]))
        self.mlp = MLP([1024, 1024, represent_features], dropout=0.5, norm=None)

    def forward(self,synapse_cordi,batch_index):
        sa0_out = (None, synapse_cordi, batch_index)
        sa1_out = self.sa1_module(*sa0_out)
        sa2_out = self.sa2_module(*sa1_out)
        x, pos, batch = sa2_out
        return self.mlp(x)

class SAModule_Transformer_tran(torch.nn.Module):
    def __init__(self, ratio, r, nn1,nn2):
        super().__init__()
        self.ratio = ratio
        self.r = r
        self.hidden_dim=90
        self.nn2 = nn2
        in_channels = nn1.channel_list[0]  
        out_channels = nn1.channel_list[-1]  
        self.pos=None
        
        self.conv = PointTransformerConv(
            in_channels=in_channels, 
            out_channels=out_channels,  
            pos_nn = MLP([3, 64, out_channels], norm=None, plain_last=False),
            # attn_nn=Linear(out_channels, out_channels)  
            attn_nn=Linear(out_channels, 1)  
        )   

    def forward(self, x, pos, batch):
        self.pos=pos
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
