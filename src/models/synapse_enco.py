import os
import sys
import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import knn,fps, global_max_pool, radius, knn_graph, global_mean_pool
from torch_geometric.utils import scatter
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from torch_geometric.nn import MLP,DynamicEdgeConv,PointNetConv,GCN2Conv
from other.data_dgcnn import load_data

#!/usr/bin/env python
# -*- coding: utf-8 -*-


import os
import sys
import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import knn,fps, global_max_pool, radius, knn_graph, global_mean_pool
from torch_geometric.utils import scatter
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from torch_geometric.nn import MLP,DynamicEdgeConv,PointNetConv,PointTransformerConv
<<<<<<< HEAD
=======
from other.data_dgcnn import load_data
>>>>>>> 73bc381812ce9ae0b6fb772af25a08190c88ea4b

def knn(x, k):
    inner = -2*torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
 
    idx = pairwise_distance.topk(k=k, dim=-1)[1]   # (batch_size, num_points, k) 
    #表示每个点的 k 个最近邻居的索引
    return idx

def distence(x):
    inner = -2*torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
    return pairwise_distance

def vision(x,pic_num,target_point,layer):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    point_clouds,_=load_data('train')
    sample_points = point_clouds[pic_num]
    norm = plt.Normalize(distence(x).min(), distence(x).max())
    colors = cm.Purples(norm(distence(x))) 

    ax.scatter(sample_points[:, 0], sample_points[:, 1], sample_points[:, 2], color=colors,s=5)
    ax.scatter(sample_points[target_point, 0], sample_points[target_point, 1], sample_points[target_point, 2],color='red',s=40)

    cbar = plt.colorbar(cm.ScalarMappable(norm=norm, cmap='Purples'), ax=ax)
    cbar.set_label('Distance to target point')

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    ax.set_title('Point Cloud ')
    plt.savefig(f'/data1/lixinyuan/project/dgcnn/pytorch/pic/{layer}.png')



def get_graph_feature(x, k=20, idx=None):
    if x is None:
        raise ValueError("Input to get_graph_feature is None.")
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points) #(8,3,1024)
    if idx is None:
        idx = knn(x, k=k)   # (batch_size, num_points, k) (8,1024,20) 意味着对于每个批次中的每个点，都有 k 个近邻的索引。
    device = torch.device('cuda')

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1)*num_points
    #生成一个从0到batch_size（不包括batch_size）的一维整数序列。 (batch_size, 1, 1)
    #view(-1, 1, 1)将上述生成的一维序列重新形状为一个三维张量。这里-1让PyTorch自动计算这一维的大小，保持元素总数不变
    #索引基础的设置是为了在处理如点云这样的数据时，将每个批次的数据点映射到一个扁平化的一维数组中，以便可以使用这些索引直接访问或修改数据
    idx = idx + idx_base
    #当 idx 和 idx_base 进行加法操作时，idx_base 会自动扩展（广播）到与 idx 相同的形状 (batch_size, num_points, k)
    #对于每个批次 i，idx_base[i, 0, 0] 中的值（即 (i * num_points)）会被加到 idx 的对应批次的所有点的所有 k 个近邻索引上。
    #这意味着，对于批次中的每个点，我们都将其索引值偏移了一个固定的量，即 (i * num_points)。这样做的目的是将每个批次的索引从相对索引（相对于批次开始的位置）转换为绝对索引（相对于整个扁平化后的数组的位置）。
    idx = idx.view(-1)
    #将 idx 展平成一维数组，以便用于后续索引操作。
    # idx=tensor([[[0]],[[1024]],[[2048]],[[3072]],[[4096]],[[5120]],[[6144]],[[7168]]])
    _, num_dims, _ = x.size()
    # 接收第二维度的大小
    x = x.transpose(2, 1).contiguous()   # (batch_size, num_points, num_dims) （8，1024，3）
    feature = x.view(batch_size*num_points, -1)[idx, :]
    # 将 x 视图调整为 (batch_size*num_points, num_dims)，并使用之前计算的 idx 来索引，选出对应的特征。 
    feature = feature.view(batch_size, num_points, k, num_dims) 
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    #repeat 方法用于沿着指定的各个维度重复张量 x 的内容。在这个命令中，参数 (1, 1, k, 1) 指定了每个维度的重复次数
    #repeat 方法用于沿着指定的各个维度重复张量 x 的内容。在这个命令中，参数 (1, 1, k, 1) 指定了每个维度的重复次数：
    feature = torch.cat((feature-x, x), dim=3).permute(0, 3, 1, 2).contiguous()
    #dim=3 参数指明了拼接的维度。在第4维num_dims进行拼接。 （batch_size，num_dims，num_points, k)
  
    return feature


class PointNet(nn.Module):
    def __init__(self, args, output_channels=40):
        super(PointNet, self).__init__()
        self.args = args
        self.conv1 = nn.Conv1d(3, 64, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
        self.conv3 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
        self.conv4 = nn.Conv1d(64, 128, kernel_size=1, bias=False)
        self.conv5 = nn.Conv1d(128, args.emb_dims, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(64)
        self.bn4 = nn.BatchNorm1d(128)
        self.bn5 = nn.BatchNorm1d(args.emb_dims)
        self.linear1 = nn.Linear(args.emb_dims, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout()
        self.linear2 = nn.Linear(512, output_channels)

    def forward(self, x):#[N,1024,H]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.adaptive_max_pool1d(x, 1).squeeze()
        # PyTorch 中的自适应最大池化操作，用于一维数据。这个函数的目标是将输入的每个通道减少到指定的输出长度（在这个案例中是 1）。意味着无论输入的维度是多少，输出的维度都会被固定。这使得模型能够灵活处理不同长度的输入。
        # 将数据从 [batch_size, num_channels, N] 转换为 [batch_size, num_channels, 1]，那么 .squeeze() 将输出形状为 [batch_size, num_channels]。
        x = F.relu(self.bn6(self.linear1(x)))
        x = self.dp1(x)
        x = self.linear2(x)
        return x
    #[N,1024,H']


class DGCNN(nn.Module):
    def __init__(self, emb_dims, output_channels=40):
        super(DGCNN, self).__init__()
        self.k = 20
        
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm1d(emb_dims)

        self.conv1 = nn.Sequential(nn.Conv2d(6, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2)) #LeakyReLU激活函数。相比于传统的ReLU激活函数，LeakyReLU在输入为负数时，允许有一个小的非零斜率（在这里是0.2），这可以帮助避免神经元完全失活的问题。
        self.conv2 = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1, bias=False), #特征连接或特征融合
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(64*2, 128, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(128*2, 256, kernel_size=1, bias=False),
                                   self.bn4,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(nn.Conv1d(512, emb_dims, kernel_size=1, bias=False),
                                   self.bn5,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.linear1 = nn.Linear(emb_dims*2, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(p=0.5)
        self.linear2 = nn.Linear(512, 256)
        self.bn7 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(p=0.5)
        self.linear3 = nn.Linear(256, output_channels)

    def forward(self, x, batch_synpase_index,device): # x size(8,3,1024)
        x=x.unsqueeze(0)
        x=x.permute(0,2,1)
        batch_size = x.size(0)
        x = get_graph_feature(x, k=self.k) #（8，6，1024，20）
        x = self.conv1(x)
        x1 = x.max(dim=-1, keepdim=False)[0].to(device) #（[8，64，1024]）
        # keepdim=False 决定了操作后是否保留原始数据的维度。False 表示不保持，即结果会降低一个维度。这使得输出的维度少于输入的维度
        # x.max() 实际上返回一个元组，其中包含两个元素：第一个元素是最大值，第二个元素是这些最大值的索引（即位置）


        x = get_graph_feature(x1, k=self.k) # （[8，128，1024，20]）
        x = self.conv2(x)
        x2 = x.max(dim=-1, keepdim=False)[0].to(device) # ([8,64,1024])

        x = get_graph_feature(x2, k=self.k) #（[8，128，1024，20]）
        x = self.conv3(x) #（[8，128，1024，20]）
        x3 = x.max(dim=-1, keepdim=False)[0].to(device) # （[8,128,1024]）

        x = get_graph_feature(x3, k=self.k) #（[8，256，1024，20]）
        x = self.conv4(x)
        x4 = x.max(dim=-1, keepdim=False)[0].to(device) #([8,256,1024])

        x = torch.cat((x1, x2, x3, x4), dim=1) #([8,512,1024])

        x = self.conv5(x) #([8,1024,1024])
        x1 = F.adaptive_max_pool1d(x, 1).view(batch_size, -1) # ([8,1024])
        #自适应最大池化，第一个参数 x 是输入张量，第二个参数 1 表示池化操作的输出长度。
        x2 = F.adaptive_avg_pool1d(x, 1).view(batch_size, -1) # ([8,1024])
        x = torch.cat((x1, x2), 1) #([8,2048])
        #自适应平均池化
        x = F.leaky_relu(self.bn6(self.linear1(x)), negative_slope=0.2)  # ([8,512])
        #Leaky ReLU 允许负输入值有一个小的非零输出（即负斜率 negative_slope），在这里设置为 0.2。这样可以保持负梯度，使得在激活输出为负时神经元依旧有梯度传递，提高了模型的训练效率和性能。
        x = self.dp1(x) # ([8,512])
        x = F.leaky_relu(self.bn7(self.linear2(x)), negative_slope=0.2) # ([8,256])
        x = self.dp2(x) # ([8,256])
        x = self.linear3(x)# ([8,40])

        x=x.to(device)
        return x
    
class SAModule(torch.nn.Module):
    def __init__(self, ratio, r, nn):
        super().__init__()
        self.ratio = ratio
        self.r = r
        self.conv = PointNetConv(nn, add_self_loops=False)

    def forward(self, x, pos, batch):
        # torch.arrange(pos.size(0)//2000)
        #[N,6]
        idx = fps(pos, batch, ratio=self.ratio)
        row, col = radius(pos, pos[idx], self.r, batch, batch[idx],
                          max_num_neighbors=64)
        edge_index = torch.stack([col, row], dim=0)
        x_dst = None if x is None else x[idx]
        x = self.conv((x, x_dst), (pos, pos[idx]), edge_index)
        pos, batch = pos[idx], batch[idx]
        return x, pos, batch
       
class GlobalSAModule(torch.nn.Module):
    def __init__(self, nn):
        super().__init__()
        self.nn = nn

    def forward(self, x, pos, batch):
        x = self.nn(torch.cat([x, pos], dim=1))
        x = global_max_pool(x, batch)
        pos = pos.new_zeros((x.size(0), 6))
        batch = torch.arange(x.size(0), device=batch.device)
        return x, pos, batch

class GlobalSAModule_pointnet3(torch.nn.Module):
    def __init__(self, nn):
        super().__init__()
        self.nn = nn

    def forward(self, pos, batch):
        x = self.nn(pos)
        x = global_max_pool(pos, batch)
        pos = pos.new_zeros((x.size(0), 6))
        batch = torch.arange(x.size(0), device=batch.device)
        return x, pos, batch
    
class pointnet1(nn.Module):
    def __init__(self,input_channels,emb_dims, output_channels):
        super().__init__()
        # self.args = args
        self.conv1 = nn.Conv1d(input_channels, 64, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
        self.conv3 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
        self.conv4 = nn.Conv1d(64, 128, kernel_size=1, bias=False)
        self.conv5 = nn.Conv1d(128, emb_dims, kernel_size=1, bias=False)
        # self.bn1 = nn.BatchNorm1d(64)
        # self.bn2 = nn.BatchNorm1d(64)
        # self.bn3 = nn.BatchNorm1d(64)
        # self.bn4 = nn.BatchNorm1d(128)
        # self.bn5 = nn.BatchNorm1d(emb_dims)
        self.linear1 = nn.Linear(emb_dims, 512, bias=False)
        # self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout()
        self.linear2 = nn.Linear(512, output_channels)

    def forward(self, x):#[N,1024,H]
        # x = F.relu(self.bn1(self.conv1(x)))
        # x = F.relu(self.bn2(self.conv2(x)))
        # x = F.relu(self.bn3(self.conv3(x)))
        # x = F.relu(self.bn4(self.conv4(x)))
        # x = F.relu(self.bn5(self.conv5(x)))
        # x = x.permute(0,2,1)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))

        x = F.adaptive_max_pool1d(x, 1).squeeze()
        # PyTorch 中的自适应最大池化操作，用于一维数据。这个函数的目标是将输入的每个通道减少到指定的输出长度（在这个案例中是 1）。意味着无论输入的维度是多少，输出的维度都会被固定。这使得模型能够灵活处理不同长度的输入。
        # 将数据从 [batch_size, num_channels, N] 转换为 [batch_size, num_channels, 1]，那么 .squeeze() 将输出形状为 [batch_size, num_channels]。
        # x = F.relu(self.bn6(self.linear1(x)))
        x = F.relu(self.linear1(x))
        x = self.dp1(x)
        x = self.linear2(x)
        return x
    #[N,1024,H']
       
class PointNet3(torch.nn.Module):
    def __init__(self):
        super(PointNet3,self).__init__()

        # Input channels account for both `pos` and node features.
        #input:[N,1,6], output:[N,1,256]
        self.sa1_module = pointnet1(input_channels=6 ,emb_dims=256, output_channels=256)
        #input:[N,1,256], output:[N,1,512]
        self.sa2_module = pointnet1(input_channels=256,emb_dims=256, output_channels=512)
        # self.sa1_module = SAModule(1, 0.2, MLP([6, 64, 64, 128]))
        # self.sa2_module = SAModule(1, 0.4, MLP([128 + 6, 128, 128, 256]))
        #input:([N,512],batch) output:[|E|,512]
        self.sa3_module = GlobalSAModule_pointnet3(MLP([256 , 256, 512, 1024]))
        # self.mlp1 = MLP([128, 512, 512], dropout=0.5, norm=None)
        # self.mlp2= MLP([256, 512, 512], dropout=0.5, norm=None)
        self.mlp = MLP([256, 512, 512], dropout=0.5, norm=None)

    def forward(self,synapse_cordi,batch_index,device):
        #synapse_cordi:torch.Size([497282, 6]) ,batch_index:Size([497282,])
        synapse_cordi=synapse_cordi.unsqueeze(1).to(device)
        synapse_cordi = synapse_cordi.permute(0,2,1)
        sa1_out = self.sa1_module(synapse_cordi)
        # sa2_out = self.sa2_module(sa1_out)
        sa3_out = self.sa3_module(sa1_out,batch_index)
        x, pos, batch = sa3_out
        return self.mlp(x)
    
    def infer_x(self,synapse_cordi,batch_index):
        sa0_out = (None, synapse_cordi, batch_index)
        sa1_out = self.sa1_module(*sa0_out)
        sa2_out = self.sa2_module(*sa1_out)
        sa3_out = self.sa3_module(*sa2_out)
        x1,_,_ = sa1_out
        x2,_,_ = sa2_out
        x3,_,_ = sa3_out
        # x1=self.mlp1(x1)
        # x2=self.mlp2(x2)
        x3=self.mlp(x3)
        return x1,x2,x3

class DGCNN_PYG(torch.nn.Module):
    def __init__(self, out_channels=512, k=5, aggr='max'):
        super().__init__()

        self.conv1 = DynamicEdgeConv(MLP([2 * 6, 64, 64, 64]), k, aggr)
        self.conv2 = DynamicEdgeConv(MLP([2 * 64, 128]), k, aggr)
        self.lin1 = Linear(128 + 64, 1024)

        self.mlp = MLP([1024, 512, 256, out_channels], dropout=0.5, norm=None)

    def forward(self, pos, batch, device):
        # pos, batch = data.pos, data.batch
        total_size=pos.size(0)
        batch_labels=torch.arange(total_size//1000)
        batches = batch_labels.repeat_interleave(1000)
        remaining_size = total_size - batches.size(0)
        if remaining_size > 0:
            batches = torch.cat([batches, torch.full((remaining_size,), batch_labels[-1]+1)]).to(device)
        pos = pos.to(device)
        x1 = self.conv1(pos, batches).to(device)
        x2 = self.conv2(x1, batches).to(device)
        out = self.lin1(torch.cat([x1, x2], dim=1)).to(device)
        out = global_max_pool(out, batch).to(device)
        out = self.mlp(out).to(device)
        out = F.log_softmax(out, dim=1).to(device) 

        return out


       
class PointNet2(torch.nn.Module):
    def __init__(self):
        super().__init__()

        # Input channels account for both `pos` and node features.
        self.sa1_module = SAModule(0.5, 0.2, MLP([6, 64, 64, 128]))
        self.sa2_module = SAModule(0.25, 0.4, MLP([128 + 6, 128, 128, 256]))
        # self.sa1_module = SAModule(1, 0.2, MLP([6, 64, 64, 128]))
        # self.sa2_module = SAModule(1, 0.4, MLP([128 + 6, 128, 128, 256]))

        self.sa3_module = GlobalSAModule(MLP([256 + 6, 256, 512, 1024]))
        # self.mlp1 = MLP([128, 512, 512], dropout=0.5, norm=None)
        # self.mlp2= MLP([256, 512, 512], dropout=0.5, norm=None)
        self.mlp = MLP([1024, 512, 512], dropout=0.5, norm=None)

    def forward(self,synapse_cordi,batch_index):
        #synapse_cordi:torch.Size([497282, 6]) ,batch_index:Size([497282,])
        sa0_out = (None, synapse_cordi, batch_index)
        sa1_out = self.sa1_module(*sa0_out)
        sa2_out = self.sa2_module(*sa1_out)
        sa3_out = self.sa3_module(*sa2_out)
        x, pos, batch = sa3_out
        return self.mlp(x)

class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_model):
        super(ScaledDotProductAttention, self).__init__()
        self.d_model = d_model

    def forward(self, query, key, value):
        # 计算 Q 和 K 的点积
        scores = torch.matmul(query, key.transpose(-2, -1)) / self.d_model**0.5  # 缩放

        # 计算 attention 权重
        attention_weights = F.softmax(scores, dim=-1)

        # 加权求和 V
        output = torch.matmul(attention_weights, value)

        return output, attention_weights

class TransformerBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.lin_in = Linear(in_channels, in_channels)
        self.lin_out = Linear(out_channels, out_channels)

        self.pos_nn = MLP([6, 64, out_channels], norm=None, plain_last=False)

        self.attn_nn = MLP([out_channels, 64, out_channels], norm=None,
                           plain_last=False)
        self.attn = ScaledDotProductAttention(d_model=out_channels)

        self.transformer = PointTransformerConv(in_channels, out_channels,
                                                pos_nn=self.pos_nn,
                                                attn_nn=self.attn_nn)

    def forward(self, x, pos, edge_index):
        residual = x.clone().detach()
        x = self.lin_in(x).relu()
        x = self.transformer(x, pos, edge_index)
        x = self.lin_out(x).relu()
        if residual.shape != x.shape:
            residual = self.lin_in(residual)
        x = x + residual
        return x
    




class TransitionDown(torch.nn.Module):
    """Samples the input point cloud by a ratio percentage to reduce
    cardinality and uses an mlp to augment features dimensionnality.
    """
    def __init__(self, in_channels, out_channels, ratio=0.25, k=16):
        super().__init__()
        self.k = k
        self.ratio = ratio
        self.mlp = MLP([in_channels, out_channels], plain_last=False)
        self.batch_norm = torch.nn.BatchNorm1d(out_channels)  # 批归一化
        self.activation = torch.nn.LeakyReLU(negative_slope=0.01)

    def forward(self, x, pos, batch):
        
        # FPS sampling
        id_clusters = fps(pos, ratio=self.ratio, batch=batch)

        # compute for each cluster the k nearest points
        sub_batch = batch[id_clusters] if batch is not None else None

        # beware of self loop
        id_k_neighbor = knn(pos, pos[id_clusters], k=self.k, batch_x=batch,
                            batch_y=sub_batch)

        # transformation of features through a simple MLP
        x = self.mlp(x)

        x = self.batch_norm(x)
        x = self.activation(x)

        # Max pool onto each cluster the features from knn in points
        x_out = scatter(x[id_k_neighbor[1]], id_k_neighbor[0], dim=0,
                        dim_size=id_clusters.size(0), reduce='max')
        
        
        
        out = x_out
        # keep only the clusters and their max-pooled features
        sub_pos = pos[id_clusters]
        return out, sub_pos, sub_batch


class Point_Transformer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, dim_model, k=16):
        super().__init__()
        self.k = k

        # dummy feature is created if there is none given
        in_channels = max(in_channels, 1)

        # first block
        self.mlp_input = MLP([in_channels, dim_model[0]], plain_last=False)

        self.transformer_input = TransformerBlock(in_channels=dim_model[0],
                                                  out_channels=dim_model[0])
        # backbone layers
        self.transformers_down = torch.nn.ModuleList()
        self.transition_down = torch.nn.ModuleList()

        for i in range(len(dim_model) - 1):
            # Add Transition Down block followed by a Transformer block
            self.transition_down.append(
                TransitionDown(in_channels=dim_model[i],
                               out_channels=dim_model[i + 1], k=self.k))

            self.transformers_down.append(
                TransformerBlock(in_channels=dim_model[i + 1],
                                 out_channels=dim_model[i + 1]))

        # class score computation
        self.mlp_output = MLP([dim_model[-1], 64, out_channels], norm=None)

    def forward(self, pos, batch):
        x = torch.ones((pos.shape[0], 6), device=pos.get_device())

        # first block
        x = self.mlp_input(x)
        edge_index = knn_graph(pos, k=self.k, batch=batch)
        x = self.transformer_input(x, pos, edge_index)

        # backbone
        for i in range(len(self.transformers_down)):
            
            x, pos, batch = self.transition_down[i](x, pos, batch=batch)

            edge_index = knn_graph(pos, k=self.k, batch=batch)
            x = self.transformers_down[i](x, pos, edge_index)
            

        # GlobalAveragePooling
        x = global_mean_pool(x, batch)

        # Class score
        out = self.mlp_output(x)

        return out
    
