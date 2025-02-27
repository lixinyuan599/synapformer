import os
import sys
project_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_path not in sys.path:
    sys.path.append(project_path)
import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import GCN2Conv
from torch_geometric.nn import knn,fps, global_max_pool, radius, knn_graph, global_mean_pool
from torch_geometric.nn import MLP, PointNetConv,GATConv,DNAConv ,GraphUNet,DynamicEdgeConv,PointTransformerConv
from torch_geometric.utils import scatter, dropout_edge
from torch import nn
import numpy as np
# from models.point_enco import DGCNN
from other.transformerpoint import PointTransformer
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.aggr import MaxAggregation
from torch_geometric.nn.pool import knn_graph
from torch_geometric.nn.pool.decimation import decimation_indices
from torch_geometric.utils import softmax
from torch import Tensor

class GCNII_Concat(torch.nn.Module):
    def __init__(self, num_nodes, dataset_num_features, dataset_num_classes, represent_features=512,\
                 hidden_channels=64, num_layers=64, alpha=0.1, theta=0.5, shared_weights=True, dropout=0.0):
        super().__init__()

        self.lins = torch.nn.ModuleList()
        self.lins.append(Linear(dataset_num_features, hidden_channels))
        self.mlp = MLP([hidden_channels, (represent_features - hidden_channels) // 3 + hidden_channels,\
                        2*(represent_features - hidden_channels) // 3 + hidden_channels,\
                        represent_features], dropout=0.5)
        self.x = nn.Parameter(torch.Tensor(num_nodes, dataset_num_features))
        torch.nn.init.xavier_uniform(self.x, gain=1)
        self.convs = torch.nn.ModuleList()

        for layer in range(num_layers):
            self.convs.append(GCN2Conv(hidden_channels, alpha, theta, layer + 1, shared_weights, normalize=True))
        self.dropout = dropout
        self.outL = Linear(represent_features, dataset_num_classes)

    def forward(self, edge_index):
        x = self.x.clone().detach()
        x = F.dropout(x, self.dropout, training=self.training)
        x = x_0 = self.lins[0](x).relu() 
        for conv in self.convs:
            x = F.dropout(x, self.dropout, training=self.training)
            x = conv(x, x_0, edge_index )
            x = x.relu()
        if torch.isnan(x).any():
            print('!!!!!!')
        x = self.mlp(x) 
        x = self.outL(x)
        return x
    
class GAT(torch.nn.Module):
    def __init__(self,num_nodes, dataset_num_features, hidden_channels, dataset_num_classes, heads):
        super().__init__()
        self.conv1 = GATConv(dataset_num_features, hidden_channels, heads, dropout=0.6)
        # On the Pubmed dataset, use `heads` output heads in `conv2`.
        self.conv2 = GATConv(hidden_channels * heads, dataset_num_classes, heads=1,
                             concat=False, dropout=0.6)
        self.x = nn.Parameter(torch.Tensor(num_nodes, dataset_num_features))
        torch.nn.init.xavier_uniform(self.x, gain=1)
    def forward(self, edge_index):
        x = F.dropout(self.x, p=0.6, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, edge_index)
        return x
    
class DNANet(torch.nn.Module):
    def __init__(self, num_nodes,dataset_num_features, hidden_channels, dataset_num_classes, num_layers,
                 heads=1, groups=1):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.lin1 = torch.nn.Linear(dataset_num_features, hidden_channels)
        self.convs = torch.nn.ModuleList()
        for i in range(num_layers):
            self.convs.append(
                DNAConv(hidden_channels, heads, groups, dropout=0.8))
        self.lin2 = torch.nn.Linear(hidden_channels, dataset_num_classes)
        self.x = nn.Parameter(torch.Tensor(num_nodes, dataset_num_features))
        torch.nn.init.xavier_uniform(self.x, gain=1)
    def reset_parameters(self):
        self.lin1.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, edge_index):
        x = F.relu(self.lin1(self.x))
        x = F.dropout(x, p=0.5, training=self.training)
        x_all = x.view(-1, 1, self.hidden_channels)
        for conv in self.convs:
            x = F.relu(conv(x_all, edge_index))
            x = x.view(-1, 1, self.hidden_channels)
            x_all = torch.cat([x_all, x], dim=1)
        x = x_all[:, -1]
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        return torch.log_softmax(x, dim=1)
    
class Graph_UNet(torch.nn.Module):
    def __init__(self,num_nodes,dataset_num_features, hidden_channels, dataset_num_classes):
        super().__init__()
        pool_ratios = [2000 / num_nodes, 0.5]
        self.unet = GraphUNet(dataset_num_features, hidden_channels, dataset_num_classes,
                              depth=3, pool_ratios=pool_ratios)
        self.x = nn.Parameter(torch.Tensor(num_nodes, dataset_num_features))
        torch.nn.init.xavier_uniform(self.x, gain=1)
    def forward(self,edge_index):
        dropout_edge_index, _ = dropout_edge(edge_index, p=0.2,
                                     force_undirected=True,
                                     training=self.training)
        x = F.dropout(self.x, p=0.92, training=self.training)

        x = self.unet(x, dropout_edge_index)
        return x
