import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import GCN2Conv
from torch_geometric.nn import knn,fps, global_max_pool, radius, knn_graph, global_mean_pool
from torch_geometric.nn import MLP, PointNetConv,GATConv,DNAConv ,GraphUNet,DynamicEdgeConv,PointTransformerConv
from torch_geometric.utils import scatter, dropout_edge
from torch import nn
import numpy as np


class BaseLearner(nn.Module):
    
    def __init__(self, connectome_enco:torch.nn.modules,synapse_enco:torch.nn.modules,hidden=256): 
        super().__init__()
        self.connectome_encoder=connectome_enco
        self.cache_x = None
        self.synapse_encoder=synapse_enco
        self.classmodel = MLP
    def forward(self,edge_index,edge_attr,synapse,synapse_index,device,scatter_size):
        if self.connectome_encoder is not None:
            node_rep=self.connectome_enco(edge_index,edge_attr)
        
        if self.synapse_encoder is not None:
            x_point=self.synapse_encoder(synapse,synapse_index)
        
        if self.synapse_encoder is not None and self.connectome_encoder is not None:
            left_node=scatter(x_point,edge_index[0],dim_size=scatter_size,reduce='max')
            right_node=scatter(x_point,edge_index[1],dim_size=scatter_size,reduce='max')
            final_node_vector=torch.cat([node_rep,left_node,right_node],dim=1)
        
        elif self.synapse_encoder is not None and self.connectome_encoder is None:
            pass
        elif self.synapse_encoder is None and self.connectome_encoder is not None:
            pass
        
        pred =self.classmodel(final_node_vector)
        return  pred
    

