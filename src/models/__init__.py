import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import GCN2Conv
from torch_geometric.nn import knn,fps, global_max_pool, radius, knn_graph, global_mean_pool
from torch_geometric.nn import MLP, PointNetConv,GATConv,DNAConv ,GraphUNet,DynamicEdgeConv,PointTransformerConv
from torch_geometric.utils import scatter, dropout_edge
from torch import nn
import numpy as np

class PointEmbed(nn.Module):

    def __init__(self, hidden_dim=90, dim=128):
        super().__init__()

        assert hidden_dim % 6 == 0

        self.embedding_dim = hidden_dim
        e = torch.pow(2, torch.arange(self.embedding_dim // 6)).float() * np.pi
        e = torch.stack([
            torch.cat([e, torch.zeros(self.embedding_dim // 6),
                       torch.zeros(self.embedding_dim // 6)]),
            torch.cat([torch.zeros(self.embedding_dim // 6), e,
                        torch.zeros(self.embedding_dim // 6)]),
            torch.cat([torch.zeros(self.embedding_dim // 6),
                        torch.zeros(self.embedding_dim // 6), e]),
            torch.cat([e, torch.zeros(self.embedding_dim // 6),
                       torch.zeros(self.embedding_dim // 6)]),
            torch.cat([torch.zeros(self.embedding_dim // 6), e,
                        torch.zeros(self.embedding_dim // 6)]),
            torch.cat([torch.zeros(self.embedding_dim // 6),
                        torch.zeros(self.embedding_dim // 6), e])
        ])
        self.register_buffer('basis', e) 

        self.mlp = nn.Linear(self.embedding_dim+6, dim)

    @staticmethod
    def embed(input, basis):
        projections = torch.einsum(
            'bd,de->be', input, basis)
        embeddings = torch.cat([projections.sin(), projections.cos()], dim=1)
        return embeddings

    def forward(self, input):

        pe = self.embed(input, self.basis)
        embed = self.mlp(torch.cat([pe, input], dim=1))  
        return embed
    


class SAModule_Transformer(torch.nn.Module):
    def __init__(self, ratio, r, nn):
        super().__init__()
        self.ratio = ratio
        self.r = r
        self.hidden_dim=90

        # Ensure nn is a list or tuple containing the input and output channels
        in_channels = nn.channel_list[0]  # Use nn[0] as input channels
        out_channels = nn.channel_list[-1]  # Use nn[-1] as output channels
        
        # Initialize the PointTransformerConv layer with input and output channels
        self.conv = PointTransformerConv(
            in_channels=in_channels,  # input feature channels
            out_channels=out_channels,  # output feature channels
            # pos_nn=Linear(6, out_channels),  # position network to map positional information
            # pos_nn = PointEmbed(self.hidden_dim, out_channels),
            pos_nn = MLP([6, 64, out_channels], norm=None, plain_last=False),
            attn_nn=Linear(out_channels, out_channels)  # attention network
        )   

    def forward(self, x, pos, batch):
        # FPS downsampling: choose the key points
        idx = fps(pos, batch, ratio=self.ratio)
        
        # Get neighbors based on radius: to form edge_index
        row, col = radius(pos, pos[idx], self.r, batch, batch[idx], max_num_neighbors=64)
        
        # Construct the edge_index
        edge_index = torch.stack([col, row], dim=0)
        
        # If x is None, use only the positions for processing
        x_dst = None if x is None else x[idx]
        
        # Pass through the PointTransformerConv layer
        x= self.conv(x, pos, edge_index)
        
        # Return the updated features, positions, and batch indices
        return x, pos, batch


class PointNet2Transformer(torch.nn.Module):
    def __init__(self):
        super().__init__()

        # Input channels account for both `pos` and node features.
        self.sa1_module = SAModule(0.5, 0.2, MLP([6, 64, 64, 128]))

        # self.sa2_module = SAModule_Transformer(0.25, 0.4, MLP([128 , 128, 128, 256]))
        self.sa2_module = SAModule_Transformer(0.25, 0.4, MLP([128 , 128, 128, 256]))


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

class SynapseNet_GAT_PointNet2Transformer_with_attr(torch.nn.Module):
    def __init__(self, num_nodes, dataset_num_features, hidden_channels,represent_features, heads, dataset_num_classes): 
        super().__init__()
        # self.conv1 = GATConv(dataset_num_features, hidden_channels, heads, dropout=0.6)
        # # On the Pubmed dataset, use `heads` output heads in `conv2`.
        # self.conv2 = GATConv(hidden_channels * heads, represent_features, heads=1,
        #                      concat=False, dropout=0.6)
        self.conv1 = GATConv(dataset_num_features, hidden_channels, heads, dropout=0.6)  # edge_dim is the size of edge features
        self.conv2 = GATConv(hidden_channels * heads, represent_features, heads=1,
                             concat=False, dropout=0.6)
        self.x = nn.Parameter(torch.Tensor(num_nodes, dataset_num_features))
        torch.nn.init.xavier_uniform(self.x, gain=1)
        self.cache_x = None
        
        #TODO
        
        self.synapse_encoder=PointNet2Transformer()
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=3)
        self.classmodel_Point=MLP([1024,256, dataset_num_classes])
        self.classmodel_GAT=MLP([512,256, dataset_num_classes])
        self.classmodel=MLP([represent_features+512+512, 512,dataset_num_classes])
        # self.classmodel=MLP([512+512, 512,dataset_num_classes])


    def forward(self,edge_index,edge_attr,synapse,synapse_index,device,scatter_size,reset_cache=True):
        
        ### connectome encoder
        if self.cache_x is None and reset_cache:
            x = F.dropout(self.x, p=0.6, training=self.training)
            x = F.elu(self.conv1(x, edge_index,edge_attr))
            x = F.dropout(x, p=0.6, training=self.training)
            x = self.conv2(x, edge_index,edge_attr)
            self.cache_x=x

        ### synapse encoder
        
        x_point=self.synapse_encoder(synapse,synapse_index)
        
        x_point=x_point.to(device=device)
    
    
        left_node=scatter(x_point,edge_index[0],dim_size=scatter_size,reduce='mean')
        right_node=scatter(x_point,edge_index[1],dim_size=scatter_size,reduce='mean')

        final_node_vector=torch.cat([self.cache_x,left_node,right_node],dim=1)
        # final_node_vector=torch.cat([left_node,right_node],dim=1)

        # x_point = torch.cat([left_node,right_node],dim=1)
        #x_point = torch.stack([x, left_node, right_node], dim=0)
        #x:torch.Size([21739, 512]), left_node:torch.Size([21739, 512]), right_node:torch.Size([21739, 512])
        #final_node_vector, _ = torch.max(x_point, dim=0)
        #final_node_vector=torch.cat([x,left_node,right_node],dim=1)
        # GATpred=self.classmodel_GAT(x)
        # Point_pred=self.classmodel_Point(x_point)
        
        # lambda1=0.45
        # pred=lambda1*GATpred+(1-lambda1)*Point_pred
        pred =self.classmodel(final_node_vector)
        return  pred
    
    @torch.no_grad
    def inference(self,edge_index,edge_attr,synapse,synapse_index,device,scatter_size,data,current_test_index,reset_cache=True):
        if self.cache_x is None and reset_cache:
            x = F.dropout(self.x, p=0.6, training=self.training)
            x = F.elu(self.conv1(x, edge_index,edge_attr))
            x = F.dropout(x, p=0.6, training=self.training)
            x = self.conv2(x, edge_index,edge_attr)
            self.cache_x=x
        
        x_point=self.synapse_encoder(synapse,synapse_index)
        
        x_point=x_point.to(device=device)
        #[|Test|,d]
        
        #[|Train|+|Test|+,]
    
        # left_node=scatter(x_point,edge_index[0],dim_size=scatter_size,reduce='mean')
        # right_node=scatter(x_point,edge_index[1],dim_size=scatter_size,reduce='mean')

        # final_node_vector=torch.cat([self.cache_x[current_test_index],x_point,x_point],dim=1)
        final_node_vector=torch.cat([x_point,x_point],dim=1)
        pred =self.classmodel(final_node_vector)
        return  pred

class Sample_Transformer(torch.nn.Module):
    def __init__(self, ratio, r, nn):
        super().__init__()
        self.ratio = ratio
        self.r = r
        self.hidden_dim=90
        self.nn_mlp = MLP([6,32,32])

        # Ensure nn is a list or tuple containing the input and output channels
        in_channels = nn.channel_list[0]  # Use nn[0] as input channels
        out_channels = nn.channel_list[-1]  # Use nn[-1] as output channels
        
        # Initialize the PointTransformerConv layer with input and output channels
        self.conv = PointTransformerConv(
            in_channels=in_channels,  # input feature channels
            out_channels=out_channels,  # output feature channels
            # pos_nn=Linear(6, out_channels),  # position network to map positional information
            # pos_nn = PointEmbed(self.hidden_dim, out_channels),
            pos_nn = MLP([6, 64, out_channels], norm=None, plain_last=False),
            attn_nn=Linear(out_channels, out_channels)  # attention network
        )   

    def forward(self, x, pos, batch):
        if x == None:
            x = self.nn_mlp(pos)
        # FPS downsampling: choose the key points
        idx = fps(pos, batch, ratio=self.ratio)
        
        # Get neighbors based on radius: to form edge_index
        row, col = radius(pos, pos[idx], self.r, batch, batch[idx], max_num_neighbors=64)
        
        # Construct the edge_index
        edge_index = torch.stack([col, row], dim=0)
        
        # If x is None, use only the positions for processing
        x_dst = None if x is None else x[idx]
        
        # Pass through the PointTransformerConv layer
        x= self.conv(x, pos, edge_index)
        # x= self.conv(x[idx], pos[idx], edge_index)
        
        
        # Return the updated features, positions, and batch indices
        return x, pos, batch



class Synapse_Transformer(torch.nn.Module):
    def __init__(self):
        super().__init__()

        # self.sa1_module = SAModule(0.5, 0.2, MLP([6, 64, 64, 128]))
        self.sa1_module = Sample_Transformer(0.5, 0.2, MLP([32, 64, 64, 128]))

        # self.sa2_module = SAModule_Transformer(0.25, 0.4, MLP([128 , 128, 128, 256]))
        self.sa2_module = Sample_Transformer(0.25, 0.4, MLP([128 , 128, 128, 256]))

        self.sa3_module = GlobalSAModule(MLP([256 + 6, 256, 512, 1024]))
        self.mlp = MLP([1024, 512, 512], dropout=0.5, norm=None)

    def forward(self,synapse_cordi,batch_index):
        #synapse_cordi:torch.Size([497282, 6]) ,batch_index:Size([497282,])
        sa0_out = (None, synapse_cordi, batch_index)
        sa1_out = self.sa1_module(*sa0_out)
        sa2_out = self.sa2_module(*sa1_out)
        sa3_out = self.sa3_module(*sa2_out)
        x, pos, batch = sa3_out
        return self.mlp(x)
               
class SynapseNet_GAT_Transformer_with_attr(torch.nn.Module):
    def __init__(self, num_nodes, dataset_num_features, hidden_channels,represent_features, heads, dataset_num_classes): 
        super().__init__()
        # self.conv1 = GATConv(dataset_num_features, hidden_channels, heads, dropout=0.6)
        # # On the Pubmed dataset, use `heads` output heads in `conv2`.
        # self.conv2 = GATConv(hidden_channels * heads, represent_features, heads=1,
        #                      concat=False, dropout=0.6)
        self.conv1 = GATConv(dataset_num_features, hidden_channels, heads, dropout=0.6)  # edge_dim is the size of edge features
        self.conv2 = GATConv(hidden_channels * heads, represent_features, heads=1,
                             concat=False, dropout=0.6)
        self.x = nn.Parameter(torch.Tensor(num_nodes, dataset_num_features))
        torch.nn.init.xavier_uniform(self.x, gain=1)
        self.cache_x = None
        
        #TODO
        
        self.synapse_encoder=Synapse_Transformer()
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=3)
        self.classmodel_Point=MLP([1024,256, dataset_num_classes])
        self.classmodel_GAT=MLP([512,256, dataset_num_classes])
        self.classmodel=MLP([represent_features+512+512, 512,dataset_num_classes])


    def forward(self,edge_index,edge_attr,synapse,synapse_index,device,scatter_size):
        
        ### connectome encoder
        x = F.dropout(self.x, p=0.6, training=self.training)
        x = F.elu(self.conv1(x, edge_index,edge_attr))
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, edge_index,edge_attr)


        ### synapse encoder
        
        x_point=self.synapse_encoder(synapse,synapse_index)
        
        x_point=x_point.to(device=device)
    
    
        left_node=scatter(x_point,edge_index[0],dim_size=scatter_size,reduce='mean')
        right_node=scatter(x_point,edge_index[1],dim_size=scatter_size,reduce='mean')
        final_node_vector=torch.cat([x,left_node,right_node],dim=1)

        pred =self.classmodel(final_node_vector)
        return  pred
    
    @torch.no_grad
    def inference(self,edge_index,edge_attr,synapse,synapse_index,device,scatter_size,data,current_test_index,reset_cache=True):
        if self.cache_x is None and reset_cache:
            x = F.dropout(self.x, p=0.6, training=self.training)
            x = F.elu(self.conv1(x, edge_index,edge_attr))
            x = F.dropout(x, p=0.6, training=self.training)
            x = self.conv2(x, edge_index,edge_attr)
            self.cache_x=x
        
        x_point=self.synapse_encoder(synapse,synapse_index)
        
        x_point=x_point.to(device=device)
        #[|Test|,d]
        
        #[|Train|+|Test|+,]
    
        # left_node=scatter(x_point,edge_index[0],dim_size=scatter_size,reduce='mean')
        # right_node=scatter(x_point,edge_index[1],dim_size=scatter_size,reduce='mean')
        final_node_vector=torch.cat([self.cache_x[current_test_index],x_point,x_point],dim=1)

        pred =self.classmodel(final_node_vector)
        return  pred


class Weight_Sample_Module(torch.nn.Module):
    def __init__(self, ratio, r, nn):
        super().__init__()
        self.ratio = ratio
        self.r = r
        self.conv = PointNetConv(nn, add_self_loops=False)

    def forward(self, x, pos, batch,weight):
        # torch.arrange(pos.size(0)//2000)
        #[N,6]
        idx = fps(pos, batch, ratio=self.ratio)
        row, col = radius(pos, pos[idx], self.r, batch, batch[idx],
                          max_num_neighbors=64)
        edge_index = torch.stack([col, row], dim=0)
        x_dst = None if x is None else x[idx]
        x = self.conv((x, x_dst), (pos, pos[idx]), edge_index)
        pos, batch, weight = pos[idx], batch[idx], weight[idx]
        return x, pos, batch, weight
    
class Weight_PointNet2(torch.nn.Module):
    def __init__(self):
        super().__init__()

        # Input channels account for both `pos` and node features.
        self.sa1_module = Weight_Sample_Module(0.5, 0.2, MLP([6, 64, 64, 128]))
        self.sa2_module = Weight_Sample_Module(0.25, 0.4, MLP([128 + 6, 128, 128, 256]))
        # self.sa1_module = SAModule(1, 0.2, MLP([6, 64, 64, 128]))
        # self.sa2_module = SAModule(1, 0.4, MLP([128 + 6, 128, 128, 256]))

        self.sa3_module = GlobalSAModule(MLP([256 + 6, 256, 512, 1024]))
        # self.mlp1 = MLP([128, 512, 512], dropout=0.5, norm=None)
        # self.mlp2= MLP([256, 512, 512], dropout=0.5, norm=None)
        self.mlp = MLP([1024, 512, 512], dropout=0.5, norm=None)

    def forward(self,synapse_cordi,batch_index,weight):
        #synapse_cordi:torch.Size([497282, 6]) ,batch_index:Size([497282,])
        sa0_out = (None, synapse_cordi, batch_index,weight)
        sa1_out = self.sa1_module(*sa0_out)
        sa2_out = self.sa2_module(*sa1_out)
        sa3_out = self.sa3_module(*sa2_out[:3])
        x, pos, batch = sa3_out
        return self.mlp(x)
    
class SAModule(torch.nn.Module):
    def __init__(self, ratio, r, nn):
        super().__init__()
        self.ratio = ratio# ratio
        self.r = r
        self.conv = PointNetConv(nn, add_self_loops=False)

    def forward(self, x, pos, batch):
        # torch.arrange(pos.size(0)//2000)
        #[N,6]
        idx = fps(pos, batch, ratio=self.ratio) # self.ratio
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
    

        
class synapse(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.sa1_module = SAModule(0.5, 0.2, MLP([6, 64, 64, 128]))
        self.sa2_module = SAModule(0.25, 0.4, MLP([128 + 6, 128, 128, 256]))
        self.sa3_module = GlobalSAModule(MLP([256 + 6, 256, 512, 1024]))

        self.mlp = MLP([1024, 512, 512,256,133], dropout=0.5, norm=None)

    def forward(self,synapse_cordi,batch_index):
        #synapse_cordi:torch.Size([497282, 6]) ,batch_index:Size([497282,])
        sa0_out = (None, synapse_cordi, batch_index)
        sa1_out = self.sa1_module(*sa0_out)
        sa2_out = self.sa2_module(*sa1_out)
        sa3_out = self.sa3_module(*sa2_out)
        x, pos, batch = sa3_out
        return self.mlp(x)


class SynapseNet_GCN_PointNet(torch.nn.Module):
    def __init__(self, num_nodes, dataset_num_features, num_calss, represent_features=512,\
                 hidden_channels=64, num_layers=64, alpha=0.1, theta=0.5, shared_weights=True, dropout=0.0): 
        super().__init__()

        self.num_layers = num_layers
        #self.x=torch.nn.Parameter(torch.Tensor(21739,256))
        super().__init__()

        self.lins = torch.nn.ModuleList()
        self.lins.append(Linear(dataset_num_features, hidden_channels))
        self.mlp = MLP([hidden_channels, (represent_features - hidden_channels) // 3 + hidden_channels,\
                        2*(represent_features - hidden_channels) // 3 + hidden_channels,\
                        represent_features], dropout=0.5)
        self.x = torch.nn.Parameter(torch.Tensor(num_nodes, dataset_num_features))
        torch.nn.init.xavier_uniform(self.x, gain=1)
        self.convs = torch.nn.ModuleList()
        
        for layer in range(num_layers):
            self.convs.append(GCN2Conv(hidden_channels, alpha, theta, layer + 1, shared_weights, normalize=True))
        self.dropout = dropout
        self.outL = Linear(represent_features, num_calss)

        #TODO
        
        self.synapse_encoder=PointNet2()
        self.classmodel=MLP([1536,512,num_calss])


    def reset_parameters(self): #重置模型中所有层的参数，以便重新训练或初始化
        for conv in self.convs:
            conv.reset_parameters()

    
    def forward(self,edge_index,edge_attr,synapse,synapse_index,device,scatter_size):
        
        ### connectome encoder
        x = F.dropout(self.x, self.dropout, training=self.training)
     
        x = self.lins[0](x).relu()
        
        x_last=x.clone().detach()
        #x_last=torch.zeros_like(x)
        for i, conv in enumerate(self.convs):
            #x = F.dropout(x, self.dropout, training=self.training)
            x = conv(x,x_last,edge_index,edge_attr)
            x = x.relu()
        #     x=x+x_last
        #     x_last=x
        x_last =x.detach()
        x = self.mlp(x)


        ### synapse encoder
        
        x_point=self.synapse_encoder(synapse,synapse_index)
        
        x_point=x_point.to(device=device)
    
        left_node=scatter(x_point,edge_index[0],dim_size=scatter_size,reduce='mean')
        right_node=scatter(x_point,edge_index[1],dim_size=scatter_size,reduce='mean')
    
        final_node_vector=torch.cat([x,left_node,right_node],dim=1)
        #512
        # output=torch.cat([x,x2],dim=0)
        #x:0.52
        #left_node,right_node:0.68
        #[x,left_node,right_node]:0.76
        pred=self.classmodel(final_node_vector)
        
        return  pred
    

class SynapseNet_without_GCN(torch.nn.Module):
    def __init__(self, num_nodes, dataset_num_features, num_calss, represent_features=512,\
                 hidden_channels=64, num_layers=64, alpha=0.1, theta=0.5, shared_weights=True, dropout=0.0): 
        super().__init__()

        self.num_layers = num_layers
        #self.x=torch.nn.Parameter(torch.Tensor(21739,256))
        super().__init__()

        self.lins = torch.nn.ModuleList()
        self.lins.append(Linear(dataset_num_features, hidden_channels))
        self.mlp = MLP([hidden_channels, (represent_features - hidden_channels) // 3 + hidden_channels,\
                        2*(represent_features - hidden_channels) // 3 + hidden_channels,\
                        represent_features], dropout=0.5)
        self.x = torch.nn.Parameter(torch.Tensor(num_nodes, dataset_num_features))
        torch.nn.init.xavier_uniform(self.x, gain=1)
        self.convs = torch.nn.ModuleList()
        
        for layer in range(num_layers):
            self.convs.append(GCN2Conv(hidden_channels, alpha, theta, layer + 1, shared_weights, normalize=True))
        self.dropout = dropout
        self.outL = Linear(represent_features, num_calss)

        #TODO
        
        self.synapse_encoder=PointNet2()
        self.classmodel=MLP([1024,512,num_calss])


    def reset_parameters(self): #重置模型中所有层的参数，以便重新训练或初始化
        for conv in self.convs:
            conv.reset_parameters()

    
    def forward(self,edge_index,edge_attr,synapse,synapse_index,device,scatter_size):
    
        ### synapse encoder
        
        x_point=self.synapse_encoder(synapse,synapse_index)
        
        x_point=x_point.to(device=device)
    
    
        left_node=scatter(x_point,edge_index[0],dim_size=scatter_size,reduce='mean')
        right_node=scatter(x_point,edge_index[1],dim_size=scatter_size,reduce='mean')
    
        final_node_vector=torch.cat([left_node,right_node],dim=1)
        #512
        # output=torch.cat([x,x2],dim=0)
        
        pred=self.classmodel(final_node_vector)
        
        return  pred
    
class SynapseNet_GCN_MLP(torch.nn.Module):
    def __init__(self, num_nodes, dataset_num_features, num_calss, represent_features=512,\
                 hidden_channels=64, num_layers=64, alpha=0.1, theta=0.5, shared_weights=True, dropout=0.0): 
        super().__init__()

        self.num_layers = num_layers
        #self.x=torch.nn.Parameter(torch.Tensor(21739,256))
        super().__init__()

        self.lins = torch.nn.ModuleList()
        self.lins.append(Linear(dataset_num_features, hidden_channels))
        self.mlp = MLP([hidden_channels, (represent_features - hidden_channels) // 3 + hidden_channels,\
                        2*(represent_features - hidden_channels) // 3 + hidden_channels,\
                        represent_features], dropout=0.5)
        self.x = torch.nn.Parameter(torch.Tensor(num_nodes, dataset_num_features))
        torch.nn.init.xavier_uniform(self.x, gain=1)
        self.convs = torch.nn.ModuleList()
        
        for layer in range(num_layers):
            self.convs.append(GCN2Conv(hidden_channels, alpha, theta, layer + 1, shared_weights, normalize=True))
        self.dropout = dropout
        self.outL = Linear(represent_features, num_calss)

        #TODO
        self.synapse_encoder=MLP([6,128,256,512])
        self.classmodel=MLP([1536,512,256,num_calss])


    def reset_parameters(self): #重置模型中所有层的参数，以便重新训练或初始化
        for conv in self.convs:
            conv.reset_parameters()

    
    def forward(self,edge_index,edge_attr,synapse,synapse_index,device,scatter_size):
        
        ### connectome encoder
        x = F.dropout(self.x, self.dropout, training=self.training)
     
        x = self.lins[0](x).relu()
        
        x_last=x.clone().detach()
        #x_last=torch.zeros_like(x)
        for i, conv in enumerate(self.convs):
            #x = F.dropout(x, self.dropout, training=self.training)
            x = conv(x,x_last,edge_index,edge_attr)
            x = x.relu()
        #     x=x+x_last
        #     x_last=x
        x_last =x.detach()
        x = self.mlp(x)


        ### synapse encoder
        x_point= global_max_pool(synapse,synapse_index)
        x_point=self.synapse_encoder(x_point)
        
        x_point=x_point.to(device=device)
    
    
        left_node=scatter(x_point,edge_index[0],dim_size=scatter_size,reduce='mean')
        right_node=scatter(x_point,edge_index[1],dim_size=scatter_size,reduce='mean')
    
        final_node_vector=torch.cat([x,left_node,right_node],dim=1)
        
        # output=torch.cat([x,x2],dim=0)
        
        pred=self.classmodel(final_node_vector)
        
        return  pred

class SynapseNet_GAT_MLP(torch.nn.Module):
    def __init__(self, num_nodes, dataset_num_features, hidden_channels,represent_features, dataset_num_classes, heads=1): 
        super().__init__()
        self.conv1 = GATConv(dataset_num_features, hidden_channels, heads, dropout=0.6)
        # On the Pubmed dataset, use `heads` output heads in `conv2`.
        self.conv2 = GATConv(hidden_channels * heads, represent_features, heads=1,
                             concat=False, dropout=0.6)
        self.x = nn.Parameter(torch.Tensor(num_nodes, dataset_num_features))
        torch.nn.init.xavier_uniform(self.x, gain=1)
    

        #TODO
        self.synapse_encoder=MLP([6,256,512])
        self.classmodel=MLP([1536,512,dataset_num_classes])
    
    def forward(self,edge_index,edge_attr,synapse,synapse_index,device,scatter_size):
        
       ### connectome encoder
        x = F.dropout(self.x, p=0.6, training=self.training)
        x = F.elu(self.conv1(x, edge_index,edge_attr))
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, edge_index,edge_attr)


        ### synapse encoder
        x_point= global_max_pool(synapse,synapse_index)
        x_point=self.synapse_encoder(x_point)
        x_point=x_point.to(device=device)

        left_node=scatter(x_point,edge_index[0],dim_size=scatter_size,reduce='mean')
        right_node=scatter(x_point,edge_index[1],dim_size=scatter_size,reduce='mean')
        final_node_vector= torch.cat([x,left_node,right_node],dim=1)
        #x_point = torch.stack([x, left_node, right_node], dim=0)
        #x:torch.Size([21739, 512]), left_node:torch.Size([21739, 512]), right_node:torch.Size([21739, 512])
        #final_node_vector, _ = torch.max(x_point, dim=0)
        #final_node_vector=torch.cat([x,left_node,right_node],dim=1)
        pred=self.classmodel(final_node_vector)
        return  pred

class SynapseNet_GAT_PointNet_with_attr(torch.nn.Module):
    def __init__(self, num_nodes, dataset_num_features, hidden_channels,represent_features, heads, dataset_num_classes): 
        super().__init__()
        # self.conv1 = GATConv(dataset_num_features, hidden_channels, heads, dropout=0.6)
        # # On the Pubmed dataset, use `heads` output heads in `conv2`.
        # self.conv2 = GATConv(hidden_channels * heads, represent_features, heads=1,
        #                      concat=False, dropout=0.6)
        self.conv1 = GATConv(dataset_num_features, hidden_channels, heads, dropout=0.6)  # edge_dim is the size of edge features
        self.conv2 = GATConv(hidden_channels * heads, represent_features, heads=1,
                             concat=False, dropout=0.6)
        self.x = nn.Parameter(torch.Tensor(num_nodes, dataset_num_features))
        torch.nn.init.xavier_uniform(self.x, gain=1)
        self.cache_x = None

        #TODO
        
        self.synapse_encoder=PointNet2()
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=3)
        self.classmodel_Point=MLP([1024,256, dataset_num_classes])
        self.classmodel_GAT=MLP([512,256, dataset_num_classes])
        self.classmodel=MLP([represent_features+512+512, 512,dataset_num_classes])
        # self.classmodel=MLP([512+512, 512,dataset_num_classes])

    def forward(self,edge_index,edge_attr,synapse,synapse_index,device,scatter_size):
        
        ### connectome encoder
        x = F.dropout(self.x, p=0.6, training=self.training)
        x = F.elu(self.conv1(x, edge_index,edge_attr))
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, edge_index,edge_attr)


        ### synapse encoder
        
        x_point=self.synapse_encoder(synapse,synapse_index)
        
        x_point=x_point.to(device=device)
    
    
        left_node=scatter(x_point,edge_index[0],dim_size=scatter_size,reduce='mean')
        right_node=scatter(x_point,edge_index[1],dim_size=scatter_size,reduce='mean')
        final_node_vector=torch.cat([x,left_node,right_node],dim=1)
        # final_node_vector=torch.cat([left_node,right_node],dim=1)

        # x_point = torch.cat([left_node,right_node],dim=1)
        #x_point = torch.stack([x, left_node, right_node], dim=0)
        #x:torch.Size([21739, 512]), left_node:torch.Size([21739, 512]), right_node:torch.Size([21739, 512])
        #final_node_vector, _ = torch.max(x_point, dim=0)
        #final_node_vector=torch.cat([x,left_node,right_node],dim=1)
        # GATpred=self.classmodel_GAT(x)
        # Point_pred=self.classmodel_Point(x_point)
        
        # lambda1=0.45
        # pred=lambda1*GATpred+(1-lambda1)*Point_pred
        pred =self.classmodel(final_node_vector)
        return  pred

    def infer_x(self,edge_index,edge_attr,synapse,synapse_index,device,scatter_size):
        
        ### connectome encoder
        x = F.dropout(self.x, p=0.6, training=self.training)
        x = F.elu(self.conv1(x, edge_index,edge_attr))
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, edge_index,edge_attr)


        ### synapse encoder
        
        x_point=self.synapse_encoder(synapse,synapse_index)
        
        x_point=x_point.to(device=device)
    
    
        left_node=scatter(x_point,edge_index[0],dim_size=scatter_size,reduce='mean')
        right_node=scatter(x_point,edge_index[1],dim_size=scatter_size,reduce='mean')
        final_node_vector=torch.cat([x,left_node,right_node],dim=1)

    
        return  final_node_vector
    
    @torch.no_grad
    def inference(self,edge_index,edge_attr,synapse,synapse_index,device,scatter_size,data,current_test_index,reset_cache=True):
        if self.cache_x is None and reset_cache:
            x = F.dropout(self.x, p=0.6, training=self.training)
            x = F.elu(self.conv1(x, edge_index,edge_attr))
            x = F.dropout(x, p=0.6, training=self.training)
            x = self.conv2(x, edge_index,edge_attr)
            self.cache_x=x
        
        x_point=self.synapse_encoder(synapse,synapse_index)
        
        x_point=x_point.to(device=device)
        #[|Test|,d]
        
        #[|Train|+|Test|+,]
    
        # left_node=scatter(x_point,edge_index[0],dim_size=scatter_size,reduce='mean')
        # right_node=scatter(x_point,edge_index[1],dim_size=scatter_size,reduce='mean')
        final_node_vector=torch.cat([self.cache_x[current_test_index],x_point,x_point],dim=1)

        pred =self.classmodel(final_node_vector)
        return  pred



class SynapseNet_GAT_PointNet_with_attr_weight(torch.nn.Module):
    def __init__(self, num_nodes, dataset_num_features, hidden_channels,represent_features, heads, dataset_num_classes): 
        super().__init__()
        # self.conv1 = GATConv(dataset_num_features, hidden_channels, heads, dropout=0.6)
        # # On the Pubmed dataset, use `heads` output heads in `conv2`.
        # self.conv2 = GATConv(hidden_channels * heads, represent_features, heads=1,
        #                      concat=False, dropout=0.6)
        self.conv1 = GATConv(dataset_num_features, hidden_channels, heads, dropout=0.6)  # edge_dim is the size of edge features
        self.conv2 = GATConv(hidden_channels * heads, represent_features, heads=1,
                             concat=False, dropout=0.6)
        self.x = nn.Parameter(torch.Tensor(num_nodes, dataset_num_features))
        torch.nn.init.xavier_uniform(self.x, gain=1)
        
        #TODO
        
        self.synapse_encoder=Weight_PointNet2()
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=3)
        self.classmodel_Point=MLP([1024,256, dataset_num_classes])
        self.classmodel_GAT=MLP([512,256, dataset_num_classes])
        self.classmodel=MLP([represent_features+512+512, 512,dataset_num_classes])


    def forward(self,edge_index,edge_attr,synapse,synapse_index,device,scatter_size,weight):
        
        ### connectome encoder
        x = F.dropout(self.x, p=0.6, training=self.training)
        x = F.elu(self.conv1(x, edge_index,edge_attr))
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, edge_index,edge_attr)


        ### synapse encoder
        
        x_point=self.synapse_encoder(synapse,synapse_index,weight)
        
        x_point=x_point.to(device=device)

        if x_point.size(0) != edge_index[0].size(0):
            
            print("batch不等于edge_index的数量")

        
        else:
    
            left_node=scatter(x_point,edge_index[0],dim_size=scatter_size,reduce='mean')
            right_node=scatter(x_point,edge_index[1],dim_size=scatter_size,reduce='mean')
            # final_node_vector=torch.cat([x,left_node,right_node],dim=1)

            x_point = torch.cat([left_node,right_node],dim=1)
            #x_point = torch.stack([x, left_node, right_node], dim=0)
            #x:torch.Size([21739, 512]), left_node:torch.Size([21739, 512]), right_node:torch.Size([21739, 512])
            #final_node_vector, _ = torch.max(x_point, dim=0)

            GATpred=self.classmodel_GAT(x)
            Point_pred=self.classmodel_Point(x_point)
            
            lambda1=0.3
            pred=lambda1*GATpred+(1-lambda1)*Point_pred
            # pred =self.classmodel(final_node_vector)
            return  pred

class SynapseNet_GAT_PointNet(torch.nn.Module):
    def __init__(self, num_nodes, dataset_num_features, hidden_channels,represent_features, heads, dataset_num_classes): 
        super().__init__()
        self.conv1 = GATConv(dataset_num_features, hidden_channels, heads, dropout=0.6)
        # On the Pubmed dataset, use `heads` output heads in `conv2`.
        self.conv2 = GATConv(hidden_channels * heads, represent_features, heads=1,
                             concat=False, dropout=0.6)
        self.x = nn.Parameter(torch.Tensor(num_nodes, dataset_num_features))
        torch.nn.init.xavier_uniform(self.x, gain=1)
        
        #TODO
        
        self.synapse_encoder=PointNet2()
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=3)
        self.classmodel_Point=MLP([1024,256, dataset_num_classes])
        self.classmodel_GAT=MLP([512,256, dataset_num_classes])
        # self.classmodel=MLP([1536,512,dataset_num_classes])
        self.classmodel=MLP([represent_features+1024,512,dataset_num_classes])


    def forward(self,edge_index,edge_attr,synapse,synapse_index,device,scatter_size):
        
        ### connectome encoder
        x = F.dropout(self.x, p=0.6, training=self.training)
        x = F.elu(self.conv1(x, edge_index,edge_attr))
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, edge_index,edge_attr)


        ### synapse encoder
        
        x_point=self.synapse_encoder(synapse,synapse_index)
        
        x_point=x_point.to(device=device)
    
    
        left_node=scatter(x_point,edge_index[0],dim_size=scatter_size,reduce='mean')
        right_node=scatter(x_point,edge_index[1],dim_size=scatter_size,reduce='mean')
        final_node_vector=torch.cat([x,left_node,right_node],dim=1)

        # x_point = torch.cat([left_node,right_node],dim=1)
        #x_point = torch.stack([x, left_node, right_node], dim=0)
        #x:torch.Size([21739, 512]), left_node:torch.Size([21739, 512]), right_node:torch.Size([21739, 512])
        #final_node_vector, _ = torch.max(x_point, dim=0)
        #final_node_vector=torch.cat([x,left_node,right_node],dim=1)
        # GATpred=self.classmodel_GAT(x)
        # Point_pred=self.classmodel_Point(x_point)
        
        # lambda1=0.45
        # pred=lambda1*GATpred+(1-lambda1)*Point_pred
        pred =self.classmodel(final_node_vector)
        return  pred
    
    def infer_x(self,edge_index,edge_attr,synapse,synapse_index,device,scatter_size):
        x = F.dropout(self.x, p=0.6, training=self.training)
        x = F.elu(self.conv1(x, edge_index,edge_attr))
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, edge_index,edge_attr)


        ### synapse encoder
        
        x_point=self.synapse_encoder(synapse,synapse_index)
        
        x_point=x_point.to(device=device)
    
    
        left_node=scatter(x_point,edge_index[0],dim_size=scatter_size,reduce='mean')
        right_node=scatter(x_point,edge_index[1],dim_size=scatter_size,reduce='mean')
        final_node_vector=torch.cat([x,left_node,right_node],dim=1)
        return final_node_vector
    
    
    
      
    def forward2(self,edge_index,edge_attr,synapse,synapse_index,device,batch_rand_index):
    
        ### connectome encoder
        x = F.dropout(self.x, p=0.6, training=self.training)
        x = F.elu(self.conv1(x, edge_index,edge_attr))
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, edge_index,edge_attr)


        ### synapse encoder
        
        x_point=self.synapse_encoder(synapse,synapse_index)
        
        x_point=x_point.to(device=device)


        left_node=scatter(x_point,batch_rand_index[0],dim_size=21739,reduce='mean')
        right_node=scatter(x_point,batch_rand_index[1],dim_size=21739,reduce='mean')
        #stacked_tensors = torch.stack([x, left_node, right_node], dim=0)
        #x:torch.Size([21739, 512]), left_node:torch.Size([21739, 512]), right_node:torch.Size([21739, 512])
        #final_node_vector, _ = torch.max(stacked_tensors, dim=0)
        

        #final_node_vector=torch.cat([x,left_node,right_node],dim=1)
        
        # output=torch.cat([x,x2],dim=0)
        
        pred=self.classmodel(x)
        
        return  pred
    
    def forward3(self,edge_index,edge_attr,synapse,synapse_index,device,batch_rand_index):
        
        ### connectome encoder
        x = F.dropout(self.x, p=0.6, training=self.training)
        x = F.elu(self.conv1(x, edge_index,edge_attr))
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, edge_index,edge_attr)


        ### synapse encoder
        
        x_point=self.synapse_encoder(synapse,synapse_index)
        
        x_point=x_point.to(device=device)
    
    
        left_node=scatter(x_point,batch_rand_index[0],dim_size=21739,reduce='mean')
        right_node=scatter(x_point,batch_rand_index[1],dim_size=21739,reduce='mean')
        stacked_tensors = torch.stack([left_node, right_node], dim=0)
        #x:torch.Size([21739, 512]), left_node:torch.Size([21739, 512]), right_node:torch.Size([21739, 512])
        final_node_vector, _ = torch.max(stacked_tensors, dim=0)
        #final_node_vector=torch.cat([x,left_node,right_node],dim=1)
        
        # output=torch.cat([x,x2],dim=0)
        
        pred=self.classmodel(final_node_vector)
        
        return  pred

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
    
class Graph_UNet_PointNet(torch.nn.Module):
    def __init__(self,num_nodes,dataset_num_features, hidden_channels, dataset_num_classes):
        super().__init__()
        pool_ratios = [2000 / num_nodes, 0.5]
        self.unet = GraphUNet(dataset_num_features, hidden_channels, 512,
                              depth=3, pool_ratios=pool_ratios)
        self.x = nn.Parameter(torch.Tensor(num_nodes, dataset_num_features))
        torch.nn.init.xavier_uniform(self.x, gain=1)
        self.synapse_encoder=PointNet2()
        self.classmodel=MLP([1536,512,dataset_num_classes])
    def forward(self,edge_index,edge_attr,synapse,synapse_index,device,scatter_size):
        dropout_edge_index, _ = dropout_edge(edge_index, p=0.2,
                                     force_undirected=True,
                                     training=self.training)
        x = F.dropout(self.x, p=0.92, training=self.training)

        x = self.unet(x, dropout_edge_index)
          ### synapse encoder
        
        x_point=self.synapse_encoder(synapse,synapse_index)
        
        x_point=x_point.to(device=device)
    
    
        left_node=scatter(x_point,edge_index[0],dim_size=scatter_size,reduce='mean')
        right_node=scatter(x_point,edge_index[1],dim_size=scatter_size,reduce='mean')
    
        final_node_vector=torch.cat([x,left_node,right_node],dim=1)
        pred=self.classmodel(final_node_vector)
        return pred
    
    def infer_x(self,edge_index,edge_attr,synapse,synapse_index,device,scatter_size):
        dropout_edge_index, _ = dropout_edge(edge_index, p=0.2,
                                     force_undirected=True,
                                     training=self.training)
        x = F.dropout(self.x, p=0.92, training=self.training)

        x = self.unet(x, dropout_edge_index)
          ### synapse encoder
        
        x_point=self.synapse_encoder(synapse,synapse_index)
        
        x_point=x_point.to(device=device)
    
    
        left_node=scatter(x_point,edge_index[0],dim_size=scatter_size,reduce='mean')
        right_node=scatter(x_point,edge_index[1],dim_size=scatter_size,reduce='mean')
    
        final_node_vector=torch.cat([x,left_node,right_node],dim=1)
        return final_node_vector
    
class Graph_UNet_MLP(torch.nn.Module):
    def __init__(self,num_nodes,dataset_num_features, hidden_channels, dataset_num_classes):
        super().__init__()
        pool_ratios = [2000 / num_nodes, 0.5]
        self.unet = GraphUNet(dataset_num_features, hidden_channels, 512,
                              depth=3, pool_ratios=pool_ratios)
        self.x = nn.Parameter(torch.Tensor(num_nodes, dataset_num_features))
        torch.nn.init.xavier_uniform(self.x, gain=1)
        self.synapse_encoder=MLP([6,256,512])
        self.classmodel=MLP([1536,512,dataset_num_classes])
    def forward(self,edge_index,edge_attr,synapse,synapse_index,device,scatter_size):
        dropout_edge_index, _ = dropout_edge(edge_index, p=0.2,
                                     force_undirected=True,
                                     training=self.training)
        x = F.dropout(self.x, p=0.92, training=self.training)

        x = self.unet(x, dropout_edge_index)

        ### synapse encoder
        
        x_point= global_max_pool(synapse,synapse_index)
        x_point=self.synapse_encoder(x_point)
        x_point=x_point.to(device=device)
    
    
        left_node=scatter(x_point,edge_index[0],dim_size=scatter_size,reduce='mean')
        right_node=scatter(x_point,edge_index[1],dim_size=scatter_size,reduce='mean')
        final_node_vector= torch.cat([x,left_node,right_node],dim=1)

        pred=self.classmodel(final_node_vector)
        return pred
    
class DNANet_MLP(torch.nn.Module):
    def __init__(self, num_nodes,dataset_num_features, hidden_channels, dataset_num_classes, num_layers,
                 heads=1, groups=1):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.lin1 = torch.nn.Linear(dataset_num_features, hidden_channels)
        self.convs = torch.nn.ModuleList()
        for i in range(num_layers):
            self.convs.append(
                DNAConv(hidden_channels, heads, groups, dropout=0.8))
        self.lin2 = torch.nn.Linear(hidden_channels, 512)
        self.x = nn.Parameter(torch.Tensor(num_nodes, dataset_num_features))
        torch.nn.init.xavier_uniform(self.x, gain=1)
        self.synapse_encoder=MLP([6,256,512])
        self.classmodel=MLP([1536,512,dataset_num_classes])

    def forward(self,edge_index,edge_attr,synapse,synapse_index,device,scatter_size):
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

        ### synapse encoder
        
        x_point= global_max_pool(synapse,synapse_index)
        x_point=self.synapse_encoder(x_point)
        x_point=x_point.to(device=device)
    
    
        left_node=scatter(x_point,edge_index[0],dim_size=scatter_size,reduce='mean')
        right_node=scatter(x_point,edge_index[1],dim_size=scatter_size,reduce='mean')
        final_node_vector= torch.cat([x,left_node,right_node],dim=1)

        pred=self.classmodel(final_node_vector)
        return pred

class DNANet_PointNet(torch.nn.Module):
    def __init__(self, num_nodes,dataset_num_features, hidden_channels,  num_layers,
                 heads, groups,dataset_num_classes):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.lin1 = torch.nn.Linear(dataset_num_features, hidden_channels)
        self.convs = torch.nn.ModuleList()
        for i in range(num_layers):
            self.convs.append(
                DNAConv(hidden_channels, heads, groups, dropout=0.8))
        self.lin2 = torch.nn.Linear(hidden_channels, 512)
        self.x = nn.Parameter(torch.Tensor(num_nodes, dataset_num_features))
        # torch.nn.init.xavier_uniform(self.x, gain=1)
        self.synapse_encoder=PointNet2()
        self.classmodel=MLP([1536,512,dataset_num_classes])

    def forward(self,edge_index,edge_attr,synapse,synapse_index,device,scatter_size):
        x = F.relu(self.lin1(self.x))
        x = F.dropout(x, p=0.5, training=self.training)
        x_all = x.view(-1, 1, self.hidden_channels)
        for conv in self.convs:
            x = F.relu(conv(x_all, edge_index))
            x = F.relu(conv(x_all, edge_index ,edge_attr))
            x = x.view(-1, 1, self.hidden_channels)
            x_all = torch.cat([x_all, x], dim=1)
        x = x_all[:, -1]
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)

        ### synapse encoder
        
        x_point=self.synapse_encoder(synapse,synapse_index)
        
        x_point=x_point.to(device=device)
    
    
        left_node=scatter(x_point,edge_index[0],dim_size=scatter_size,reduce='mean')
        right_node=scatter(x_point,edge_index[1],dim_size=scatter_size,reduce='mean')
    
        final_node_vector=torch.cat([x,left_node,right_node],dim=1)
        pred=self.classmodel(final_node_vector)
        return pred
    @torch.no_grad
    # def infer_x(self,synapse,synapse_index,device):
        
    #     x1,x2,x3=self.synapse_encoder(synapse,synapse_index)
    #     x1,x2,x3=x1,x2,x3.to(device=device)
    #     return x1,x2,x3
    def infer_x(self,edge_index,edge_attr,synapse,synapse_index,device,scatter_size):
        
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

        ### synapse encoder
        
        x_point=self.synapse_encoder(synapse,synapse_index)
        
        x_point=x_point.to(device=device)
    
    
        left_node=scatter(x_point,edge_index[0],dim_size=scatter_size,reduce='mean')
        right_node=scatter(x_point,edge_index[1],dim_size=scatter_size,reduce='mean')
    
        final_node_vector=torch.cat([x,left_node,right_node],dim=1)
        return final_node_vector
        
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
    
    # def forward2(self, x, pos, edge_index):
    #     # 通过线性变换生成 Q, K, V
    #     query = self.lin_in(x)
    #     key = self.lin_in(x)
    #     value = self.lin_in(x)
        
    #     # 计算 Scaled Dot-Product Attention
    #     attn_output, attn_weights = self.attn(query, key, value)
    #     x = PointTransformerConv(in_channels, out_channels,
    #                                             pos_nn=self.pos_nn,
    #                                             attn_nn=self.attn_nn)
    #     # 激活函数
    #     x = self.lin_out(attn_output)
    #     x = self.relu(x)
        
    #     return x



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
    
class DNANet_Point_Transformer(torch.nn.Module):
    def __init__(self, num_nodes,dataset_num_features, hidden_channels, dataset_num_classes, num_layers,
                 heads=1, groups=1):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.lin1 = torch.nn.Linear(dataset_num_features, hidden_channels)
        self.convs = torch.nn.ModuleList()
        for i in range(num_layers):
            self.convs.append(
                DNAConv(hidden_channels, heads, groups, dropout=0.8))
        self.lin2 = torch.nn.Linear(hidden_channels, 512)
        self.x = nn.Parameter(torch.Tensor(num_nodes, dataset_num_features))
        torch.nn.init.xavier_uniform(self.x, gain=1)
        self.synapse_encoder=Point_Transformer(in_channels=6, out_channels=512, dim_model=[32, 64, 128, 256, 512], k=16)
        self.classmodel=MLP([1536,512,dataset_num_classes])

    def forward(self,edge_index,edge_attr,synapse,synapse_index,device):
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

        ### synapse encoder
        
        x_point=self.synapse_encoder(synapse,synapse_index)
        
        x_point=x_point.to(device=device)
    
    
        left_node=scatter(x_point,edge_index[0],dim_size=21739,reduce='mean')
        right_node=scatter(x_point,edge_index[1],dim_size=21739,reduce='mean')
    
        final_node_vector=torch.cat([x,left_node,right_node],dim=1)
        pred=self.classmodel(final_node_vector)
        return pred
    
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
        batch_labels=torch.arange(total_size//50)
        batches = batch_labels.repeat_interleave(50)
        remaining_size = total_size - batches.size(0)
        if remaining_size > 0:
            batches = torch.cat([batches, torch.full((remaining_size,), batch_labels[-1]+1)]).to(device)
        # pos = pos.to(device)
        try:
            if pos.device != batches.device:
                print(f"pos and batches are on different devices: {pos.device} and {batches.device}")
                batches  = batches.to(pos.device)
        except Exception as e:
            print(f"Unexpected error: {e}")
        x1 = self.conv1(pos, batch)
        x2 = self.conv2(x1, batch)
        out = self.lin1(torch.cat([x1, x2], dim=1)).to(device)
        out = global_max_pool(out, batch)
        out = self.mlp(out).to(device)
        out = F.log_softmax(out, dim=1)

        return out

class SynapseNet_GDCNN_PYG(torch.nn.Module):
    def __init__(self, num_calss): 
        super().__init__()
        
        self.synapse_encoder=DGCNN_PYG()
        self.classmodel=MLP([1024,512,num_calss])


    def forward(self,edge_index,edge_attr,synapse,synapse_index,device,scatter_size):
    
        ### synapse encoder
        
        x_point=self.synapse_encoder(synapse,synapse_index,device)
        
        x_point=x_point.to(device=device)
    
    
        left_node=scatter(x_point,edge_index[0],dim_size=scatter_size,reduce='mean')
        right_node=scatter(x_point,edge_index[1],dim_size=scatter_size,reduce='mean')
    
        final_node_vector=torch.cat([left_node,right_node],dim=1)
        #512
        # output=torch.cat([x,x2],dim=0)
        
        pred=self.classmodel(final_node_vector)
        
        return  pred

       

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

class SynapseNet_point3(torch.nn.Module):
    def __init__(self, num_calss): 
        super().__init__()
        
        self.synapse_encoder=PointNet3()
        self.classmodel=MLP([1024,512,num_calss])


    def forward(self,edge_index,edge_attr,synapse,synapse_index,device,scatter_size):
    
        ### synapse encoder
        
        x_point=self.synapse_encoder(synapse,synapse_index,device)
        
        x_point=x_point.to(device=device)
    
    
        left_node=scatter(x_point,edge_index[0],dim_size=scatter_size,reduce='mean')
        right_node=scatter(x_point,edge_index[1],dim_size=scatter_size,reduce='mean')
    
        final_node_vector=torch.cat([left_node,right_node],dim=1)
        #512
        # output=torch.cat([x,x2],dim=0)
        
        pred=self.classmodel(final_node_vector)
        
        return  pred

class SynapseNet_GAT_GDCNN(torch.nn.Module):
    def __init__(self,num_nodes, dataset_num_features, hidden_channels,represent_features, heads, dataset_num_classes): 
        super().__init__()
        self.conv1 = GATConv(dataset_num_features, hidden_channels, heads, dropout=0.6)  # edge_dim is the size of edge features
        self.conv2 = GATConv(hidden_channels * heads, represent_features, heads=1,
                             concat=False, dropout=0.6)
        self.x = nn.Parameter(torch.Tensor(num_nodes, dataset_num_features))
        torch.nn.init.xavier_uniform(self.x, gain=1)

        self.synapse_encoder=DGCNN_PYG()
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=3)
        self.classmodel=MLP([represent_features+512+512,512,dataset_num_classes])


    def forward(self,edge_index,edge_attr,synapse,synapse_index,device,scatter_size):
        
        ### connectome encoder
        x = F.dropout(self.x, p=0.6, training=self.training)
        x = F.elu(self.conv1(x, edge_index,edge_attr))
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, edge_index,edge_attr)

        ### synapse encoder
        x_point=self.synapse_encoder(synapse,synapse_index,device)
        x_point=x_point.to(device=device)
    
        left_node=scatter(x_point,edge_index[0],dim_size=scatter_size,reduce='mean')
        right_node=scatter(x_point,edge_index[1],dim_size=scatter_size,reduce='mean')
        final_node_vector=torch.cat([x,left_node,right_node],dim=1)
        
        pred=self.classmodel(final_node_vector)
        
        return  pred
