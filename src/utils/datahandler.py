import torch
import numpy as np
import torch.nn.functional as F
import wandb
from tqdm import tqdm
import sys
import pandas as pd
from torch_geometric.utils import unbatch,mask_select
import os
import random
import src.utils.config as config

args = config.parse_arguments()

def edge_dataloader(data,device,edge_index):
    rand_ind=torch.randperm(edge_index.size(1)).to(device)
    rand_index=torch.nn.functional.embedding(rand_ind,edge_index.T)
    rand_index=rand_index.T.to(device)
    edge_loader=torch.chunk(rand_index,args.chunk_size,dim=1)
    return edge_loader

def data_rand(data, device,edge_index):
    rand_ind=torch.randperm(edge_index.size(1)).to(device)
    edge_ind_loader=torch.chunk(rand_ind,args.chunk_size,dim=0)
    c=unbatch(data.synapse,data.synapse_index)
    for i in range(len(edge_ind_loader)):
        batch_ind_edge=edge_ind_loader[i]
        batch_rand_index=torch.nn.functional.embedding(batch_ind_edge,edge_index.T)
        batch_rand_index=batch_rand_index.T.to(device)

        batch_rand_attr=torch.nn.functional.embedding(batch_ind_edge,data.edge_attr.unsqueeze(1))
        batch_rand_attr=batch_rand_attr.squeeze().to(device)
        res=[]
        batch_synapse_attr=[]
        batch_synpase_index=[]
        k=0
        MAX_SYNAPSE=100000
        for idx ,i in enumerate(batch_ind_edge):
            if len(c[i]) >= MAX_SYNAPSE:
                random.shuffle(c[i])
                newc=c[i][:MAX_SYNAPSE]
                res.append(newc)
                batch_synpase_index.append((torch.ones(size=(MAX_SYNAPSE,))*k).to(torch.int64))
                batch_synapse_attr.append((torch.ones(size=(len(c[i]),))*batch_rand_attr[idx].cpu()))
                k+=1
            else:
                res.append(c[i])
                batch_synpase_index.append((torch.ones(size=(len(c[i]),))*k).to(torch.int64))
                batch_synapse_attr.append((torch.ones(size=(len(c[i]),))*batch_rand_attr[idx].cpu()))
                k+=1
                
        batch_synpase_index=torch.cat(batch_synpase_index,dim=0).to(device)
        batch_synapse=(torch.cat(res,dim=0)/10000).to(device)
        batch_synapse_attr=torch.cat(batch_synapse_attr,dim=0).to(device)

        yield (batch_synapse, batch_synpase_index,batch_rand_index,batch_rand_attr,batch_synapse_attr)

def data_rand_mamba(data, device, edge_index):
    rand_ind = torch.randperm(edge_index.size(1)).to(device)
    edge_ind_loader = torch.chunk(rand_ind, args.chunk_size, dim=0)
    c = unbatch(data.synapse, data.synapse_index)

    for i in range(len(edge_ind_loader)):
        batch_ind_edge = edge_ind_loader[i]
        batch_rand_index = torch.nn.functional.embedding(batch_ind_edge, edge_index.T)
        batch_rand_index = batch_rand_index.T.to(device)

        batch_rand_attr = torch.nn.functional.embedding(batch_ind_edge, data.edge_attr.unsqueeze(1))
        batch_rand_attr = batch_rand_attr.squeeze().to(device)

        res = []
        batch_synapse_attr = []
        batch_synpase_index = []
        k = 0
        MAX_SYNAPSE = 100000

        for idx, i in enumerate(batch_ind_edge):
            if len(c[i]) >= MAX_SYNAPSE:
                random.shuffle(c[i])
                newc = c[i][:MAX_SYNAPSE]
            else:
                newc = c[i]
            res.append(newc)
            batch_synpase_index.append((torch.ones(size=(len(newc),)) * k).to(torch.int64))
            batch_synapse_attr.append((torch.ones(size=(len(newc),)) * batch_rand_attr[idx].cpu()))
            k += 1

        batch_synpase_index = torch.cat(batch_synpase_index, dim=0).to(device)
        batch_synapse = (torch.cat(res, dim=0) / 10000).to(device)
        batch_synapse_attr = torch.cat(batch_synapse_attr, dim=0).to(device)

        # # 🎯 转为 Mamba 输入
        # soe_sequences = []
        # for i in range(k):
        #     mask = (batch_synpase_index == i)
        #     seq = batch_synapse[mask]  # shape: [num_synapses, feature_dim]
        #     soe_sequences.append(seq)

        yield (  batch_synapse, batch_synpase_index,batch_rand_index, batch_rand_attr )

def data_chunk(data, chunk_size, train_mask,device):
    synapse = data.synapse.to(device)
    synapse_index = data.synapse_index.to(device)

    synapse_chunks = torch.chunk(synapse, chunk_size, dim=0)
    synapse_index_chunks = torch.chunk(synapse_index, chunk_size, dim=0)
    train_mask =torch.chunk(train_mask, chunk_size, dim=0)

    for syn_chunk, idx_chunk ,batch_mask in zip(synapse_chunks, synapse_index_chunks,train_mask):
        # 局部重新编号
        _, relabel = torch.unique(idx_chunk, return_inverse=True)
        yield syn_chunk, relabel, batch_mask
