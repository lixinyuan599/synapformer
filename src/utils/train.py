import os
import sys
project_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_path not in sys.path:
    sys.path.append(project_path)
import torch
import torch.nn.functional as F

def train(model, data, batch_edge, batch_rand_attr,batch_synapse, batch_synpase_index, train_index, optimizer, device,scatter_size):
    model.train()    
    optimizer.zero_grad()
    out,_ ,_= model(batch_edge,batch_rand_attr,batch_synapse,batch_synpase_index,device,scatter_size)
    out=out[train_index]
    y = data.y[train_index]
    loss = F.cross_entropy(out, y)
    loss.backward()
    optimizer.step()
    pred=out.argmax(dim=-1)
    hit=pred==y
    acc=float(torch.sum(hit)/y.size(0))
    return acc, loss

def train_tran(model, data, batch_synapse, batch_synpase_index, train_index, optimizer, device):
    model.train()    
    optimizer.zero_grad()
    out= model(batch_synapse,batch_synpase_index)
    out=out[train_index]
    y = data.y[train_index]
    loss = F.cross_entropy(out, y)
    loss.backward()
    optimizer.step()
    pred=out.argmax(dim=-1)
    hit=pred==y
    acc=float(torch.sum(hit)/y.size(0))
    return acc, loss
