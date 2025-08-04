import os
import sys
project_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_path not in sys.path:
    sys.path.append(project_path)
import torch
import torch.nn.functional as F

def train(model, data, batch_edge, batch_rand_attr,batch_synapse, batch_synpase_index, train_index, optimizer, device,scatter_size,batch_synapse_attr):
    model.train()    
    optimizer.zero_grad()
    # edge_attr=torch.ones(size=(batch_edge.size(1),)) 
    # edge_attr=edge_attr.to(device=device)
    out = model(batch_edge,batch_rand_attr,batch_synapse,batch_synpase_index,device,scatter_size)
    ### pointnet_transformer
    # out=model(batch_synapse,batch_synpase_index)
    out=out[train_index]#(85167,133)
    y = data.y[train_index]
    loss = F.cross_entropy(out, y)
    loss.backward()
    optimizer.step()
    pred=out.argmax(dim=-1)
    hit=pred==y
    acc=float(torch.sum(hit)/y.size(0))
    return acc, loss
