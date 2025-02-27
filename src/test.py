import os
import sys
project_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_path not in sys.path:
    sys.path.append(project_path)
import torch
import torch.nn.functional as F


def test(model, data, batch_edge, batch_rand_attr, batch_synapse, batch_synpase_index, test_index, device,scatter_size,batch_synapse_attr):
    model.eval()
    with torch.no_grad():
        # edge_attr=torch.ones(size=(batch_edge.size(1),))
        # edge_attr=edge_attr.to(device=device)
        out = model(batch_edge, batch_rand_attr,batch_synapse,batch_synpase_index, device,scatter_size)
        ### pointnet_transformer
        # out=model(batch_synapse,batch_synpase_index)
        out=out[test_index]
        y = data.y[test_index]
        pred=out.argmax(dim=-1)
        hit=pred==y
        acc=float(torch.sum(hit)/y.size(0))
    return acc

def test_fix(model, data, batch_edge, batch_rand_attr, batch_synapse, batch_synpase_index, test_index, device,scatter_size,batch_synapse_attr):
    model.eval()
    with torch.no_grad():
        # edge_attr=torch.ones(size=(batch_edge.size(1),))
        # edge_attr=edge_attr.to(device=device)
        out = model(batch_edge, batch_rand_attr,batch_synapse,batch_synpase_index, device,scatter_size)
        ### pointnet_transformer
        # out=model(batch_synapse,batch_synpase_index)
        out=out[test_index]
        y = data.y[test_index]
        valid_mask = (y != 0)
        y_without0=y[valid_mask]
        out_without0=out[valid_mask]
        pred=out_without0.argmax(dim=-1)
        hit=pred==y_without0
        acc=float(torch.sum(hit)/y_without0.size(0))
        # f1=f1_score(y_without0.cpu(),pred.cpu(),average='micro')
        # print(f'f1:{f1}')
    return acc
