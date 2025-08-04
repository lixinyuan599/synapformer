import os
import sys
project_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_path not in sys.path:
    sys.path.append(project_path)
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score, precision_score,recall_score


def test(model, data, batch_edge, batch_rand_attr, batch_synapse, batch_synpase_index, test_index, device,scatter_size,batch_synapse_attr):
    model.eval()
    with torch.no_grad():
        out = model(batch_edge, batch_rand_attr,batch_synapse,batch_synpase_index, device,scatter_size)
        out=out[test_index]
        y = data.y[test_index]
        pred=out.argmax(dim=-1)
        hit=pred==y
        acc=float(torch.sum(hit)/y.size(0))
    return acc

def test_fix(model, data, batch_edge, batch_rand_attr, batch_synapse, batch_synpase_index, test_index, device,scatter_size,batch_synapse_attr):
    model.eval()
    with torch.no_grad():
        out = model(batch_edge, batch_rand_attr,batch_synapse,batch_synpase_index, device,scatter_size)
        out=out[test_index]
        y = data.y[test_index]
        valid_mask = (y != 0)
        y_without0=y[valid_mask]
        out_without0=out[valid_mask]
        pred=out_without0.argmax(dim=-1)
        hit=pred==y_without0
        acc=float(torch.sum(hit)/y_without0.size(0))
    return acc

def test_all(model, data, batch_edge, batch_rand_attr, batch_synapse, batch_synpase_index, test_index, device,scatter_size):
    model.eval()
    with torch.no_grad():
        out,point_attn,pos= model(batch_edge, batch_rand_attr,batch_synapse,batch_synpase_index, device,scatter_size)
        out=out[test_index]
        y = data.y[test_index]
        valid_mask = (y != 0)
        y_without0=y[valid_mask]
        out_without0=out[valid_mask]
        pred=out_without0.argmax(dim=-1)
        hit=pred==y_without0
        acc=float(torch.sum(hit)/y_without0.size(0))
        y_true = y_without0.cpu().numpy()  # 真实标签
        y_pred = pred.cpu().numpy()  # 预测标签

        # 计算精确率（Precision）、召回率（Recall）、F1-score
        precision = precision_score(y_true, y_pred, average='macro')
        f1 = f1_score(y_true, y_pred, average='macro')
        recall = recall_score(y_true, y_pred, average='macro')
        # point_attn=torch.save(point_attn, '/data3/lixinyuan/synapse/MICCAI/model/point_attn.pt')
        # pos=torch.save(pos, '/data3/lixinyuan/synapse/MICCAI/model/pos.pt')
    return acc,precision,f1,recall, point_attn, pos

def test_all(model, data, batch_edge, batch_rand_attr, batch_synapse, batch_synpase_index, test_index, device,scatter_size):
    model.eval()
    with torch.no_grad():
        out,point_attn,pos= model(batch_edge, batch_rand_attr,batch_synapse,batch_synpase_index, device,scatter_size)
        out=out[test_index]
        y = data.y[test_index]
        valid_mask = (y != 0)
        y_without0=y[valid_mask]
        out_without0=out[valid_mask]
        pred=out_without0.argmax(dim=-1)
        hit=pred==y_without0
        acc=float(torch.sum(hit)/y_without0.size(0))
        y_true = y_without0.cpu().numpy()  # 真实标签
        y_pred = pred.cpu().numpy()  # 预测标签

        # 计算精确率（Precision）、召回率（Recall）、F1-score
        precision = precision_score(y_true, y_pred, average='macro')
        f1 = f1_score(y_true, y_pred, average='macro')
        recall = recall_score(y_true, y_pred, average='macro')
        # point_attn=torch.save(point_attn, '/data3/lixinyuan/synapse/MICCAI/model/point_attn.pt')
        # pos=torch.save(pos, '/data3/lixinyuan/synapse/MICCAI/model/pos.pt')
        # torch.save(out,'/data3/lixinyuan/synapse/MICCAI/src/synapformer_out.pt')
        # torch.save(out,'/data3/lixinyuan/synapse/MICCAI/src/synapformer_out.pt')
    # return acc,precision,f1,recall, point_attn, pos
    return acc,precision,f1,recall
