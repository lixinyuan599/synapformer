import os
import sys
project_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_path not in sys.path:
    sys.path.append(project_path)
import src.utils.config as config
import torch
from data.hemibrain import HemiBrainSynapse
from data.manc import Manc
import torch
import numpy as np
import sys
import random
from src.models.model import Synapformer
from src.utils.datahandler import data_rand
from src.utils.train import train
from src.utils.test import test_all





args = config.parse_arguments()

def setup_seed(seed):
    torch.manual_seed(seed) 
    torch.cuda.manual_seed_all(seed) 
    np.random.seed(seed) 
    random.seed(seed) 


def main(name):
    seed=22
    setup_seed(seed)

    if name=="Hemibrain":
        device = torch.device(f'cuda:{args.deviceID}' if torch.cuda.is_available() else 'cpu')
        path='/data3/synapse/MICCAI/data/HemiBrain'
        dataset=HemiBrainSynapse(root=path,device=device)
        data = dataset[0].to(device)
        scatter_size=21739
        selected_edge_index=data.edge_index

    if name=="Manc":
        device = torch.device(f'cuda:{args.deviceID}' if torch.cuda.is_available() else 'cpu')
        path='/data3/lixinyuan/synapse/MICCAI/data/Manc'
        dataset=Manc(root=path,device=device)
        data = dataset[0].to(device)
        scatter_size=23188
        selected_edge_index = data.edge_index

    synapse_judgment=args.synapse
    node_num=len(dataset.data.y)
    y=data.y


    if name=="Hemibrain":
        model = Synapformer(node_num, dataset_num_features=1024, hidden_channels=64,represent_features=512,\
                             dataset_num_classes=dataset.num_classes, heads=1)
        
    if name=="Manc":
        model = Synapformer(node_num, dataset_num_features=1024, hidden_channels=32,represent_features=512,\
                             dataset_num_classes=dataset.num_classes, heads=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.train_lr)
    model = model.to(device)

### train 
    max_test_acc=0
    best_loss=float('inf')
    no_improve=0
    patience=args.patience
    for epoch in range(args.epoch):
        print('*'*60)
        if synapse_judgment==True:
            chunk=0
            bad=0
            for batch_synapse, batch_synpase_index, batch_rand_index,batch_rand_attr,batch_synapse_attr in data_rand(data, device,selected_edge_index):
                chunk=chunk+1
                bad=bad+1
                train_acc, loss=train(model, data, batch_rand_index, batch_rand_attr, batch_synapse, batch_synpase_index, data.train_mask, optimizer, device,scatter_size) # Manc:23188 Himebrain:21739

                if chunk % 10==0:
                    test_acc,precision,f1,recall=test_all(model, data, batch_rand_index, batch_rand_attr, batch_synapse, batch_synpase_index, data.test_mask, device,scatter_size)
                    print(f'epoch:{epoch:03d} , Test_ACC:{test_acc:0.4f}, Precision:{precision:0.4f}, F1:{f1:0.4f}, Recall:{recall:0.4f}')

                    if test_acc > max_test_acc:
                        max_test_acc = test_acc

                if loss < best_loss:
                    best_loss=loss
                    no_improve=0
                    print(f'epoch:{epoch:03d}, chunk:{chunk:02d}, Train_loss:{loss:0.6f}, Train_ACC:{train_acc:0.4f}')
                else:
                    no_improve +=1
            test_acc,precision,f1,recall=test_all(model, data, batch_rand_index, batch_rand_attr, batch_synapse, batch_synpase_index, data.test_mask, device,scatter_size)
            print(f'epoch:{epoch:03d} , Test_ACC:{test_acc:0.4f}, Precision:{precision:0.4f}, F1:{f1:0.4f}, Recall:{recall:0.4f}')
        if no_improve >= patience:
            print('early stopping!!!')
            break

               

    return max_test_acc


if __name__ == '__main__':
    # max_test_acc=main("Hemibrain")
    max_test_acc=main("Manc")

    print(f' Max_Test ACC: {max_test_acc:.4f}')