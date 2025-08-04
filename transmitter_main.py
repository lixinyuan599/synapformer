import os
import sys
project_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_path not in sys.path:
    sys.path.append(project_path)
import src.utils.config as config
import torch
import torch
import numpy as np
import sys
import random
from src.models.model import Synapformer,Synapformer2,Synapformer3
from src.utils.datahandler import data_rand_hemi_6,test_loader_all_hemi
from src.train import train
from src.test import test_fix, test,test_all
from src.models.NeuronLearner import BaseLearner
from src.models.connectome_enco import GAT_without_attr
from src.models.synapse_enco import synapse_former_6
from datetime import datetime



args = config.parse_arguments()

def setup_seed(seed):
    torch.manual_seed(seed) 
    torch.cuda.manual_seed_all(seed) 
    np.random.seed(seed) 
    random.seed(seed) 

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def main(name):
    seed=22
    setup_seed(seed)

    name=="Hemibrain"
    device = torch.device(f'cuda:{args.deviceID}' if torch.cuda.is_available() else 'cpu')
    data=torch.load('/data3/lixinyuan/neuron_learning/data/hemibrain/post_data.pt').to(device)
    y=data.y
    selected_edge_index=data.edge_index
    synapse_judgment=args.synapse    
    num_classes=len(y.unique())
    node_num=y.shape[0]




    index = torch.randperm(len(y)).tolist() 
    train_index = index[:len(y)//10*9] 
    test_index = index[len(y)//10*8:len(y)//10*9]
    val_index = index[len(y)//10*9:]

    train_mask = torch.zeros((len(y), ), dtype=torch.bool).to(device)
    val_mask = torch.zeros((len(y),), dtype=torch.bool).to(device)
    test_mask = torch.zeros((len(y), ), dtype=torch.bool).to(device)
    train_mask[train_index] = True
    val_mask[val_index] = True
    test_mask[test_index] = True
    
    model = BaseLearner(
                    connectome_enco = GAT_without_attr(node_num,  dataset_num_features=1024, hidden_channels=32, represent_features=512, heads=2),
                    synapse_enco=synapse_former_6(represent_features=512),
                    dataset_num_classes=num_classes)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.train_lr)
    model = model.to(device)

    initial_params = count_parameters(model)
    print(f"Initial number of parameters: {initial_params}")

    with open('/data3/lixinyuan/neuron_learning/src/models/model_parameters.txt', 'a') as file:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        file.write(f"{timestamp} - {args.model_name} Initial number of parameters: {initial_params}\n")

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
            for batch_synapse, batch_synpase_index, batch_rand_index,batch_rand_attr,batch_synapse_attr ,batch_train_mask,batch_test_mask,batch_y in data_rand_hemi_6(data, device,selected_edge_index,train_mask,test_mask,y):
                chunk=chunk+1
                bad=bad+1
                train_acc, loss=train(model, data, batch_rand_index, batch_rand_attr, batch_synapse, batch_synpase_index, train_mask, optimizer, device,batch_synapse_attr,batch_y, batch_train_mask) # Manc:23188 Himebrain:21739

                if chunk % 10==0:
                    edge_index_test_chunk,edge_attr_test_chunk,synapse_test_chunk,batch_test_chunk,y_test_chunks = test_loader_all_hemi(data,test_mask,device)
                    test_acc,precision,f1,recall,out,select_y =test_all(model, data, edge_index_test_chunk, edge_attr_test_chunk, synapse_test_chunk, batch_test_chunk , device, y_test_chunks)
                    print(f'epoch:{epoch:03d} , Test_ACC:{test_acc:0.4f}, Precision:{precision:0.4f}, F1:{f1:0.4f}, Recall:{recall:0.4f}')
                        
                    if test_acc > max_test_acc:
                        max_test_acc = test_acc

                if loss < best_loss:
                    best_loss=loss
                    no_improve=0
                    print(f'epoch:{epoch:03d}, chunk:{chunk:02d}, Train_loss:{loss:0.6f}, Train_ACC:{train_acc:0.4f}')
                else:
                    no_improve +=1

            
            edge_index_test_chunk,edge_attr_test_chunk,synapse_test_chunk,batch_test_chunk,y_test_chunks = test_loader_all_hemi(data,test_mask,device)
            test_acc,precision,f1,recall,_,_,=test_all(model, data, edge_index_test_chunk, edge_attr_test_chunk, synapse_test_chunk, batch_test_chunk , device, y_test_chunks)
            print(f'epoch:{epoch:03d} , Test_ACC:{test_acc:0.4f}, Precision:{precision:0.4f}, F1:{f1:0.4f}, Recall:{recall:0.4f}')
        if no_improve >= patience:
            print('early stopping!!!')
            break

               

    return max_test_acc


if __name__ == '__main__':
    # max_test_acc=main("FAFB")
    max_test_acc=main("Hemibrain")    
    print(f' Max_Test ACC: {max_test_acc:.4f}')