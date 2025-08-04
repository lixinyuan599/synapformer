from hashlib import new
from typing import Optional, Callable, List
import os.path as osp
import numpy as np
import torch
import torch_geometric
from torch_geometric.data import InMemoryDataset, download_url
from torch_geometric.data import Data
import glob
import networkx as nx
import pandas as pd
from tqdm import tqdm
import os
import json
os.environ['PYDEVD_CONTAINER_RANDOM_ACCESS_MAX_ITEMS'] = '24000' 


class Manc(InMemoryDataset):
    url=None
    r"""
    Args:
        root (string): Root directory where the dataset should be saved.
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
    """

    url = 'null'
    _version='1.2'

    def __init__(self, root: str,split='node', transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None,process=False,restore_split=None,device=None):

        super().__init__(root, transform, pre_transform)
        if device is None:
            self.device='cpu'
        else:
            self.device=device
        if process:
            self._process()
        if restore_split is None:
            self._data, self.slices = torch.load(self.processed_paths[0],map_location=self.device)
        elif isinstance(restore_split,str):
            self._data= torch.load(restore_split,map_location=self.device)
        self._data, self.slices = self.collate([self._data])

    @property
    def num_edges(self):
        return self._data.edge_index.size(1)
    @property
    def edge_index(self):
        return self._data.edge_index
    @property
    def y(self):
        return self._data.y
    
    @property
    def num_nodes(self):
        return self._data.num_nodes
    @property
    def raw_dir(self) -> str:
        return osp.join(self.root, 'raw')
    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, 'processed')
    
    @property
    def raw_file_names(self) -> List[str]:
        return ['connections-per-roi.csv','connections2.csv','neurons.csv']
    @property
    def processed_file_names(self) -> str:
        return 'data.pt'
    @property
    def neu2ID(self):
        if hasattr(self,'neu2ID_dict'):
            return self.neu2ID_dict
        else:
            res=pd.read_csv(osp.join(self.raw_dir,'neurons.csv'))   
            self.neu2ID_dict=dict(zip(res['bodyId'],list(range(len(res['bodyId'])))))
            return self.neu2ID_dict
    @property
    def label2ID(self):
        if hasattr(self,'label2ID_dict'):
            return self.label2ID_dict
        else:
            res=pd.read_csv(osp.join(self.raw_dir,'neurons.csv'))
            label_set= set(res['type'].to_list())
            self.label2ID_dict={l:k for k,l in enumerate(label_set)}
            return self.label2ID_dict
    @property
    def neuron2label(self):
        if hasattr(self,'neuron2label_dict'):
            return self.neuron2label_dict
        else:
            
            res=pd.read_csv(osp.join(self.raw_dir,'neurons.csv'))   
            self.neuron2label_dict=dict(zip(res['bodyId'],res['type'].str[:2]))
            return self.neuron2label_dict
    @property
    def typedim(self):
        if hasattr(self,'cache_type_dim'):
            return self.cache_type_dim
        else:
            label2ID=self.label2ID
            self.cache_type_dim=len(label2ID.keys())
            return self.cache_type_dim
    
    @staticmethod    
    def save_neuron2label_dict(self, output_file):

        if hasattr(self, 'neuron2label_dict'):
            df = pd.DataFrame(list(self.neuron2label_dict.items()), columns=['bodyId', 'type'])
            df.to_csv(output_file, index=False)
            print(f"Saved neuron2label_dict to {output_file}")
        else:
            print("neuron2label_dict does not exist.")

    @staticmethod
    def read_data(raw_dir):
        res=pd.read_csv(osp.join(raw_dir,'neurons.csv')) 
        neu2ID=dict(zip(res['bodyId'],list(range(len(res['bodyId'])))))
        ID2neu={v:k for k,v in neu2ID.items()}
        edge_index=[]
        with open(osp.join(raw_dir,'connections2.csv'),'r') as fin:
            for i in fin.readlines():
                line=i.strip().split(',')
                if len(line)==3:
                    edge_index.append([neu2ID[int(line[0])],neu2ID[int(line[1])]])
        edge_index=torch.Tensor(edge_index)
        edge_index=torch.transpose(edge_index,0,1)
        edge_index=edge_index.to(dtype=torch.int64)
        edge_index=edge_index.contiguous()




        connections=pd.read_csv(osp.join(raw_dir,'connections2.csv'))
        

        edge_attr=connections['weight'].to_list()
        edge_attr=torch.Tensor(edge_attr)
        neuron2label=dict(zip(res['bodyId'],res['type'].str[:2]))

        label_set= set(neuron2label.values())
        label2ID = {np.nan: 0}
        label2ID.update({l: k+1 for k, l in enumerate(sorted(label_set - {np.nan}))})
        with open ('/data3/synapse/Synapses/Manc/label.txt','w',encoding='utf-8') as file:
            json.dump(label2ID,file,ensure_ascii=False,indent=4)
            
        y=[]
        neuronname=[]
        labelname=[]
        for i_neuron in range(len(ID2neu)):
            label_x=neuron2label[ID2neu[i_neuron]]
            y.append(label2ID[label_x])
            nname=ID2neu[i_neuron]
            neuronname.append(nname)
            labelname.append(label_x)
        y=torch.tensor(y)

        x=torch.nn.functional.one_hot(y)
        x=x.to(dtype=torch.float32)


        sysnase_connection_dict = {}
        csv_files=glob.glob('/data0/project/Synapses/Manc/synapses/*.csv')
        
        for csv_path in tqdm(csv_files):
            neu_name=osp.basename(csv_path)
            neu_name=neu_name.replace('.csv','')
            
            df = pd.read_csv(csv_path)
            
            
            for i in range(len(df)):
                
                row=df.loc[i]
                pre_neu=int(row['bodyId_pre'])
                post_neu=int(row['bodyId_post'])
                if pre_neu in neu2ID and post_neu in neu2ID.keys():
                    pre_neu=int(neu2ID[pre_neu])
                    post_neu=int(neu2ID[post_neu])

                    cordi=row[['x_pre','y_pre','z_pre','x_post','y_post','z_post']].to_numpy()
                    cordi = cordi.astype(np.int64)  
                    cordi = torch.tensor(cordi, dtype=torch.int64)
                    
                    if (pre_neu,post_neu) in sysnase_connection_dict:
                        sysnase_connection_dict[(pre_neu,post_neu)].append(cordi)
                    else:
                        sysnase_connection_dict[(pre_neu,post_neu)]=[cordi]

        synapse=[]
        synapse_index=[]
        synapse_id=0

    
       
        for i in tqdm(range(edge_index.size(1))):
            
            neu1,neu2=int(edge_index[:,i][0]),int(edge_index[:,i][1])
            
            if (neu1,neu2) in sysnase_connection_dict:
                synapse.append(sysnase_connection_dict[(neu1,neu2)])
                synapse_index.append([synapse_id]*len(sysnase_connection_dict[(neu1,neu2)]))
                synapse_id+=1
            else:
                data_list = [0, 0, 0, 0, 0, 0]
                tensor_data = torch.tensor(data_list)
                synapse.append([tensor_data])
                synapse_index.append([synapse_id])
                synapse_id+=1
            
        synapse=np.concatenate(synapse,axis=0)
        synapse=torch.from_numpy(synapse)
        
        synapse_index = torch.cat([torch.tensor(index) for index in synapse_index])
        synapse_index=torch.tensor(synapse_index)    

        index = torch.randperm(len(y)).tolist() 
        train_index = index[:len(y)//10*8] 
        test_index = index[len(y)//10*8:len(y)//10*9]
        val_index = index[len(y)//10*9:]

        train_mask = torch.zeros((len(y), ), dtype=torch.bool)
        val_mask = torch.zeros((len(y),), dtype=torch.bool)
        test_mask = torch.zeros((len(y), ), dtype=torch.bool)
        train_mask[train_index] = True
        val_mask[val_index] = True
        test_mask[test_index] = True

        data=Data(edge_index=edge_index,y=y,edge_attr=edge_attr,synapse=synapse,synapse_index=synapse_index,neuronname=neuronname,labelname=labelname)
        data.train_mask = train_mask
        data.val_mask = val_mask
        data.test_mask = test_mask

        return data



    def process(self):
        data = self.read_data(self.raw_dir)
        data = data if self.pre_transform is None else self.pre_transform(data)
        torch.save(self.collate([data]), self.processed_paths[0])
    def manual_split(self,path):
        res=pd.read_csv(osp.join(self.raw_dir,'neurons.csv')) 
        neuron2label=dict(zip(res['bodyId'],res['type']))
        label_set= set(neuron2label.values())
        label2ID={l:k for k,l in enumerate(label_set)}
        neu2ID=dict(zip(res['bodyId'],list(range(len(res['bodyId'])))))
        
        for k,v in self.neu2ID.items():
            if neu2ID[k]!=v:
                raise ValueError('this file doesn`t match this object(%s)'%__name__)
        raise NotImplemented
    def __repr__(self) -> str:
        return 'Manc'
if __name__=='__main__':
    from torch_geometric.loader import NeighborLoader
    from torch_geometric.datasets import Planetoid
    p='/data3/synapse/Synapses/Manc'
    data=Manc(p)
    a=1
    