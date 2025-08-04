from hashlib import new
from typing import Optional, Callable, List
import os.path as osp
import numpy as np
import torch
import torch_geometric
from torch_geometric.data import InMemoryDataset, download_url
from torch_geometric.data import Data
import json
import pandas as pd
import glob
import csv
import re
import pdb
from tqdm import tqdm
import pickle

class HemiBrainSynapse(InMemoryDataset):
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
    

    def __init__(self, root: str, transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None,process=False,grain='neuclass',restore_split=None,device=None):


        
        GRAIN_DICT={'neuclass':'Neu_class','neuropil':'Neuropil'}
        
        assert grain.lower() in GRAIN_DICT
        
        self.grain=GRAIN_DICT[grain.lower()]
        
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
        return osp.join(self.root, 'Processed')

    @property
    def raw_file_names(self) -> List[str]:
        return ['edgeroi2label.json','neuron_connection.txt','neuron2ROI.json','Hemibrain_grained_class.json']
    @property
    def processed_file_names(self) -> str:
        return 'data.pt'
    @property
    def neu2ID(self):
        if hasattr(self,'neu2ID_dict'):
            return self.neu2ID_dict
        else:
            neu2ID={}
            with open(osp.join(self.raw_dir,'neuron2ID.txt'),'r') as fin:
                for i in fin.readlines():
                    line=i.strip().split(',')
                    neu2ID[line[0]]=int(line[1])
            self.neu2ID_dict=neu2ID
            return self.neu2ID_dict
    @property
    def label2ID(self):
        if hasattr(self,'label2ID_dict'):
            return self.label2ID_dict
        else:

            self.label2ID_dict={'TBD':0}
            label_index=1
            self.neuron2label
            for l in self.neuron2label.values():
                if l !='TBD' and l not in self.label2ID_dict:
                    self.label2ID_dict[l]=label_index
                    label_index+=1
            return self.label2ID_dict
    @property
    def neuron2label(self):
        if hasattr(self,'neuron2label_dict'):
            return self.neuron2label_dict
        else:
            self.neuron2label_dict={}
            with open(osp.join(self.raw_dir,'Hemibrain_grained_class.json'),'r') as fin:
                finegrainclass=json.load(fin)
            for neu in self.neu2ID:
                if neu in finegrainclass:
                    if self.grain in finegrainclass[neu]:
                        self.neuron2label_dict[neu]=finegrainclass[neu][self.grain]
                    else:
                        self.neuron2label_dict[neu]='TBD'
                else:
                    self.neuron2label_dict[neu]='TBD'
            return self.neuron2label_dict
    def process(self):

        self.neu2ID
        self.neuron2label
        ID2neu={v:k for k,v in self.neu2ID.items()}
        edge_index=[]
        edge_attr=[]
        edge_weight=[]
        with open(osp.join(self.raw_dir,'edgeroi2ID.json'),'r') as fin:
            edgeroi2ID=json.load(fin)
        with open(osp.join(self.raw_dir,'neuron_connection.txt'),'r') as fin:
            for i in fin.readlines():
                line=i.strip().split(',')
                if len(line)==4:
                    edge_index.append([self.neu2ID[line[0]],self.neu2ID[line[1]]])
                    edge_attr.append(edgeroi2ID[line[2]])
                    edge_weight.append(int(line[3]))
                    
        edge_index=torch.Tensor(edge_index)
        edge_index=torch.transpose(edge_index,0,1)
        edge_index=edge_index.to(dtype=torch.int64)
        edge_index=edge_index.contiguous()
        edge_attr=torch.Tensor(edge_attr)
        self.label2ID

        y=[]
        labelname=[]
        neuronname=[]
        synapse=[]
        synapse_index=[]
        

        for i_neuron in range(len(ID2neu)):
            nname=ID2neu[i_neuron]
            label_x=self.neuron2label[nname]
            neuronname.append(nname)
            labelname.append(label_x)
            y.append(self.label2ID[label_x])
        y=torch.tensor(y)




    ### 创建 csv ———> dict
        sysnase_connection_dict = {}
        # index_dict={}
        # count=0
        csv_files=glob.glob('/data3/lixinyuan/synapse/Synapses/HemiBrain/synapses/*.csv')
        
        for csv_path in tqdm(csv_files):
            neu_name=osp.basename(csv_path)
            neu_name=neu_name.replace('.csv','')
            
            df = pd.read_csv(csv_path)
            
            
            for i in range(len(df)):
                
                row=df.loc[i]
                pre_neu=str(row['bodyId_pre'])
                post_neu=str(row['bodyId_post'])
                if pre_neu and post_neu in self.neu2ID.keys():
                    pre_neu=int(self.neu2ID[pre_neu])
                    post_neu=int(self.neu2ID[post_neu])

                    cordi=row[['x_pre','y_pre','z_pre','x_post','y_post','z_post']].to_numpy()
                    cordi = cordi.astype(np.int64)  # 或 np.int64，取决于需要的类型
                    cordi = torch.tensor(cordi, dtype=torch.int64)
                    
                    if (pre_neu,post_neu) in sysnase_connection_dict:
                        sysnase_connection_dict[(pre_neu,post_neu)].append(cordi)
                    else:
                        sysnase_connection_dict[(pre_neu,post_neu)]=[cordi]

        with open ("/data3/lixinyuan/synapse/Synapses/HemiBrain/HemiBrain_sysnase_connection_dict111.pkl" , "wb") as file:
            pickle.dump(sysnase_connection_dict , file)
        print("down!!!!!!!!!!!!!!!")


    ### 创建 synapse 和 synapse index      
        
        
        synapse=[]
        synapse_index=[]
        synapse_id=0

    
        #count=0
        for i in tqdm(range(edge_index.size(1))):
            
            neu1,neu2=int(edge_index[:,i][0]),int(edge_index[:,i][1])
            
            if (neu1,neu2) in sysnase_connection_dict:
                synapse.append(sysnase_connection_dict[(neu1,neu2)])
                #synapse_index.append(synapse_index[-1]+len(sysnase_connection_dict[(neu1,neu2)]))
                synapse_index.append([synapse_id]*len(sysnase_connection_dict[(neu1,neu2)]))
                synapse_id+=1
            else:
                data_list = [0, 0, 0, 0, 0, 0]
                tensor_data = torch.tensor(data_list)
                synapse.append([tensor_data])
                #synapse_index.append(synapse_index[-1]+1)
                synapse_index.append([synapse_id])
                synapse_id+=1
            # count+=1
            # if count>50:
            #     break
            
        synapse=np.concatenate(synapse,axis=0)
        synapse=torch.from_numpy(synapse)
        
        synapse_index = torch.cat([torch.tensor(index) for index in synapse_index])
        synapse_index=torch.tensor(synapse_index)    

        


        index = torch.randperm(len(y)).tolist() #torch.randperm(len(y)) 生成一个从 0 到 len(y)-1 的随机排列的整数序列。
        train_index = index[:len(y)//10*8] #len(y)//10*8 计算出 y 长度的 80%（整数除法）
        test_index = index[len(y)//10*8:len(y)//10*9]
        val_index = index[len(y)//10*9:]

        train_mask = torch.zeros((len(y), ), dtype=torch.bool)
        val_mask = torch.zeros((len(y),), dtype=torch.bool)
        test_mask = torch.zeros((len(y), ), dtype=torch.bool)
        train_mask[train_index] = True
        val_mask[val_index] = True
        test_mask[test_index] = True

        #data = Data(edge_index=edge_index,y=y,labelnames=labelname,names=neuronname,edge_attr=edge_attr)
        data = Data(edge_index=edge_index,y=y,labelnames=labelname,names=neuronname,edge_attr=edge_attr,synapse=synapse,synapse_index=synapse_index)
        
    
        data.train_mask = train_mask
        data.val_mask = val_mask
        data.test_mask = test_mask
        
        # data.train_index=train_index
        # data.test_index=test_index
        # data.val_index=val_index

        data = data if self.pre_transform is None else self.pre_transform(data)
        torch.save(self.collate([data]), self.processed_paths[0])
    def __repr__(self) -> str:
        return 'HemiBrain'



if __name__=='__main__':
    from torch_geometric.loader import NeighborLoader
    from torch_geometric.datasets import Planetoid
    p='/data3/lixinyuan/synapse/Synapses/HemiBrain'
    data=HemiBrainSynapse(p)
    a=1
    