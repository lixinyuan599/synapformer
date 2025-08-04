import navis.interfaces.neuprint as neu
from neuprint import NeuronCriteria as NC
from neuprint.queries import fetch_all_rois,fetch_neurons,fetch_primary_rois, fetch_adjacencies
import neuprint
from neuprint import Client
import pandas as pd

client = neu.Client(
    "https://neuprint.janelia.org/",
    token=token,
    dataset="manc:v1.2.1",
)

neurons=[]
with open('/data3/synapse/MICCAI/src/data/Manc/raw/manc_class_info/neuron2ID.txt', 'r') as file:
    for line in file:
        columns = line.strip().split(' ')
        if columns:
            first_column = columns[0]
            neurons.append(int(first_column))
            
    print(len(neurons))

from neuprint import fetch_synapse_connections, SynapseCriteria as SC
from tqdm import tqdm
import os
for neuron in tqdm(neurons,desc="Processing neurons"):
    path=f'/data3/synapse/MICCAI/src/data/Manc/synapses/{neuron}.csv'
    if os.path.exists(path):
        pass
    else:
        connect = fetch_synapse_connections(neuron,None,batch_size=3000,client=client)

        df = pd.DataFrame(connect)
        df.to_csv(f'/data3/synapse/MICCAI/src/data/Manc/synapses/{neuron}.csv', index=False)
print('down!!!!!!!!!!!!!!!!!')