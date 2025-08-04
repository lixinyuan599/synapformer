# Synapformer

## 1. Brief Introduction
We design an SOE encoder, Synapformer. The proposed architec-
ture achieves collaborative learning between neuron-level connectivity
and synatic connections, enabling fine-grained representations of neu-
rons in a neural circuits. 
## 2. Installation
* Linux
* CUDA environment
* Python >= 3.9
* NVIDIA CUDA Compiler Driver NVCC version >= 10.0. This is used for compiling the dependencies of torch_geometric: torch-cluster,torch-sparse,torch-scatter.
* [Pytorch](https://pytorch.org/) >= 2.0
* [Pytorch Geometric](https://pyg.org/) >= 2.3
## 3. Our main dependencies
```
torch                     2.3.0+cu121
torch_cluster             1.6.3+pt23cu121
torch_geometric           2.5.3
torch_scatter             2.1.2+pt23cu121
torch_sparse              0.6.18+pt23cu121
torch_spline_conv         1.2.2+pt23cu121
torchaudio                2.3.0+cu121
torchvision               0.18.0+cu121
pyg-lib                   0.4.0+pt23cu121
```
### Step 1
Please make sure that the PyTorch Geometric (PyG) version you download is compatible with the PyTorch and CUDA versions you are using.You can check the official website of [PyG](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html).
### Step 2
Ensure that nvcc is accessible from terminal and check CUDA version:
```
nvcc --version
>>>NVIDIA (R) Cuda compiler driver Copyright (c) 2005-2023 NVIDIA Corporation Built on Tue_Feb__7_19:32:13_PST_2023 Cuda compilation tools, release 12.1, V12.1.66 Build cuda_12.1.r12.1/compiler.32415258_0
```
```
nvidia-smi |grep Version
>>>CUDA Version: 12.1
```
### Step 3
Download synapses data:
```
python fetch_hemibrain_synapse.py
python fetch_manc_synapse.py
```
### Step 4
Download connectome data:
```
unzip raw.zip -d /path/to/destination/
```
### Step 4
```
python main.py
```
## Code structure:
```
Source Code
├── data
|   ├──download_synapses
|   ├──HemiBrain
|   ├──Manc
|   ├──hemibrain.py
|   └──manc.py
├── src
|   ├──models
|   |   ├──connectome_enco.py
|   |   ├──synapse_enco.py
|   |   ├──model.py
|   |   └──NeuronLearner.py
|   ├──utils
|   |   ├──config.py
|   |   ├──datahandler.py
|   |   ├──test.py
|   |   └──train.py
|   ├──classification_main.py
|   └──transmitter_main.py
└── requirements.txt
```
