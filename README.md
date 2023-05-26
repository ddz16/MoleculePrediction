# MoleculePrediction
 This repository contains the code of the downstream task (molecule property prediction) in the paper "Natural Language-informed Understanding of Molecule Graphs”



# Environment

Our environment:
```
Ubuntu 16.04.7 LTS
gcc version: 5.4.0
GPU: NVIDIA TITAN RTX 24GB
NVIDIA-SMI 455.45.01
Driver Version: 455.45.01
CUDA Version: 11.1
```

Based on anaconda or miniconda, you can install the required packages as follows:
```
# python 3.9
conda create --name MoleculePrediction python=3.9
conda activate MoleculePrediction

# pytorch 1.8.1
pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html

# torch_geometric 
# you can download the following *.whl files in https://data.pyg.org/whl/
pip install https://data.pyg.org/whl/torch-1.8.0%2Bcu111/torch_cluster-1.5.9-cp39-cp39-linux_x86_64.whl
pip install https://data.pyg.org/whl/torch-1.8.0%2Bcu111/torch_scatter-2.0.7-cp39-cp39-linux_x86_64.whl
pip install https://data.pyg.org/whl/torch-1.8.0%2Bcu111/torch_sparse-0.6.11-cp39-cp39-linux_x86_64.whl
pip install torch-geometric==1.7.2

# other packages
pip install pandas
pip install rdkit-pypi
pip install tqdm
pip install tensorboardx
pip install networkx
```

# Datasets

```
wget http://snap.stanford.edu/gnn-pretrain/data/chem_dataset.zip
unzip chem_dataset.zip
```


# Our Pretrained models

To apply MoMu, we use the graph encoder in the pre-trained MoMu-S and MoMu-K as the initialization, respectively. You can download them on [the Baidu Netdisk](https://pan.baidu.com/s/1jvMP_ysQGTMd_2sTLUD45A), the password is **1234**. We then fine-tune the graph encoder on the training sets of these datasets for predicting molecular properties, respectively.

MoMu-K checkpoint:

```
checkpoints/littlegin=graphclinit_bert=kvplm_epoch=299-step=18300.ckpt
```

MoMu-S checkpoint:

```
checkpoints/littlegin=graphclinit_bert=scibert_epoch=299-step=18300.ckpt
```

After downloading, you should put these two checkpoints into the `MoMu_checkpoints/` folder.


# Reproduce results

We provide 16 model checkpoints in the `GIN_checkpoints/` folder: **2 model initializations** (MoMu-K, MoMu-S) finetuned on the train sets of **8 datasets** (bbbp, tox21, toxcast, sider, clintox, muv, hiv, bace):

```
bbbp_MoMu-K.pth,       bbbp_MoMu-S.pth
tox21_MoMu-K.pth,      tox21_MoMu-S.pth
toxcast_MoMu-K.pth,    toxcast_MoMu-S.pth
sider_MoMu-K.pth,      sider_MoMu-S.pth
clintox_MoMu-K.pth,    clintox_MoMu-S.pth
muv_MoMu-K.pth,        muv_MoMu-S.pth
hiv_MoMu-K.pth,        hiv_MoMu-S.pth
bace_MoMu-K.pth,       bace_MoMu-S.pth
```

Thus, you can directly use these checkpoints to evaluate the finetuned gnn model on the test sets of **8 datasets**. Here are the scripts for evaluation and the reproduced results (results will be written into the `result.log` file):

**MoMu-K Script**
```
./predict_MoMu-K.sh
```
**MoMu-K Results**
```
bbbp 0.7133487654320989
tox21 0.7612348746538177
toxcast 0.6340565073924614
sider 0.6141872557091731
clintox 0.7979885656692038
muv 0.7023955217302135
hiv 0.7644469765735145
bace 0.7706485828551556
```

**MoMu-S Script**
```
./predict_MoMu-S.sh
```
**MoMu-S Results**
```
bbbp 0.7119984567901235
tox21 0.7549320642441969
toxcast 0.6403441989756496
sider 0.6065912378558332
clintox 0.8032660827859452
muv 0.697392886740407
hiv 0.741512968577995
bace 0.7788210745957224
```

# Finetuning

If you want to initialize the gnn model with the MoMu-S and MoMu-K we provide, and finetune the gnn model on the training set of 8 datasets, you can run the following scripts:

**Finetune on MoMu-K**:

```
./finetune_MoMu-K.sh
```

**Finetune on MoMu-S**:

```
./finetune_MoMu-S.sh
```

**Finetuning Example**:

Finetune MoMu-K on the muv dataset.
```
MoleculeDataset(93087)
scaffold
Data(x=[15, 2], edge_attr=[30, 2], y=[17], edge_index=[2, 30], id=[1])

Iteration:   0%|          | 0/2328 [00:00<?, ?it/s]
Adam (
Parameter Group 0
    amsgrad: False
    betas: (0.9, 0.999)
    eps: 1e-08
    lr: 0.001
    weight_decay: 0

Parameter Group 1
    amsgrad: False
    betas: (0.9, 0.999)
    eps: 1e-08
    lr: 0.001
    weight_decay: 0
)
====epoch 1

Iteration:   0%|          | 1/2328 [00:00<14:27,  2.68it/s]
Iteration:   0%|          | 3/2328 [00:00<10:55,  3.55it/s]
Iteration:   0%|          | 6/2328 [00:00<08:18,  4.66it/s]
Iteration:   0%|          | 8/2328 [00:00<06:25,  6.02it/s]
Iteration:   0%|          | 10/2328 [00:00<05:08,  7.51it/s]
Iteration:   1%|          | 12/2328 [00:01<04:29,  8.59it/s]
Iteration:   1%|          | 14/2328 [00:01<04:01,  9.56it/s]
Iteration:   1%|          | 16/2328 [00:01<03:45, 10.23it/s]
Iteration:   1%|          | 18/2328 [00:01<03:32, 10.86it/s]
Iteration:   1%|          | 20/2328 [00:01<03:25, 11.21it/s]
Iteration:   1%|          | 22/2328 [00:01<03:20, 11.50it/s]
Iteration:   1%|          | 24/2328 [00:02<03:16, 11.73it/s]
Iteration:   1%|          | 26/2328 [00:02<03:07, 12.27it/s]
Iteration:   1%|          | 28/2328 [00:02<03:09, 12.15it/s]
Iteration:   1%|▏         | 30/2328 [00:02<03:14, 11.82it/s]
Iteration:   1%|▏         | 32/2328 [00:02<03:15, 11.73it/s]
Iteration:   1%|▏         | 34/2328 [00:02<03:17, 11.63it/s]
Iteration:   2%|▏         | 36/2328 [00:03<03:15, 11.73it/s]
Iteration:   2%|▏         | 38/2328 [00:03<03:10, 12.03it/s]
Iteration:   2%|▏         | 40/2328 [00:03<03:07, 12.19it/s]
Iteration:   2%|▏         | 42/2328 [00:03<03:12, 11.88it/s]
Iteration:   2%|▏         | 44/2328 [00:03<03:08, 12.09it/s]
Iteration:   2%|▏         | 46/2328 [00:03<03:02, 12.50it/s]
Iteration:   2%|▏         | 48/2328 [00:04<03:05, 12.28it/s]
Iteration:   2%|▏         | 50/2328 [00:04<02:54, 13.09it/s]
...
...
```


# Acknowledgment

We adapted the code of the PyTorch implementation of [GraphCL](https://github.com/Shen-Lab/GraphCL/tree/master/transferLearning_MoleculeNet_PPI/chem). Thanks to the original authors for their work!

# Citation

```
@article{su2022molecular,
  title={Natural Language-informed Understanding of Molecule Graphs},
  author={Bing Su, Dazhao Du, Zhao Yang, Yujie Zhou, Jiangmeng Li, Anyi Rao, Hao Sun, Zhiwu Lu, Ji-Rong Wen},
  year={2022}
}
```
