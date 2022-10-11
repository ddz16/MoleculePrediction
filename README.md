# MoleculePrediction
 This repository contains the code of the downstream task (molecule property prediction) in the paper "Natural Language-informed Understanding of Molecule Graphs”

# Acknowledgment

We adapted the code of the PyTorch implementation of [GraphCL](https://github.com/Shen-Lab/GraphCL/tree/master/transferLearning_MoleculeNet_PPI/chem). Thanks to the original authors for their work!

# Dependencies & Dataset

Please refer to https://github.com/snap-stanford/pretrain-gnns#installation for environment setup and https://github.com/snap-stanford/pretrain-gnns#dataset-download to download dataset.

If you cannot manage to install the old torch-geometric version, one alternative way is to use the new one (maybe ==1.6.0) and make some modifications based on this issue snap-stanford/pretrain-gnns#14. This might leads to some inconsistent results with those in the paper.

# Our Pretrained models

To apply MoMu, we use the graph encoder in the pre-trained MoMu-S and MuMu-K as the initialization, respectively. You can download them on [the Baidu Netdisk](https://pan.baidu.com/s/1jvMP_ysQGTMd_2sTLUD45A), the password is **1234**. We then fine-tune the graph encoder on the training sets of these datasets for predicting molecular properties, respectively.

MuMu-K checkpoint:

```
checkpoints/littlegin=graphclinit_bert=kvplm_epoch=299-step=18300.ckpt
```

MuMu-S checkpoint:

```
checkpoints/littlegin=graphclinit_bert=scibert_epoch=299-step=18300.ckpt
```

After downloading, you should put these two checkpoints into the `checkpoints` folder.

# Finetuning
Finetune on MoMu-K:

```
./finetune_MuMu-K.sh
```

Finetune on MoMu-S:

```
./finetune_MuMu-S.sh
```

Results will be recorded in `result.log`.

# Sample Result

Finetune MuMu-K on the muv dataset

Finetune process:
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

Prediction results:
```
muv 0 0.686400937903898
muv 1 0.7208213502256825
muv 2 0.6904045632349599
muv 3 0.6867226222897985
muv 4 0.746586288785911
muv 5 0.6702708041353815
muv 6 0.7474960366735247
muv 7 0.7077071846386775
muv 8 0.7278327257241599
muv 9 0.72664916270257
```
# Citation

```
@article{su2022molecular,
  title={Natural Language-informed Understanding of Molecule Graphs},
  author={Bing Su, Dazhao Du, Zhao Yang, Yujie Zhou, Jiangmeng Li, Anyi Rao, Hao Sun, Zhiwu Lu, Ji-Rong Wen},
  year={2022}
}
```