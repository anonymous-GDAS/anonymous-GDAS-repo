# **Graph Domain Adaptation Sampling: Mitigating Structural Discrepancy for Improved Transferability**

---
This repository contains the author's implementation in PyTorch for the paper "Graph Domain Adaptation Sampling: Mitigating Structural Discrepancy for Improved Transferability", anonymous code for NIPS.
## 0.Clone repository

    git clone https://github.com/anonymous-GDAS/anonymous-GDAS-repo.git
    cd anonymous-GDAS-repo
## 1.Create a conda environment
    conda create -n GDAS python=3.7
    conda activate GDAS
## 2.Install python libraries
    pip install -r requirement.txt
## 3.Data preparation
- Download the datasets <https://github.com/daiquanyu/AdaGCN_TKDE/tree/main/input>
- Place these files under `./data/`

The directory structure should look like


    data/
    |-- acmv9.mat
    |-- citationv1.mat
    |-- dblpv7.mat

## 4.Process
### 4.1 Hyperparameter explain
You can change different settings by several hyperparameters.
Parameters in `[]` are what you can set.

    --source     # source dataset [acm dblp citation]
    --target     # target dataset [acm dblp citation] // different from source
    --aug_ratio # the proportion of sampled subgraph to the overall graph (note that the actual proportion is 1-aug_ratio) [0~1]
    --ep         # set the epoch of training [1 5 10 15 20]
    --bsz        # set the batchsize of training [10 20 30 40 50]
    --type       # type of the GNN encoder [gcn gin sage]
### 4.2 Baselines
    cd baseline
    # run CDAN
    python UDAGCN_baseline_CDAN.py
    # run DANN
    python UDAGCN_baseline_DANN.py
    # run mmd
    python UDAGCN_baseline_mmd.py

### 4.3 GDAS 
    # simply run
    python main.py --source acm --target dblp --ep 5 --bsz 10