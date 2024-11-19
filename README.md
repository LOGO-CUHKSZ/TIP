# Boosting Graph Pooling with Persistent Homology

This is the repo for our paper: [Boosting Graph Pooling with Persistent Homology](https://arxiv.org/abs/2402.16346) presented at NeurlPS 2024.

## Setup
Our code can run smoothly on Ubuntu 20.04. The basic environment settings are as follows:
``` Bash
Python=3.7
PyTorch=1.12.1+cu113
torch_geometric=2.3.1
gudhi=3.8.0
GraphRicciCurvature=0.5.3.1
ogb=1.2.1
networkx=2.6.3
```
Then setup the  environment using the following command:
```python
python setup.py build_ext --inplace
```

## How to run
```python
python main_diffpool_tip.py --device=cuda:0
```


## Citation
```commandline
@inproceedings{
ying2024boosting,
title={Boosting Graph Pooling with Persistent Homology},
author={Chaolong Ying and Xinjian Zhao and Tianshu Yu},
booktitle={The Thirty-eighth Annual Conference on Neural Information Processing Systems},
year={2024},
url={https://openreview.net/forum?id=WcmqdY2AKu}
}
```