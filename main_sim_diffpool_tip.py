import torch
import argparse
from torch.optim import Adam
from torch import tensor
from torch.optim.lr_scheduler import ReduceLROnPlateau
import os
import json
from diffpool.train import train, evaluate, train_regression, evaluate_regression
from utils import set_seed, compute_ricci_curvature, spectral_similarity, get_persistence_homology_distance
from data.datasets import get_sim_exp_dataset
import gudhi as gd
import matplotlib.pyplot as plt
import networkx as nx

# import wandb
from gudhi.wasserstein.wasserstein import wasserstein_distance

parser = argparse.ArgumentParser(description='DiffPool - PyTorch Geometric.')
parser.add_argument('--seed', type=int, default=123, help='Random seed.')
parser.add_argument('--logdir', type=str, default='results/diffpool', help='Log directory')
parser.add_argument('--dataset', type=str, default='torus')
parser.add_argument('--reproduce', action='store_true', default=False)
parser.add_argument('--cleaned', action='store_true', default=False, help='Used to eliminate isomorphisms in IMDB')
parser.add_argument('--save', action='store_true', default=False)
parser.add_argument('--lr', type=float, default=0.001, help='Learning rate.')
parser.add_argument('--batch_size', type=int, default=20, help='Batch size.')
parser.add_argument('--max_epochs', type=int, default=100, help='Number of epochs to train.')
parser.add_argument('--interval', type=int, default=1, help='Interval for printing train statistics.')
parser.add_argument('--early_stop_patience', type=int, default=50)
parser.add_argument('--lr_decay_patience', type=int, default=10)
parser.add_argument('--use_feature', action='store_true', default=True) # This is only implemented in TUDataset
parser.add_argument('--wandb', action='store_true', default=False)
# model
parser.add_argument('--pooling_type', type=str, default='gnn',
                    choices=['gnn', 'bernoulli', 'categorical', 'normal', 'uniform'])
parser.add_argument('--invariant', action='store_true', default=False, help='Invariant version of random pooling')
parser.add_argument('--gnn_dim', type=int, default=32)
parser.add_argument('--mlp_hidden_dim', type=int, default=50)
parser.add_argument('--num_pooling_layers', type=int, default=1)
# args = parser.parse_args()
args, unknown = parser.parse_known_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
set_seed(args.seed)


from diffpool.diffpool import DiffPool
args.logdir = args.logdir


args.logdir = f'{args.logdir}/{args.dataset}/{args.pooling_type}/' \
              f'{args.num_pooling_layers}_layers/{args.gnn_dim}_dim'

if not os.path.exists(f'{args.logdir}'):
    os.makedirs(f'{args.logdir}')

if not os.path.exists(f'{args.logdir}/models/') and args.save:
    os.makedirs(f'{args.logdir}/models/')

with open(f'{args.logdir}/summary.json', 'w') as f:
    json.dump(args.__dict__, f, indent=2)
    

A, X, y, data = get_sim_exp_dataset(args.dataset)


model = DiffPool(num_features=10, num_classes=1,
                 max_num_nodes=A.shape[0], num_layers=args.num_pooling_layers,
                 gnn_hidden_dim=args.gnn_dim, gnn_output_dim=args.gnn_dim,
                 mlp_hidden_dim=args.mlp_hidden_dim, pooling_type=args.pooling_type,
                 invariant=args.invariant, encode_edge=False, pre_sum_aggr=args.dataset=='IMDB-BINARY').to(device)

optimizer = Adam(model.parameters(), lr=args.lr)
scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, min_lr=1e-5,
                              patience=args.lr_decay_patience, verbose=True)


train_sup_losses, train_lp_losses, train_entropy_losses = [], [], []
val_sup_losses, val_lp_losses, val_entropy_losses = [], [], []
test_sup_losses, test_lp_losses, test_entropy_losses = [], [], []
val_accuracies, test_accuracies = [], []

epochs_no_improve = 0  # used for early stopping
best_val_acc = 0
best_test_acc = 0
for epoch in range(0, args.max_epochs):
    model.train()
    optimizer.zero_grad()
    data.to(device)
    out, lp_loss, entropy_loss, ph_loss, vis_data = model(data, vis=True)
    ori_phs, new_phs, ori_adj, adj, s = vis_data[0]
    # loss = wasserstein_distance(ori_phs[1][0], new_phs[1][0], enable_autodiff=True, keep_essential_parts=False)
    loss = ph_loss
    loss.backward()
    optimizer.step()
    
    print(f'{epoch:03d}: Train Loss: {loss:.3f}')
    

topo_sim = get_persistence_homology_distance(ori_adj.detach(), adj.detach())
print("topological similarity: %f" % topo_sim)

