import torch
import argparse
from torch.optim import Adam
from torch import tensor
from torch.optim.lr_scheduler import ReduceLROnPlateau
import os
import json
from dmon.dmon_pool import DMONPool
from utils import set_seed, get_persistence_homology_distance
from data.datasets import get_data, get_sim_exp_dataset
import gudhi as gd
import matplotlib.pyplot as plt
import networkx as nx
# import wandb
from gudhi.wasserstein.wasserstein import wasserstein_distance

parser = argparse.ArgumentParser(description='DMoNPool - PyTorch Geometric')
parser.add_argument('--seed', type=int, default=123)
parser.add_argument('--logdir', type=str, default='results/dmon', help='Log directory.')
parser.add_argument('--dataset', type=str, default='grid2d')
parser.add_argument('--cleaned', action='store_true', default=False, help='Used to eliminate isomorphisms in IMDB')
parser.add_argument('--save', action='store_true', default=False)
parser.add_argument('--lr', type=float, default=0.001, help='Learning rate.')
parser.add_argument('--batch_size', type=int, default=1, help='Batch size.')
parser.add_argument('--max_epochs', type=int, default=1000, help='Maximum number of epochs to train.')
parser.add_argument('--interval', type=int, default=1, help='Interval for printing train statistics.')
parser.add_argument('--early_stop_patience', type=int, default=50)
parser.add_argument('--lr_decay_patience', type=int, default=10)

# model
parser.add_argument('--pooling_type', type=str, choices=['mlp', 'random'], default='mlp')
parser.add_argument('--num_layers', type=int, default=3)
parser.add_argument('--hidden_dim', type=int, default=64)

args, unknown = parser.parse_known_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
set_seed(args.seed)


args.logdir = f'{args.logdir}/{args.dataset}/{args.pooling_type}/' \
              f'{args.num_layers}_layers/{args.hidden_dim}_dim'

if not os.path.exists(f'{args.logdir}'):
    os.makedirs(f'{args.logdir}')

if not os.path.exists(f'{args.logdir}/models/') and args.save:
    os.makedirs(f'{args.logdir}/models/')

with open(f'{args.logdir}/summary.json', 'w') as f:
    json.dump(args.__dict__, f, indent=2)

A, X, y, data = get_sim_exp_dataset(args.dataset)

model = DMONPool(num_features=10, num_classes=1,
                   max_num_nodes=A.shape[0], hidden=args.hidden_dim,
                   pooling_type=args.pooling_type, num_layers=args.num_layers, encode_edge=False).to(device)

optimizer = Adam(model.parameters(), lr=args.lr)
scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, min_lr=1e-5,
                              patience=args.lr_decay_patience, verbose=True)


train_sup_losses, train_lp_losses, train_entropy_losses = [], [], []
val_sup_losses, val_lp_losses, val_entropy_losses = [], [], []
test_sup_losses, test_lp_losses, test_entropy_losses = [], [], []
val_accuracies, test_accuracies = [], []

epochs_no_improve = 0  # used for early stopping
for epoch in range(1, args.max_epochs + 1):
    model.train()
    optimizer.zero_grad()
    data.to(device)
    out, lp_loss, entropy_loss, ph_loss, save_data = model(data)
    ori_phs, new_phs, ori_adj, adj, s = save_data
    loss = ph_loss
    # loss = wasserstein_distance(ori_phs[1][0], new_phs[1][0], enable_autodiff=True, keep_essential_parts=False)
    loss.backward()
    optimizer.step()
    
    print(f'{epoch:03d}: Train Loss: {loss:.3f}')
    
topo_sim = get_persistence_homology_distance(ori_adj.detach(), adj.detach())
print("topological similarity: %f" % topo_sim)