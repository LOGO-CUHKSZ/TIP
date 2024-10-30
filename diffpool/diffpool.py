from math import ceil

import torch
from ogb.graphproppred.mol_encoder import AtomEncoder
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import DenseSAGEConv
from torch_geometric.utils import to_dense_batch, to_dense_adj, dense_to_sparse, dropout_adj
from torch_geometric.nn import DenseGraphConv, graclus
from utils import fetch_assign_matrix, GCNConv, get_persistence_homology_distance, get_ph_distance, dense_diff_pool
from torch.nn.functional import gumbel_softmax
from torch_geometric.data import Data, Batch
import numpy as np
from topolayer.models import TopologyLayer
import gudhi as gd
import matplotlib.pyplot as plt
import networkx as nx

NUM_SAGE_LAYERS = 3

class SAGEConvolutions(nn.Module):
    def __init__(self,
                 in_channels,
                 hidden_channels,
                 out_channels,
                 lin=True,
                 use_bn=True):
        super().__init__()
        self.use_bn = use_bn
        normalize = True
        self.conv1 = DenseSAGEConv(in_channels, hidden_channels, normalize)
        self.conv2 = DenseSAGEConv(hidden_channels, hidden_channels, normalize)
        self.conv3 = DenseSAGEConv(hidden_channels, out_channels, normalize)

        
        if self.use_bn:
            self.bn1 = nn.BatchNorm1d(hidden_channels)
            self.bn2 = nn.BatchNorm1d(hidden_channels)
        if lin is True:
            self.lin = nn.Linear((NUM_SAGE_LAYERS - 1) * hidden_channels + out_channels, out_channels)
        else:
            # GNN's intermediate representation is given by the concatenation of SAGE layers
            self.lin = None

    def bn(self, i, x):
        batch_size, num_nodes, num_channels = x.size()

        x = x.view(-1, num_channels)
        x = getattr(self, 'bn{}'.format(i))(x)
        x = x.view(batch_size, num_nodes, num_channels)
        return x

    def forward(self, x, adj, mask=None):
        batch_size, num_nodes, in_channels = x.size()

        x0 = x
        if self.use_bn:
            x1 = self.bn(1, F.relu(self.conv1(x0, adj, mask)))
            x2 = self.bn(2, F.relu(self.conv2(x1, adj, mask)))
        else:
            x1 = F.relu(self.conv1(x0, adj, mask))
            x2 = F.relu(self.conv2(x1, adj, mask))
        x3 = self.conv3(x2, adj, mask)

        x = torch.cat([x1, x2, x3], dim=-1)

        # This is used by GNN_pool
        if self.lin is not None:
            x = self.lin(x)

        return x
    

class DiffPoolLayer(nn.Module):

    def __init__(self, dim_input, dim_hidden, dim_embedding, current_num_clusters,
                 no_new_clusters, pooling_type, invariant):

        super().__init__()

        self.invariant = invariant
        self.gnn_pool = SAGEConvolutions(dim_input, dim_hidden, no_new_clusters)
        self.gnn_embed = SAGEConvolutions(dim_input, dim_hidden, dim_embedding, lin=False)


    def forward(self, x, adj, mask=None):


        s = self.gnn_pool(x, adj, mask)         
        x = self.gnn_embed(x, adj, mask)
        x, adj, l, e, s = dense_diff_pool(x, adj, s, mask)
        
        # To make adj sparse using gumbel softmax

        min_val = torch.min(adj.view(adj.shape[0], -1), dim=-1)[0]
        max_val = torch.max(adj.view(adj.shape[0], -1), dim=-1)[0]
        adj = (adj - min_val.unsqueeze(-1).unsqueeze(-1)) / (max_val - min_val).unsqueeze(-1).unsqueeze(-1)
    
        pad = 1 - adj.unsqueeze(-1)
        logits = torch.cat((adj.unsqueeze(-1), pad), dim=-1)
        adj = gumbel_softmax(logits, tau=1, hard=True)[:, :, :, 0]
        
        # symmetric
        adj = torch.triu(adj) + torch.triu(adj).transpose(1, 2)
        
        diagonal_indices = torch.arange(adj.shape[-1])
        # adj[:, diagonal_indices, diagonal_indices] /= 2
        adj[:, diagonal_indices, diagonal_indices] = 1
        
        return x, adj, l, e, s


class DiffPool(nn.Module):

    def __init__(self, num_features, num_classes, max_num_nodes, num_layers, gnn_hidden_dim,
                 gnn_output_dim, mlp_hidden_dim, pooling_type, invariant, encode_edge, pre_sum_aggr=False):
        super().__init__()

        self.encode_edge = encode_edge
        self.pre_sum_aggr = pre_sum_aggr
        self.max_num_nodes = max_num_nodes
        self.pooling_type = pooling_type
        self.num_diffpool_layers = num_layers

        # Reproduce paper choice about coarse factor
        coarse_factor = 0.1 if num_layers == 1 else 0.25

        gnn_dim_input = num_features
        if encode_edge:
            gnn_dim_input = gnn_hidden_dim
            self.conv1 = GCNConv(gnn_hidden_dim, aggr='add')

        if self.pre_sum_aggr:
            self.conv1 = DenseGraphConv(gnn_dim_input, gnn_dim_input)

        no_new_clusters = ceil(coarse_factor * self.max_num_nodes)
        gnn_embed_dim_output = (NUM_SAGE_LAYERS - 1) * gnn_hidden_dim + gnn_output_dim
        
        self.num_coord_funs = 3
        self.num_coord_funs1 = 3
        coord_funs = {"Triangle_transform": self.num_coord_funs,
                        "Gaussian_transform": self.num_coord_funs,
                        "Line_transform": self.num_coord_funs,
                        "RationalHat_transform": self.num_coord_funs
                        }

        coord_funs1 = {"Triangle_transform": self.num_coord_funs1,
                        "Gaussian_transform": self.num_coord_funs1,
                        "Line_transform": self.num_coord_funs1,
                        "RationalHat_transform": self.num_coord_funs1
                        }

        layers = []
        topo_layers = []
        topo_layers.append(TopologyLayer(
                        gnn_dim_input, gnn_hidden_dim, num_filtrations=1,
                        num_coord_funs=coord_funs, filtration_hidden=1,
                        dim1=True, num_coord_funs1=coord_funs1,
                        residual_and_bn=True, swap_bn_order=False,
                        share_filtration_parameters=False, fake=False,
                        tanh_filtrations=True,
                        dist_dim1=False
                        ))
        current_num_clusters = self.max_num_nodes
        for i in range(num_layers):
            diffpool_layer = DiffPoolLayer(gnn_dim_input, gnn_hidden_dim, gnn_output_dim, current_num_clusters,
                                           no_new_clusters, pooling_type, invariant)
            layers.append(diffpool_layer)
            
            # Update embedding sizes
            gnn_dim_input = gnn_embed_dim_output
            current_num_clusters = no_new_clusters
            no_new_clusters = ceil(no_new_clusters * coarse_factor)
            
            topo_layers.append(TopologyLayer(
                        gnn_dim_input, gnn_hidden_dim, num_filtrations=1,
                        num_coord_funs=coord_funs, filtration_hidden=1,
                        dim1=True, num_coord_funs1=coord_funs1,
                        residual_and_bn=True, swap_bn_order=False,
                        share_filtration_parameters=False, fake=False,
                        tanh_filtrations=True,
                        dist_dim1=False
                        ))
        
        
        self.diffpool_layers = nn.ModuleList(layers)
        self.topo_layers = nn.ModuleList(topo_layers)

        self.final_embed = SAGEConvolutions(gnn_embed_dim_output, gnn_hidden_dim, gnn_output_dim, lin=False)
        final_embed_dim_output = gnn_embed_dim_output * (num_layers + 1)

        self.lin1 = nn.Linear(final_embed_dim_output, mlp_hidden_dim)
        self.lin2 = nn.Linear(mlp_hidden_dim, num_classes)
        self.atom_encoder = AtomEncoder(emb_dim=gnn_hidden_dim)

    def forward(self, data, vis=False):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        if self.encode_edge:
            x = self.atom_encoder(x)
            x = self.conv1(x, edge_index, data.edge_attr)

        x, mask = to_dense_batch(x, batch=batch)
        adj = to_dense_adj(edge_index, batch=batch)

        if self.pre_sum_aggr:
            x = self.conv1(x, adj, mask)

        x_all, l_total, e_total = [], 0, 0
        ph_loss = 0
        vis_data = []

        ori_phs, ori_graph_activations1 = self.topo_layers[0](data.x, data)
        ori_adj = adj.clone()
        
        for i in range(self.num_diffpool_layers):
            if i != 0:
                mask = None

            x, adj, l, e, s = self.diffpool_layers[i](x, adj, mask)  # x has shape (batch, MAX_no_nodes, feature_size)
            x_all.append(torch.max(x, dim=1)[0])
            
            new_adj = adj.clone()
            
            # calculate the loss of ph
            # modified from dense_to_sparse
            edge_index = adj.nonzero().t()
            edge_attr = adj[edge_index[0], edge_index[1], edge_index[2]]
            row = edge_index[1]
            col = edge_index[2]
            tmp_edge_index = torch.stack([row, col], dim=0)
            data_list = []
            batch_size = adj.shape[0]
            for t in range(batch_size):
                index = edge_index[0] == t
                data_list.append(Data(x=x[t], edge_index=tmp_edge_index[:, index], edge_attr=edge_attr[index]))
            data = Batch.from_data_list(data_list)
            new_phs, new_graph_activations1 = self.topo_layers[i+1](data.x, data)
            
            new_edge_attr = (new_phs[1][:, :, 1] - new_phs[1][:, :, 0]).mean(dim=0)
            adj = to_dense_adj(data.edge_index, data.batch, new_edge_attr)   

            # topo loss
            ph_loss += F.mse_loss(new_graph_activations1, ori_graph_activations1, reduction='mean')
            
            # visualize
            if vis:
                vis_data.append((ori_phs, new_phs, ori_adj, adj, s))
                
            l_total += l
            e_total += e

        x = self.final_embed(x, adj)
        x_all.append(torch.max(x, dim=1)[0])

        x = torch.cat(x_all, dim=1)  # shape (batch, feature_size x diffpool layers)
        x = F.relu(self.lin1(x))
        x = self.lin2(x)
        
        return x, l_total, e_total, ph_loss, vis_data
