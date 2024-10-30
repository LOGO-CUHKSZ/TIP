from math import ceil

import torch
import torch.nn.functional as F
from torch.nn import Linear
import torch.nn as nn
from torch_geometric.nn import DenseGraphConv, GraphConv
from torch_geometric.utils import to_dense_batch, to_dense_adj
from dmon.dmon_pool_mod import dense_dmon_pool
from utils import fetch_assign_matrix
from ogb.graphproppred.mol_encoder import AtomEncoder
from utils import GCNConv
from topolayer.models import TopologyLayer
from torch_geometric.data import Data, Batch
from gudhi.wasserstein.wasserstein import wasserstein_distance


class DMONPool(torch.nn.Module):
    def __init__(self, num_features, num_classes, max_num_nodes, hidden, pooling_type,
                 num_layers, encode_edge=False):
        super(DMONPool, self).__init__()
        self.encode_edge = encode_edge

        self.atom_encoder = AtomEncoder(emb_dim=hidden)

        self.pooling_type = pooling_type
        self.convs = nn.ModuleList()
        self.pools = nn.ModuleList()
        self.num_layers = num_layers
        
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
        topo_layers = []
        
        
        for i in range(num_layers):
            if i == 0:
                if encode_edge:
                    self.convs.append(GCNConv(hidden, aggr='add'))
                else:
                    self.convs.append(GraphConv(num_features, hidden, aggr='add'))
                    topo_layers.append(TopologyLayer(
                        num_features, num_features, num_filtrations=1,
                        num_coord_funs=coord_funs, filtration_hidden=24,
                        dim1=True, num_coord_funs1=coord_funs1,
                        residual_and_bn=True, swap_bn_order=False,
                        share_filtration_parameters=False, fake=False,
                        tanh_filtrations=True,
                        dist_dim1=False
                        ))
            else:
                self.convs.append(DenseGraphConv(hidden, hidden))
                topo_layers.append(TopologyLayer(
                        hidden, hidden, num_filtrations=1,
                        num_coord_funs=coord_funs, filtration_hidden=24,
                        dim1=True, num_coord_funs1=coord_funs1,
                        residual_and_bn=True, swap_bn_order=False,
                        share_filtration_parameters=False, fake=False,
                        tanh_filtrations=True,
                        dist_dim1=False
                        ))

        self.rms = []
        num_nodes = max_num_nodes
        for i in range(num_layers - 1):
            num_nodes = ceil(0.25 * num_nodes)
            if pooling_type == 'mlp':
                self.pools.append(Linear(hidden, num_nodes))
            else:
                self.rms.append(fetch_assign_matrix('uniform', ceil(2 * num_nodes), num_nodes))

        final_embed_dim_output = hidden * (num_layers)
        self.lin1 = Linear(final_embed_dim_output, hidden)
        self.lin2 = Linear(hidden, num_classes)
        
        
        
        self.topo_layers = nn.ModuleList(topo_layers)

    def forward(self, data, vis=False):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        if self.encode_edge:
            x = self.atom_encoder(x)
            x = F.relu(self.convs[0](x, edge_index, data.edge_attr))
        else:
            x = F.relu(self.convs[0](x, edge_index))

        x, mask = to_dense_batch(x, batch)
        adj = to_dense_adj(edge_index, batch)
        ori_phs, ori_graph_activations1 = self.topo_layers[0](data.x, data)
        ori_adj = adj.clone()
        s_all = []
        adjs = []
        
        x_all = []
        # original graph ph
        ph_loss = 0        

        if self.pooling_type != 'mlp':
            s = self.rms[0][:x.size(1), :].unsqueeze(dim=0).expand(x.size(0), -1, -1).to(x.device)
        else:
            s = self.pools[0](x)
            s_all.append(s)

        x, adj, sl, ol, cl = dense_dmon_pool(x, adj, s, mask)
        x_all.append(torch.max(x, dim=1)[0])
        
        adjs.append(adj)

        for i in range(1, self.num_layers - 1):
            x = F.relu(self.convs[i](x, adj))
            if self.pooling_type != 'mlp':
                s = self.rms[i][:x.size(1), :].unsqueeze(dim=0).expand(x.size(0), -1, -1).to(x.device)
            else:
                s = self.pools[i](x)
            s_all.append(s)
            x, adj, sl_aux, ol_aux, cl_aux = dense_dmon_pool(x, adj, s)
            sl += sl_aux
            ol += ol_aux
            cl += cl_aux
            
            adjs.append(adj)

            x_all.append(torch.max(x, dim=1)[0])
            
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
            new_phs, new_graph_activations1 = self.topo_layers[i](data.x, data)
            
            # further modify adj
            new_edge_attr = (new_phs[1][:, :, 1] - new_phs[1][:, :, 0]).mean(dim=0)
            # thres = nn.Threshold(0.001, 0)
            # new_edge_attr = thres(new_edge_attr)
            adj = to_dense_adj(data.edge_index, data.batch, new_edge_attr)
            # topo loss
            ph_loss += F.mse_loss(new_graph_activations1, ori_graph_activations1, reduction='mean')
            # wasserstein distance
            # ph_loss += wasserstein_distance(ori_phs[1][0], new_phs[1][0], enable_autodiff=True, keep_essential_parts=False)

        x = self.convs[self.num_layers-1](x, adj)
        x_all.append(torch.max(x, dim=1)[0])
        x = torch.cat(x_all, dim=1)
        x = F.relu(self.lin1(x))
        x = self.lin2(x)
        
        if vis:
            save_data = (ori_phs, new_phs, ori_adj, adjs, s_all)
        else:
            save_data = None
        return x, sl, ol+cl, ph_loss, save_data
