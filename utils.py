import numpy as np
import torch
from ogb.graphproppred.mol_encoder import BondEncoder
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import to_dense_adj, unbatch
import copy
import torch.nn.functional as F
import gudhi as gd
import networkx as nx
from GraphRicciCurvature.OllivierRicci import OllivierRicci
from GraphRicciCurvature.FormanRicci import FormanRicci
from gudhi.wasserstein.wasserstein import wasserstein_distance
# from distance import wasserstein_distance
from multiprocessing import Pool
from torch_geometric.utils import degree
import os
from typing import Optional, Tuple
from torch import Tensor
import matplotlib.pyplot as plt

EPS = 1e-15


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def fetch_assign_matrix(random, dim1, dim2, normalize=False):
    if random == 'uniform':
        m = torch.rand(dim1, dim2)
    elif random == 'normal':
        m = torch.randn(dim1, dim2)
    elif random == 'bernoulli':
        m = torch.bernoulli(0.3*torch.ones(dim1, dim2))
    elif random == 'categorical':
        idxs = torch.multinomial((1.0/dim2)*torch.ones((dim1, dim2)), 1)
        m = torch.zeros(dim1, dim2)
        m[torch.arange(dim1), idxs.view(-1)] = 1.0

    if normalize:
        m = m / (m.sum(dim=1, keepdim=True) + EPS)
    return m


def knn_filter(data, k=8):
    dists = torch.cdist(data.pos, data.pos)
    sorted_dists, _ = dists.sort(dim=-1)
    sigmas = sorted_dists[:, 1:k + 1].mean(dim=-1) + 1e-8  # avoid division by 0
    adj = (-(dists / sigmas) ** 2).exp()  # adjacency matrix
    adj = 0.5 * (adj + adj.T)  # gets a symmetric matrix
    adj.fill_diagonal_(0)  # removing self-loops
    knn_values, knn_ids = adj.sort(dim=-1, descending=True)
    adj[torch.arange(adj.size(0)).unsqueeze(1), knn_ids[:, k:]] = 0.0  # selecting the knns. The matrix is not symmetric.
    data.edge_index = adj.nonzero().T
    data.edge_attr = adj[adj > 0].view(-1).unsqueeze(1)
    data.x = torch.cat((data.x, data.pos), dim=1)
    return data


def rwr_filter(data, c=0.1):
    adj = to_dense_adj(data.edge_index).squeeze()
    adj = 0.5*(adj + adj.T)
    adj = adj + torch.eye(adj.shape[0])
    d = torch.diag(torch.sum(adj, 1))
    d_inv = d**(-0.5)
    d_inv[torch.isinf(d_inv)] = 0.
    w_tilda = torch.matmul(d_inv, adj)
    w_tilda = np.matmul(w_tilda, d_inv)
    q = torch.eye(w_tilda.shape[0]) - c * w_tilda
    q_inv = torch.inverse(q)
    rwr = (1 - c) * q_inv
    rwr, _ = torch.sort(rwr, dim=1, descending=True)
    sparse_rwr = rwr.to_sparse()
    data.edge_index = sparse_rwr.indices()
    data.edge_attr = sparse_rwr.values().unsqueeze(1).float()
    return data


def graph_permutation(data):
    d2 = copy.deepcopy(data)
    for i in range(d2.batch.max()+1):
        idx = (d2.batch == i).nonzero().squeeze()
        gsize = idx.shape[0]
        rp = torch.randperm(idx.shape[0])
        idx_perm = idx[rp]
        d2.x[idx_perm, :] = data.x[idx, :]
        for j in range(gsize):
            d2.edge_index[data.edge_index == idx[j]] = idx_perm[j]
    return d2


# code taken from https://github.com/snap-stanford/ogb/blob/master/examples/graphproppred/mol/conv.py
class GCNConv(MessagePassing):
    def __init__(self, emb_dim, aggr):
        super(GCNConv, self).__init__(aggr=aggr)

        self.linear = torch.nn.Linear(emb_dim, emb_dim)
        self.root_emb = torch.nn.Embedding(1, emb_dim)
        self.bond_encoder = BondEncoder(emb_dim = emb_dim)

    def forward(self, x, edge_index, edge_attr):
        x = self.linear(x)
        edge_embedding = self.bond_encoder(edge_attr)
        return self.propagate(edge_index, x=x, edge_attr=edge_embedding) + F.relu(x + self.root_emb.weight)

    def message(self, x_j, edge_attr):
        return F.relu(x_j + edge_attr)

    def update(self, aggr_out):
        return aggr_out


def compute_distance(adj1, adj2):
    dis1 = 0
    # Graph1
    G1 = nx.from_numpy_array(adj1)
    # G_OT1 = OllivierRicci(G1, alpha=0.5, method="Sinkhorn")
    G_OT1 = FormanRicci(G1)
    G_OT1.compute_ricci_curvature()
    st1 = gd.SimplexTree()
    for n1, n2 in G_OT1.G.edges():
        st1.insert(simplex=[n1, n2], filtration=G_OT1.G[n1][n2]['formanCurvature'])
        # st1.insert(simplex=[n1, n2], filtration=G_OT1.G[n1][n2]['ricciCurvature'])
        # st1.insert(simplex=[n1, n2], filtration=max(G1.degree[n1], G1.degree[n2]))
    st1.extend_filtration()
    barcode1 = st1.extended_persistence() # min_persistence=-1.0
    barcode1_dim0 = st1.persistence_intervals_in_dimension(0)
    barcode1_dim1 = st1.persistence_intervals_in_dimension(1)

    # Graph2
    G2 = nx.from_numpy_array(adj2)
    # G_OT2 = OllivierRicci(G2, alpha=0.5, method="Sinkhorn")
    G_OT2 = FormanRicci(G2)
    G_OT2.compute_ricci_curvature()
    st2 = gd.SimplexTree()
    for n1, n2 in G_OT2.G.edges():
        st2.insert(simplex=[n1, n2], filtration=G_OT2.G[n1][n2]['formanCurvature'])
        # st2.insert(simplex=[n1, n2], filtration=G_OT2.G[n1][n2]['ricciCurvature'])
        # st2.insert(simplex=[n1, n2], filtration=max(G2.degree[n1], G2.degree[n2]))
    st2.extend_filtration()
    barcode2 = st2.extended_persistence() # min_persistence=-1.0
    barcode2_dim0 = st2.persistence_intervals_in_dimension(0)
    barcode2_dim1 = st2.persistence_intervals_in_dimension(1)
    
    dis = wasserstein_distance(barcode1_dim1, barcode2_dim1, matching=False, order=1, internal_p=2)
    if dis > 0:
        dis1 += dis
    
    # gd.plot_persistence_diagram(barcode1_dim1)
    # gd.plot_persistence_diagram(barcode2_dim1)
    # plt.show()
        
    return dis1


def get_persistence_homology_distance(adj1, adj2):
    # compute extended persistence in parallel
    
    batch_size, _, _ = adj1.size()
    adj1 = adj1.detach().cpu().numpy()
    adj2 = adj2.detach().cpu().numpy()
    # adj2[adj2<0.1] = 0
    
    
    dis0 = 0
    dis1 = 0
    adj_lists = []
    for i in range(batch_size):
        # np.fill_diagonal(adj2[i], 0)
        adj_lists.append((adj1[i], adj2[i]))
    
    max_cpu = batch_size
    with Pool(max_cpu) as p:
        dis_list = p.starmap(compute_distance, adj_lists)
    dis1 = np.sum(dis_list)

    return dis1
        

def get_ph_distance(ph1, ph2, data1, data2):
    num_views = ph1.shape[0]
    batch_size = data1.num_graphs
    dis1 = 0
    ph1_list = unbatch_edge_features(ph1, data1.edge_index, data1.batch)
    ph2_list = unbatch_edge_features(ph2, data2.edge_index, data2.batch)
    for t in range(batch_size):
        for i in range(num_views):
            dis = wasserstein_distance(ph1_list[t][i], ph2_list[t][i], enable_autodiff=True, keep_essential_parts=False)
            if dis > 0:
                dis1 += dis
    return dis1

def unbatch_edge_features(src, edge_index, batch):
    # modified from unbatch_edge_index
    deg = degree(batch, dtype=torch.int64)
    ptr = torch.cat([deg.new_zeros(1), deg.cumsum(dim=0)[:-1]], dim=0)

    edge_batch = batch[edge_index[0]]
    edge_index = edge_index - ptr[edge_batch]
    sizes = degree(edge_batch, dtype=torch.int64).cpu().tolist()
    
    return src.split(sizes, dim=1)


def dense_diff_pool(
    x: Tensor,
    adj: Tensor,
    s: Tensor,
    mask: Optional[Tensor] = None,
    normalize: bool = True,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    r"""The differentiable pooling operator from the `"Hierarchical Graph
    Representation Learning with Differentiable Pooling"
    <https://arxiv.org/abs/1806.08804>`_ paper

    .. math::
        \mathbf{X}^{\prime} &= {\mathrm{softmax}(\mathbf{S})}^{\top} \cdot
        \mathbf{X}

        \mathbf{A}^{\prime} &= {\mathrm{softmax}(\mathbf{S})}^{\top} \cdot
        \mathbf{A} \cdot \mathrm{softmax}(\mathbf{S})

    based on dense learned assignments :math:`\mathbf{S} \in \mathbb{R}^{B
    \times N \times C}`.
    Returns the pooled node feature matrix, the coarsened adjacency matrix and
    two auxiliary objectives: (1) The link prediction loss

    .. math::
        \mathcal{L}_{LP} = {\| \mathbf{A} -
        \mathrm{softmax}(\mathbf{S}) {\mathrm{softmax}(\mathbf{S})}^{\top}
        \|}_F,

    and (2) the entropy regularization

    .. math::
        \mathcal{L}_E = \frac{1}{N} \sum_{n=1}^N H(\mathbf{S}_n).

    Args:
        x (torch.Tensor): Node feature tensor
            :math:`\mathbf{X} \in \mathbb{R}^{B \times N \times F}`, with
            batch-size :math:`B`, (maximum) number of nodes :math:`N` for
            each graph, and feature dimension :math:`F`.
        adj (torch.Tensor): Adjacency tensor
            :math:`\mathbf{A} \in \mathbb{R}^{B \times N \times N}`.
        s (torch.Tensor): Assignment tensor
            :math:`\mathbf{S} \in \mathbb{R}^{B \times N \times C}`
            with number of clusters :math:`C`.
            The softmax does not have to be applied before-hand, since it is
            executed within this method.
        mask (torch.Tensor, optional): Mask matrix
            :math:`\mathbf{M} \in {\{ 0, 1 \}}^{B \times N}` indicating
            the valid nodes for each graph. (default: :obj:`None`)
        normalize (bool, optional): If set to :obj:`False`, the link
            prediction loss is not divided by :obj:`adj.numel()`.
            (default: :obj:`True`)

    :rtype: (:class:`torch.Tensor`, :class:`torch.Tensor`,
        :class:`torch.Tensor`, :class:`torch.Tensor`)
    """
    x = x.unsqueeze(0) if x.dim() == 2 else x
    adj = adj.unsqueeze(0) if adj.dim() == 2 else adj
    s = s.unsqueeze(0) if s.dim() == 2 else s

    batch_size, num_nodes, _ = x.size()

    s = torch.softmax(s, dim=-1)

    if mask is not None:
        mask = mask.view(batch_size, num_nodes, 1).to(x.dtype)
        x, s = x * mask, s * mask

    out = torch.matmul(s.transpose(1, 2), x)
    out_adj = torch.matmul(torch.matmul(s.transpose(1, 2), adj), s)

    link_loss = adj - torch.matmul(s, s.transpose(1, 2))
    link_loss = torch.norm(link_loss, p=2)
    if normalize is True:
        link_loss = link_loss / adj.numel()

    ent_loss = (-s * torch.log(s + 1e-15)).sum(dim=-1).mean()

    return out, out_adj, link_loss, ent_loss, s


def compute_ricci_curvature(adj):
    G = nx.from_numpy_array(adj)
    orc = OllivierRicci(G, alpha=0.5, method="Sinkhorn")
    # G_OT1 = FormanRicci(G1)
    orc.compute_ricci_curvature()
    G_orc = orc.G.copy()
    plt.subplot(2, 1, 1)
    ricci_curvtures = nx.get_edge_attributes(G_orc, "ricciCurvature").values()
    plt.hist(ricci_curvtures,bins=20)
    plt.xlabel('Ricci curvature')
    plt.title("Histogram of Ricci Curvatures")

    # Plot the histogram of edge weights
    plt.subplot(2, 1, 2)
    weights = nx.get_edge_attributes(G, "weight").values()
    plt.hist(weights,bins=20)
    plt.xlabel('Edge weight')
    plt.title("Histogram of Edge weights")

    plt.tight_layout()
    plt.show()
    

def spectral_similarity(adj1, x1, adj2, x2):
    d = 10
    _, eig_vec = torch.linalg.eig(adj1)
    # x1 = torch.cat(x1, eig_vec[:, :d], dim=-1)
    x1 = eig_vec[:, :, :d].real
    x2 = x2[:, :, :d]
    D1 = torch.sum(adj1, dim=2, keepdim=True) * torch.eye(adj1.shape[-1]).cuda()
    D2 = torch.sum(adj2, dim=2, keepdim=True) * torch.eye(adj2.shape[-1]).cuda()
    
    loss = torch.abs(x1.transpose(1, 2).matmul(D1).matmul(x1) - x2.transpose(1, 2).matmul(D2).matmul(x2))
    return torch.mean(torch.diagonal(loss))


def edge_alpha(weight):
    min_weight = 0  
    max_weight = 1  
    alpha_min = 0.0  
    alpha_max = 0.2  
    alpha = np.interp(weight, (min_weight, max_weight), (alpha_min, alpha_max))
    return alpha    
    