import torch
import torch.nn.functional as F
EPS = 1e-15


def dense_dmon_pool(x, adj, s, mask = None):
    r"""
    Args:
        x (torch.Tensor): Node feature tensor
            :math:`\mathbf{X} \in \mathbb{R}^{B \times N \times F}`, with
            batch-size :math:`B`, (maximum) number of nodes :math:`N` for
            each graph, and feature dimension :math:`F`.
            Note that the cluster assignment matrix
            :math:`\mathbf{S} \in \mathbb{R}^{B \times N \times C}` is
            being created within this method.
        adj (torch.Tensor): Adjacency tensor
            :math:`\mathbf{A} \in \mathbb{R}^{B \times N \times N}`.
        mask (torch.Tensor, optional): Mask matrix
            :math:`\mathbf{M} \in {\{ 0, 1 \}}^{B \times N}` indicating
            the valid nodes for each graph. (default: :obj:`None`)

    :rtype: (:class:`torch.Tensor`, :class:`torch.Tensor`,
        :class:`torch.Tensor`, :class:`torch.Tensor`,
        :class:`torch.Tensor`, :class:`torch.Tensor`)
    """
    x = x.unsqueeze(0) if x.dim() == 2 else x
    adj = adj.unsqueeze(0) if adj.dim() == 2 else adj

    s = torch.softmax(s, dim=-1)

    (batch_size, num_nodes, _), k = x.size(), s.size(-1)

    if mask is not None:
        mask = mask.view(batch_size, num_nodes, 1).to(x.dtype)
        x, s = x * mask, s * mask

    out = F.selu(torch.matmul(s.transpose(1, 2), x))
    out_adj = torch.matmul(torch.matmul(s.transpose(1, 2), adj), s)

    # Spectral loss:
    degrees = torch.einsum('ijk->ik', adj).transpose(0, 1)
    m = torch.einsum('ij->', degrees)

    ca = torch.matmul(s.transpose(1, 2), degrees)
    cb = torch.matmul(degrees.transpose(0, 1), s)

    normalizer = torch.matmul(ca, cb) / 2 / m
    decompose = out_adj - normalizer
    spectral_loss = -_rank3_trace(decompose) / 2 / m
    spectral_loss = torch.mean(spectral_loss)

    # Orthogonality regularization:
    ss = torch.matmul(s.transpose(1, 2), s)
    i_s = torch.eye(k).type_as(ss)
    ortho_loss = torch.norm(
        ss / torch.norm(ss, dim=(-1, -2), keepdim=True) -
        i_s / torch.norm(i_s), dim=(-1, -2))
    ortho_loss = torch.mean(ortho_loss)

    # Cluster loss:
    cluster_loss = torch.norm(torch.einsum(
        'ijk->ij', ss)) / adj.size(1) * torch.norm(i_s) - 1

    # Fix and normalize coarsened adjacency matrix:
    ind = torch.arange(k, device=out_adj.device)
    out_adj[:, ind, ind] = 0
    d = torch.einsum('ijk->ij', out_adj)
    d = torch.sqrt(d)[:, None] + EPS
    out_adj = (out_adj / d) / d.transpose(1, 2)

    # # To make adj sparse using gumbel softmax
    min_val = torch.min(out_adj.view(out_adj.shape[0], -1), dim=-1)[0]
    max_val = torch.max(out_adj.view(out_adj.shape[0], -1), dim=-1)[0]
    out_adj = (out_adj - min_val.unsqueeze(-1).unsqueeze(-1)) / (max_val - min_val).unsqueeze(-1).unsqueeze(-1)

    pad = 1 - out_adj.unsqueeze(-1)
    logits = torch.cat((out_adj.unsqueeze(-1), pad), dim=-1)
    out_adj = F.gumbel_softmax(logits, tau=1, hard=True)[:, :, :, 0]
    
    # symmetric
    out_adj = torch.triu(out_adj) + torch.triu(out_adj).transpose(1, 2)
    
    diagonal_indices = torch.arange(out_adj.shape[-1])
    out_adj[:, diagonal_indices, diagonal_indices] = 1

    return out, out_adj, spectral_loss, ortho_loss, cluster_loss


def _rank3_trace(x):
    return torch.einsum('ijj->i', x)


def _rank3_diag(x):
    eye = torch.eye(x.size(1)).type_as(x)
    out = eye * x.unsqueeze(2).expand(*x.size(), x.size(1))
    return out
