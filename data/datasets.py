from sklearn.model_selection import StratifiedShuffleSplit
import os.path as osp
import os

import torch
from torch_geometric.datasets import TUDataset, ZINC
from torch_geometric.utils import degree, dense_to_sparse
import torch_geometric.transforms as T

from utils import knn_filter, rwr_filter
from ogb.graphproppred import PygGraphPropPredDataset, Evaluator
from torch_geometric.data import DataLoader, Data, Batch
from pygsp import graphs
import numpy as np
from torch_geometric.data import InMemoryDataset

import pickle


rng = np.random.default_rng(1)

class FilterMaxNodes(object):
    def __init__(self, max_nodes):
        self.max_nodes = max_nodes

    def __call__(self, data):
        return data.num_nodes <= self.max_nodes


class FilterConstant(object):
    def __init__(self, dim):
        self.dim = dim

    def __call__(self, data):
        data.x = torch.ones(data.num_nodes, self.dim)
        return data


class NormalizedDegree(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, data):
        deg = degree(data.edge_index[0], dtype=torch.float)
        deg = (deg - self.mean) / self.std
        data.x = deg.view(-1, 1)
        return data


def get_data(name, batch_size, rwr=False, cleaned=False):
    if name == 'ogbg-molhiv':
        data_train, data_val, data_test, data = get_molhiv()
        num_classes = 2
        max_num_nodes = 0
        for d in data:
            max_num_nodes = max(d.num_nodes, max_num_nodes)
    elif name == 'ZINC':
        data_train, data_val, data_test = get_mod_zinc(rwr)
        max_num_nodes = 37
        num_classes = 1
    elif name == 'SMNIST':
        data_train, data_val, data_test = get_smnist(rwr)
        max_num_nodes = 75
        num_classes = 10
    elif name == 'EXPWL1':
        ### Dataset
        path = "data/EXPWL1/"
        dataset = EXPWL1Dataset(path, transform=DataToFloat()) 
        
        
        # data_train, data_val, data_test = data_split(dataset)
        # Random shuffle the data
        rnd_idx = rng.permutation(len(dataset))
        dataset = dataset[list(rnd_idx)]
        
        data_train = dataset[len(dataset) // 5:]
        data_val = dataset[:len(dataset) // 10]
        data_test = dataset[len(dataset) // 10:len(dataset) // 5]

        # compute avg number of nodes
        avg_nodes = int(dataset.data.num_nodes/len(dataset))

        # compute max number of nodes
        max_num_nodes = 0
        for d in dataset:
            max_num_nodes = max(d.num_nodes, max_num_nodes)
        num_classes = 2
        
    elif name == 'CYCLE':
        
        path = osp.dirname(osp.realpath(__file__))
        path += "/Cycles_3"
        data = SyntheticBaseDataset(path)
        num_classes = 2
        max_num_nodes = 0
        for d in data:
            max_num_nodes = max(d.num_nodes, max_num_nodes)
        data_train, data_val, data_test = data_split(data)
        

    else:
        data = get_tudataset(name, rwr, cleaned=cleaned)
        num_classes = data.num_classes
        max_num_nodes = 0
        for d in data:
            max_num_nodes = max(d.num_nodes, max_num_nodes)
        data_train, data_val, data_test = data_split(data)

    stats = dict()
    stats['num_features'] = data_train.num_node_features
    stats['num_classes'] = num_classes
    stats['max_num_nodes'] = max_num_nodes

    evaluator, encode_edge = (Evaluator(name), True) if name == 'ogbg-molhiv' else (None, False)

    if name == 'ZINC':
        encode_edge = True
        
    train_loader = DataLoader(data_train, batch_size, shuffle=True)
    val_loader = DataLoader(data_val, batch_size, shuffle=False)
    test_loader = DataLoader(data_test, batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, stats, evaluator, encode_edge


def get_molhiv():
    path = osp.dirname(osp.realpath(__file__))
    dataset = PygGraphPropPredDataset(name='ogbg-molhiv', root=path)
    split_idx = dataset.get_idx_split()
    # max_num_nodes = torch.tensor(dataset.data.num_nodes).max().item()
    return dataset[split_idx["train"]], dataset[split_idx["valid"]], dataset[split_idx["test"]], dataset


def get_tudataset(name, rwr, cleaned=False):
    transform = None
    if rwr:
        transform = rwr_filter
    if name == 'ENZYMES' or 'PROTEINS':
        path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'TU')
    else:
        path = osp.join(osp.dirname(osp.realpath(__file__)), ('rwr' if rwr else ''))
    dataset = TUDataset(path, name, pre_transform=transform, use_edge_attr=rwr, cleaned=cleaned)

    if dataset.data.x is None:
        max_degree = 0
        degs = []
        for data in dataset:
            degs += [degree(data.edge_index[0], dtype=torch.long)]
            max_degree = max(max_degree, degs[-1].max().item())
        dataset.transform = FilterConstant(10)#T.OneHotDegree(max_degree)    
        # dataset.transform = T.OneHotDegree(max_degree)
    return dataset


def data_split(dataset):
    skf_train = StratifiedShuffleSplit(n_splits=1, test_size=0.2)

    train_idx, val_test_idx = list(skf_train.split(torch.zeros(len(dataset)), dataset.data.y))[0]
    train_data = dataset[torch.from_numpy(train_idx)]

    skf_val = StratifiedShuffleSplit(n_splits=1, test_size=0.5)

    val_idx, test_idx = list(skf_val.split(torch.zeros(val_test_idx.size), dataset.data.y[val_test_idx]))[0]
    val_data = dataset[torch.from_numpy(val_test_idx[val_idx])]
    test_data = dataset[torch.from_numpy(val_test_idx[test_idx])]
    return train_data, val_data, test_data


def get_mod_zinc(rwr):
    transform = None
    name = 'ZINC'
    if rwr:
        transform = rwr_filter

    path = osp.join(osp.dirname(osp.realpath(__file__)), ('rwr' if rwr else ''), name)
    train = ZINC(path, split='train', pre_transform=transform)
    test = ZINC(path, split='test', pre_transform=transform)
    val = ZINC(path, split='val', pre_transform=transform)
    return train, val, test


def normalize_point_cloud(x):
    offset = np.mean(x, -2, keepdims=True)
    scale = np.abs(x).max()
    x = (x - offset) / scale
    x /= np.linalg.norm(x, axis=0, keepdims=True)

    return x


def get_sim_exp_dataset(name):
    if name.lower() == "grid2d":
        G = graphs.Grid2d(N1=8, N2=8)
    elif name.lower() == "ring":
        G = graphs.Ring(N=64)
    elif name.lower() == "bunny":
        G = graphs.Bunny()
    elif name.lower() == "airfoil":
        G = graphs.Airfoil()
    elif name.lower() == "minnesota":
        G = graphs.Minnesota()
    elif name.lower() == "torus":
        G = graphs.Torus(8, 8)
    elif name.lower() == "sensor":
        G = graphs.Sensor(N=64)
    elif name.lower() == "community":
        G = graphs.Community(N=64)
    elif name.lower() == "barabasialbert":
        G = graphs.BarabasiAlbert(N=64)
    elif name.lower() == "davidsensornet":
        G = graphs.DavidSensorNet(N=64)
    elif name.lower() == "erdosrenyi":
        G = graphs.ErdosRenyi(N=64)
    else:
        raise ValueError("Unknown dataset: {}".format(name))
    
    if not hasattr(G, "coords"):
        G.set_coordinates(kind="spring")
    x = G.coords.astype(np.float32)
    y = np.zeros(x.shape[0])  # X[:,0] + X[:,1]
    A = G.W
    if A.dtype.kind == "b":
        A = A.astype("i")


    AA = torch.Tensor(A.toarray())
    
    _, eig_vec = torch.linalg.eig(AA)
    
    edge_index, _ = dense_to_sparse(AA)
    data = Data(edge_index=edge_index, x=eig_vec[:, :10].real)
    data_list = []
    data_list.append(data)
    data_batch = Batch.from_data_list(data_list)
    
    x = normalize_point_cloud(x)

    return AA, x, y, data_batch


class EXPWL1Dataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super(EXPWL1Dataset, self).__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ["EXPWL1.pkl"]

    @property
    def processed_file_names(self):
        return 'data.pt'

    def download(self):
        pass  

    def process(self):
        # Read data into huge `Data` list.
        data_list = pickle.load(open(os.path.join(self.root, "raw/EXPWL1.pkl"), "rb"))

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
        

class DataToFloat(T.BaseTransform):
    def __call__(self, data):
        data.x = data.x.to(torch.float32)
        return data
    

class SyntheticBaseDataset(InMemoryDataset):
    def __init__(self, root=None, transform=None, pre_transform=None, **kwargs):
        super(SyntheticBaseDataset, self).__init__(
            root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['graphs.txt', 'labels.pt']

    @property
    def processed_file_names(self):
        return ['synthetic_data.pt']

    def process(self):
        # Read data into huge `Data` list.
        with open(f"{self.root}/graphs.txt", "rb") as fp:   # Unpickling
            x_list, edge_list = pickle.load(fp)

        labels = torch.load(f"{self.root}/labels.pt")
        data_list = [Data(x=x_list[i], edge_index=edge_list[i],
                          y=labels[i][None]) for i in range(len(x_list))]

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


