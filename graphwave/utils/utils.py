# -*- coding: utf-8 -*-
"""
@author: cdonnat
"""
### Random tools useful for saveing stuff and manipulating pickle/numpy objects
import numpy as np
import pickle
import gzip
import re
import networkx as nx
import dgl
import torch


import logging
from functools import lru_cache

import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.data.sampler import SubsetRandomSampler

import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class DataSplit:

    def __init__(self, dataset, test_train_split=0.8, val_train_split=0.2, shuffle=False):
        self.dataset = dataset

        dataset_size = len(dataset)
        self.indices = list(range(dataset_size))
        test_split = int(np.floor(test_train_split * dataset_size))

        if shuffle:
            np.random.shuffle(self.indices)

        train_indices, self.test_indices = self.indices[:test_split], self.indices[test_split:]
        train_size = len(train_indices)
        validation_split = int(np.floor((1 - val_train_split) * train_size))

        self.train_indices, self.val_indices = train_indices[ : validation_split], train_indices[validation_split:]

        self.train_sampler = SubsetRandomSampler(self.train_indices)
        self.val_sampler = SubsetRandomSampler(self.val_indices)
        self.test_sampler = SubsetRandomSampler(self.test_indices)

    def get_train_split_point(self):
        return len(self.train_sampler) + len(self.val_indices)

    def get_validation_split_point(self):
        return len(self.train_sampler)

    @lru_cache(maxsize=4)
    def get_split(self, batch_size=50, num_workers=4):
        logging.debug('Initializing train-validation-test dataloaders')
        self.train_loader = self.get_train_loader(batch_size=batch_size, num_workers=num_workers)
        self.val_loader = self.get_validation_loader(batch_size=batch_size, num_workers=num_workers)
        self.test_loader = self.get_test_loader(batch_size=batch_size, num_workers=num_workers)
        return self.train_loader, self.val_loader, self.test_loader

    @lru_cache(maxsize=4)
    def get_train_loader(self, batch_size=50, num_workers=4):
        logging.debug('Initializing train dataloader')
        self.train_loader = torch.utils.data.DataLoader(self.dataset, batch_size=batch_size, sampler=self.train_sampler, shuffle=False, num_workers=num_workers)
        return self.train_loader

    @lru_cache(maxsize=4)
    def get_validation_loader(self, batch_size=50, num_workers=4):
        logging.debug('Initializing validation dataloader')
        self.val_loader = torch.utils.data.DataLoader(self.dataset, batch_size=batch_size, sampler=self.val_sampler, shuffle=False, num_workers=num_workers)
        return self.val_loader

    @lru_cache(maxsize=4)
    def get_test_loader(self, batch_size=50, num_workers=4):
        logging.debug('Initializing test dataloader')
        self.test_loader = torch.utils.data.DataLoader(self.dataset, batch_size=batch_size, sampler=self.test_sampler, shuffle=False, num_workers=num_workers)
        return self.test_loader


def save_obj(obj, name, path, compress=False):
    # print path+name+ ".pkl"
    if compress is False:
        with open(path + name + ".pkl", 'wb') as f:
            pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
    else:
        with gzip.open(path + name + '.pklz','wb') as f:
            pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name,compressed=False):
    if compressed is False:
        with open(name, 'rb') as f:
            return pickle.load(f)
    else:
        with gzip.open(name,'rb') as f:
            return pickle.load(f)


def atof(text):
    try:
        retval = float(text)
    except ValueError:
        retval = text
    return retval


def natural_keys(l):
    '''
        alist.sort(key=natural_keys) sorts in human order
        http://nedbatchelder.com/blog/200712/human_sorting.html
        (See Toothy's implementation in the comments)
        float regex comes from https://stackoverflow.com/a/12643073/190597
        '''
    t = np.array([ int(re.split(r"([a-zA-Z]*)([0-9]*)", c)[2]) for c in l  ])
    order = np.argsort(t)
    return [l[o] for o in order]


def saveNet2txt(G, colors=[], name="net", path="plots/"):
    '''saves graph to txt file (for Gephi plotting)
    INPUT:
    ========================================================================
    G:      nx graph
    colors: colors of the nodes
    name:   name of the file
    path:   path of the storing folder
    OUTPUT:
    ========================================================================
    2 files containing the edges and the nodes of the corresponding graph
    '''
    if len(colors) == 0:
        colors = range(nx.number_of_nodes(G))
    graph_list_rep = [["Id","color"]] + [[i,colors[i]]
                      for i in range(nx.number_of_nodes(G))]
    np.savetxt(path + name + "_nodes.txt", graph_list_rep, fmt='%s %s')
    edges = G.edges(data=False)
    edgeList = [["Source", "Target"]] + [[v[0], v[1]] for v in edges]
    np.savetxt(path + name + "_edges.txt", edgeList, fmt='%s %s')
    print ("saved network  edges and nodes to txt file (for Gephi vis)")
    return


def read_real_datasets(datasets):
    edge_path = "datasets/{}/out1_graph_edges.txt".format(datasets)
    node_feature_path = "datasets/{}/out1_node_feature_label.txt".format(datasets)
    with open(edge_path) as edge_file:
        edge_file_lines = edge_file.readlines()
        G = nx.parse_edgelist(edge_file_lines[1:], nodetype=int)
        # g = dgl.from_networkx(G)
    with open(node_feature_path) as node_feature_file:
        node_lines = node_feature_file.readlines()[1:]
        feature_list = []
        labels = []
        for node_line in node_lines:
            node_id, feature, label = node_line.split("\t")
            labels.append(int(label))
            features = feature.split(",")
            feature_list.append([float(feature) for feature in features])
        feature_array = np.array(feature_list)
        features = torch.from_numpy(feature_array)
        # g.ndata['attr'] = features.float()
        labels = torch.from_numpy(np.array(labels))
    return G, labels

class NodeClassificationDataset(Dataset):
    def __init__(self, node_embeddings, labels):
        self.len = node_embeddings.shape[0]
        self.x_data = node_embeddings
        self.y_data = labels.long()

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len

class MLP(nn.Module):
    def __init__(self, num_layers, input_dim, hidden_dim, output_dim):
        '''
            num_layers: number of layers in the neural networks (EXCLUDING the input layer). If num_layers=1, this reduces to linear model.
            input_dim: dimensionality of input features
            hidden_dim: dimensionality of hidden units at ALL layers
            output_dim: number of classes for prediction
            device: which device to use
        '''

        super(MLP, self).__init__()

        self.linear_or_not = True  # default is linear model
        self.num_layers = num_layers

        if num_layers < 1:
            raise ValueError("number of layers should be positive!")
        elif num_layers == 1:
            # Linear model
            self.linear = nn.Linear(input_dim, output_dim)
        else:
            # Multi-layer model
            self.linear_or_not = False
            self.linears = torch.nn.ModuleList()
            self.batch_norms = torch.nn.ModuleList()

            self.linears.append(nn.Linear(input_dim, hidden_dim))
            for layer in range(num_layers - 2):
                self.linears.append(nn.Linear(hidden_dim, hidden_dim))
            self.linears.append(nn.Linear(hidden_dim, output_dim))

            for layer in range(num_layers - 1):
                self.batch_norms.append(nn.BatchNorm1d((hidden_dim)))

    def forward(self, x):
        if self.linear_or_not:
            # If linear model
            return self.linear(x)
        else:
            # If MLP
            h = x
            for layer in range(self.num_layers - 1):
                h = F.relu(self.batch_norms[layer](self.linears[layer](h)))
            return self.linears[self.num_layers - 1](h)