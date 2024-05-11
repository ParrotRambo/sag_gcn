import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import os
import pandas as pd


class GCNDataset(Dataset):
    def __init__(self, data_path_neural, data_path_behavior, binary=False):
        self.data_path_neural = data_path_neural
        self.data_path_behavior = data_path_behavior

        self.files =  os.listdir(data_path_behavior)
        self.files = np.array([x.split(".")[0] for x in self.files])

        self.labels = np.array([pd.read_csv(os.path.join(self.data_path_behavior, x + '.tsv'), sep='\t')['Group'].values[0] for x in self.files])
        if binary:
            self.labels[self.labels != 'CON'] = 1
            self.labels[self.labels == 'CON'] = 0
            self.labels = self.labels.astype(int)
        else:
            self.labels = pd.get_dummies(self.labels).to_numpy()


    def __len__(self):
        return len(self.files)


    def __getitem__(self, idx):
        data = np.load(os.path.join(self.data_path_neural, self.files[idx] + '.pconn.npy'))
        adj_mat = np.zeros_like(data)
        adj_mat[data != 0] = 1
        return torch.tensor(data, dtype=torch.float32), torch.tensor(adj_mat, dtype=torch.float32), torch.tensor(self.labels[idx], dtype=torch.float32)
    

class SAGPool(nn.Module):
    def __init__(self, in_channels, ratio=0.8, non_linearity=torch.tanh):
        super().__init__()
        self.in_channels = in_channels
        self.ratio = ratio
        self.score_layer = GCNConv(in_channels,1)
        self.non_linearity = non_linearity


    def forward(self, x, adj_mat):
        # Score the nodes
        score = self.score_layer(x, adj_mat).squeeze()

        # Sort and take first N nodes
        score, perm = torch.sort(score)
        perm = perm[:int(perm.size(0) * self.ratio)]
        x = x[perm] * self.non_linearity(score[perm]).view(-1, 1)

        # Choose only specific rows and columns in adj matrix
        adj_mat = adj_mat[perm, :][:, perm]

        return x, adj_mat
    

class GCNConv(nn.Module):
    def __init__(self, in_features, out_features, use_bias=True):
        super(GCNConv, self).__init__()
        self.weight = nn.Parameter(torch.FloatTensor(torch.zeros(size=(in_features, out_features))),requires_grad=True)
        if use_bias:
            self.bias = nn.Parameter(torch.FloatTensor(torch.zeros(size=(out_features,))),requires_grad=True)
        else:
            self.register_parameter('bias', None)

        self.initialize_weights()

    def initialize_weights(self):
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x, adj_mat):
        x = x @ self.weight
        if self.bias is not None:
            x += self.bias
            
        d = torch.count_nonzero(adj_mat, dim=1)
        d = torch.diag(1 / torch.sqrt(d))

        return torch.sparse.mm(d @ adj_mat @ d, x)
    

class GCNSAGBLock(nn.Module):
    def __init__(self, in_features, nhid, pooling_ratio):
        super().__init__()
        self.in_features = in_features
        self.nhid = nhid
        self.pooling_ratio = pooling_ratio

        self.conv = GCNConv(self.in_features, self.nhid)
        self.pool = SAGPool(self.nhid, ratio=self.pooling_ratio)


    def forward(self, x, adj_mat):
        out = F.leaky_relu(self.conv(x, adj_mat))
        out, adj_mat = self.pool(out, adj_mat)
        return out, adj_mat
        

class mySequential(nn.Sequential):
    def forward(self, *inputs):
        for module in self._modules.values():
            if type(inputs) == tuple:
                inputs = module(*inputs)
            else:
                inputs = module(inputs)
        return inputs


class GCNetwork(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.num_features = args.num_features
        self.nhid = args.nhid
        self.num_classes = args.num_classes
        self.pooling_ratio = args.pooling_ratio
        self.linear_in = args.linear_in

        self.blocks = [GCNSAGBLock(self.num_features, self.nhid[0], self.pooling_ratio)]
        for i in range(len(self.nhid[:-1])):
            self.blocks.append(GCNSAGBLock(self.nhid[i], self.nhid[i+1], self.pooling_ratio))
        self.blocks = mySequential(*self.blocks)

        self.flatten = nn.Flatten(start_dim=0)
        self.lin1 = nn.Linear(self.nhid[-1]*self.linear_in, self.nhid[-1])
        self.lin2 = nn.Linear(self.nhid[-1], self.nhid[-1]//2)
        self.lin3 = nn.Linear(self.nhid[-1]//2, self.num_classes)
        #self.lin3 = nn.Linear(self.nhid[-1]*self.linear_in, self.num_classes)


    def forward(self, x, adj_mat):
        out = x
        out, adj_mat = self.blocks(out, adj_mat)

        out = self.flatten(out)
        out = F.leaky_relu(self.lin1(out))
        out = F.leaky_relu(self.lin2(out))
        return self.lin3(out).squeeze()