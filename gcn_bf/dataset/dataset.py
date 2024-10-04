# -*- coding: utf-8 -*-

r"""This module contains the model class. 

:Authors:   Kevin Michalewicz <k.michalewicz22@imperial.ac.uk>

"""
from torch_geometric.data import Data
from torch.utils.data import Dataset

class GCNBfDataset(Dataset):
    def __init__(self, edge_indices, edge_attributes, X, Y, device, pdb=None, Y_base=None, lm=None):
        self.edge_indices = [edge_index.to(device) for edge_index in edge_indices]
        self.edge_attributes = [edge_attr.to(device) for edge_attr in edge_attributes]
        if lm is None:
            self.X = [x.to(device) for x in X]
            self.num_features = X[0].shape[1]
        else:
            self.X = []
            self.num_features = 1
            print('Generating embeddings with Transformer')
            for x in X:
                mask = x > 4 # Special tokens go from 0 to 4 (incl.) in AntiBERTa
                x = lm(x[None,:], output_attentions=False, output_hidden_states=True)['hidden_states'][-1]
                x = x[mask.unsqueeze(-1).expand_as(x)].view(x.size(0), -1, x.size(-1))
                self.X.append(x.to(device).squeeze()[:250])
            print('End of generation')
        self.Y = [y.to(device) for y in Y]
        self.pdb = pdb
        self.out_channels = 1
        self.Y_base = Y_base

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        edge_index = self.edge_indices[idx]
        edge_attr = self.edge_attributes[idx]
        X = self.X[idx]
        Y = self.Y[idx]
        if self.pdb is not None:
            pdb = self.pdb[idx]
        else:
            pdb = None
    
        return Data(x=X, edge_index=edge_index, y=Y, edge_attr=edge_attr, pdb=pdb)