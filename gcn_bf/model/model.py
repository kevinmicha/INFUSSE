# -*- coding: utf-8 -*-

r"""This module contains the model class. 

:Authors:   Kevin Michalewicz <k.michalewicz22@imperial.ac.uk>

"""

import torch
from torch_geometric.nn import GCNConv

class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, lm, lm_dim=1024):
        super().__init__()
        self.lm_dim = lm_dim
        self.lm = lm
        self.aa_linear = torch.nn.Linear(in_channels, self.lm_dim, bias=False)
        self.lm_linear = torch.nn.Linear(self.lm_dim, self.lm_dim, bias=True)

        self.conv1 = GCNConv(self.lm_dim, hidden_channels, normalize=True)
        self.conv2 = GCNConv(hidden_channels, hidden_channels, normalize=True)

        self.conv3 = GCNConv(hidden_channels, out_channels, normalize=True)

    def forward(self, x, edge_index, edge_weight=None):
        if self.lm_dim == 1024:
            mask = x > 4 # Special tokens go from 0 to 4 (incl.) in AntiBERTa
            x = self.lm(x[None,:], output_attentions=False, output_hidden_states=True)['hidden_states'][-1]
            x = x[mask.unsqueeze(-1).expand_as(x)].view(x.size(0), -1, x.size(-1))
        else:
            x = self.lm(x)[0]
            x = (self.aa_linear(x) + self.lm_linear(x)).relu()
        x = self.conv1(x, torch.squeeze(edge_index), torch.squeeze(edge_weight)).relu()
        x = self.conv2(x, torch.squeeze(edge_index), torch.squeeze(edge_weight)).relu()
        x = self.conv3(x, torch.squeeze(edge_index), torch.squeeze(edge_weight))

        return x