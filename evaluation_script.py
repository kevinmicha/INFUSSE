import argparse
import matplotlib.pyplot as plt
import numpy as np
import os.path as osp
import time

import torch
import torch.nn.functional as F

from adabelief_pytorch import AdaBelief
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid
from torch_geometric.logging import init_wandb, log
from torch_geometric.nn import GCNConv
from torch_geometric.utils.convert import from_scipy_sparse_matrix, from_networkx
from torch.utils.data import Dataset, random_split
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data

from scipy import sparse

parser = argparse.ArgumentParser()
parser.add_argument('--hidden_channels', type=int, default=256)
parser.add_argument('--lr', type=float, default=5e-3)
parser.add_argument('--epochs', type=int, default=3000)
args = parser.parse_args()

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

class GCNBfDataset(Dataset):
    def __init__(self, edge_indices, edge_attributes, X, Y, pdb=None, Y_base=None):
        self.edge_indices = edge_indices
        self.edge_attributes = edge_attributes
        self.X = X
        self.Y = Y
        self.pdb = pdb
        self.num_features = X[0].shape[1]
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
    
        return Data(x=X, edge_index=edge_index, y=Y, edge_attr=edge_attr)

# Edge data (test set)
edge_data = torch.load('edge_data_proteins.pt')
edge_indices = edge_data['edge_index_proteins']
edge_attributes = edge_data['edge_attr_proteins']

# Input-output (test set)
X = torch.load('gcn_inputs_proteins.pt')
Y = torch.load('b_factors_proteins.pt')

# Bulding the test dataset 
dataset = GCNBfDataset(edge_indices, edge_attributes, X, Y)
test_loader = DataLoader(dataset, batch_size=32, shuffle=True)
test_size = len(dataset) 

class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels, normalize=True)
        self.conv2 = GCNConv(hidden_channels, hidden_channels, normalize=True)
        self.conv3 = GCNConv(hidden_channels, out_channels, normalize=True)

    def forward(self, x, edge_index, edge_weight=None):
        x = self.conv1(x, torch.squeeze(edge_index), torch.squeeze(edge_weight)).relu()
        x = self.conv2(x, torch.squeeze(edge_index), torch.squeeze(edge_weight)).relu()
        x = self.conv3(x, torch.squeeze(edge_index), torch.squeeze(edge_weight))

        return x

# Calling model and optimiser
model = torch.load(f'model_gnm_hidden_channels_{args.hidden_channels}_lr_{args.lr}_epochs_{args.epochs}.pth')
optimiser = torch.optim.AdamW(model.parameters(), lr=args.lr)


@torch.no_grad()
def test():
    model.eval()
    test_loss = 0.0
    corr = 0.0

    for loader in test_loader:
        pred = model(loader.x, loader.edge_index, loader.edge_attr)
        loss = torch.nn.MSELoss(reduction='mean')(torch.squeeze(pred), torch.squeeze(loader.y))
        test_loss += loader.num_graphs * loss.item() / test_size
        corr += loader.num_graphs * torch.corrcoef(torch.stack((torch.squeeze(pred), torch.squeeze(loader.y))))[0,1] / test_size 

    return float(test_loss), float(corr)

start = time.time()
tmp_test_acc, corr = test()
log(Corr=corr, Test=tmp_test_acc)
eval_time = time.time() - start
print(f'Median time for evaluation: {eval_time:.4f}s')
