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
parser.add_argument('--graphs', type=str, default='gnm')
parser.add_argument('--hidden_channels', type=int, default=256)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--epochs', type=int, default=500)
args = parser.parse_args()

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

class GCNBfDataset(Dataset):
    def __init__(self, edge_indices, edge_attributes, X, Y, pdb=None, Y_base=None):
        self.edge_indices = [edge_index.to(device) for edge_index in edge_indices]
        self.edge_attributes = [edge_attr.to(device) for edge_attr in edge_attributes]
        self.X = [x.to(device) for x in X]
        self.Y = [y.to(device) for y in Y]
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
        else:
            pdb = None
    
        return Data(x=X, edge_index=edge_index, y=Y, edge_attr=edge_attr, pdb=pdb)

checkpoint_path = f'checkpoints/{args.graphs}_feat_lstm_hidden_channels_{args.hidden_channels}_lr_{args.lr}_epochs_{args.epochs}/'

edge_data = torch.load(checkpoint_path+'edge_data.pt')
edge_indices = edge_data['edge_index']
edge_attributes = edge_data['edge_attr']
X = torch.load(checkpoint_path+'gcn_inputs.pt')
Y = torch.load(checkpoint_path+'b_factors.pt')
pdb_codes = np.load(checkpoint_path+'pdb_codes.npy')

dataset = GCNBfDataset(edge_indices, edge_attributes, X, Y, pdb=pdb_codes)
test_indices = np.load(checkpoint_path+'test_indices.npy')
train_size = int(0.95 * len(dataset))  
test_size = len(test_indices)
train_dataset, test_dataset = [dataset[i] for i in range(len(dataset)) if i not in test_indices], [dataset[i] for i in test_indices]
train_loader = DataLoader(train_dataset)
test_loader = DataLoader(test_dataset)

print(len(dataset))

class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.lm_dim = 512
        self.lstm = lstm
        self.aa_linear = nn.Linear(in_channels, self.lm_dim, bias=False)
        self.lm_linear = nn.Linear(self.lm_dim, self.lm_dim, bias=True)

        self.conv1 = GCNConv(self.lm_dim, hidden_channels, normalize=True)
        self.conv2 = GCNConv(hidden_channels, hidden_channels, normalize=True)

        self.conv3 = GCNConv(hidden_channels, out_channels, normalize=True)

    def forward(self, x, edge_index, edge_weight=None):

        x = (self.aa_linear(x) + self.lm_linear(self.lstm(x)[0])).relu()
        x = self.conv1(x, torch.squeeze(edge_index), torch.squeeze(edge_weight)).relu()
        x = self.conv2(x, torch.squeeze(edge_index), torch.squeeze(edge_weight)).relu()
        x = self.conv3(x, torch.squeeze(edge_index), torch.squeeze(edge_weight))

        return x

# Calling model and optimiser
model = torch.load(checkpoint_path+f'model_{args.graphs}.pth', map_location=device)
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
