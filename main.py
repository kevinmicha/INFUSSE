import argparse
import numpy as np
import os.path as osp
import time

import torch
import torch.nn.functional as F

import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid
from torch_geometric.logging import init_wandb, log
from torch_geometric.nn import GCNConv
from torch_geometric.utils.convert import from_scipy_sparse_matrix, from_networkx
from torch.utils.data import Dataset, DataLoader, random_split

from scipy import sparse

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='Cora')
parser.add_argument('--hidden_channels', type=int, default=128)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--epochs', type=int, default=10000)
parser.add_argument('--use_gdc', action='store_true', help='Use GDC')
parser.add_argument('--wandb', action='store_true', help='Track experiment')
args = parser.parse_args()

if torch.cuda.is_available():
    device = torch.device('cuda')
#elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
#    device = torch.device('mps')
else:
    device = torch.device('cpu')

class GCNBfDataset(Dataset):
    def __init__(self, edge_indices, edge_attributes, X, Y, Y_base):
        self.edge_indices = edge_indices
        self.edge_attributes = edge_attributes
        self.X = X
        self.Y = Y
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
        Y_base = self.Y_base[idx]
        return edge_index, edge_attr, X, Y, Y_base

# Edge data
edge_data = torch.load('edge_data.pt')
edge_indices = edge_data['edge_index']
edge_attributes = edge_data['edge_attr']
Y_base = edge_data['inv_laplacian']

# Inputs to GCN 
X = torch.load('gcn_inputs.pt')

# B factor ground truths
Y = torch.load('b_factors.pt')

# Bulding the dataset and splitting into training and test sets
dataset = GCNBfDataset(edge_indices, edge_attributes, X, Y, Y_base)
train_size = int(0.95 * len(dataset))  
test_size = len(dataset) - train_size  
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels, normalize=True)
        self.conv2 = GCNConv(hidden_channels, hidden_channels, normalize=True)
        self.conv3 = GCNConv(hidden_channels, out_channels, normalize=True)

    def forward(self, x, edge_index, edge_weight=None):
        #x = F.dropout(torch.squeeze(x), p=0.5, training=self.training)
        x = self.conv1(x, torch.squeeze(edge_index), torch.squeeze(edge_weight)).relu()
        #x = F.dropout(x, p=0.05, training=self.training)
        x = self.conv2(x, torch.squeeze(edge_index), torch.squeeze(edge_weight)).relu()
        #x = F.dropout(x, p=0.05, training=self.training)
        x = self.conv3(x, torch.squeeze(edge_index), torch.squeeze(edge_weight))

        return x

model = GCN(
    in_channels=dataset.num_features,
    hidden_channels=args.hidden_channels,
    out_channels=dataset.out_channels,
).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=0.002) 

print(count_parameters(model))

def train():
    model.train()
    tr_loss = 0.0
    for edge_index, edge_attr, x, y, _ in train_loader:
        optimizer.zero_grad()
        out = model(x, edge_index, edge_attr)
        #out = (out - out.mean()) / out.std()
        loss = torch.nn.MSELoss(reduction='mean')(torch.squeeze(out), torch.squeeze(y))
        tr_loss += loss.item() / train_size 
        loss.backward()
        optimizer.step()
    return float(tr_loss)


@torch.no_grad()
def test():
    model.eval()
    test_loss = 0.0
    corr = 0.0
    for edge_index, edge_attr, x, y, _ in test_loader:
        pred = model(x, edge_index, edge_attr)
        #pred = (pred - pred.mean()) / pred.std()
        loss = torch.nn.MSELoss(reduction='mean')(torch.squeeze(pred), torch.squeeze(y))
        test_loss += loss.item() / test_size 
        corr += torch.corrcoef(torch.stack((torch.squeeze(pred), torch.squeeze(y))))[0,1] / test_size 
        print(pred)
        print('----')
        print(y)
        print('++++')
    return float(test_loss), float(corr)


best_val_acc = test_acc = 0
times = []
for epoch in range(1, args.epochs + 1):
    start = time.time()
    loss = train()
    tmp_test_acc, corr = test()
    #if val_acc > best_val_acc:
    #    best_val_acc = val_acc
    #    test_acc = tmp_test_acc
    #log(Epoch=epoch, Loss=loss, Train=train_acc, Val=val_acc, Test=test_acc)
    log(Epoch=epoch, Loss=loss, Corr=corr, Test=tmp_test_acc)
    times.append(time.time() - start)
print(f'Median time per epoch: {torch.tensor(times).median():.4f}s')