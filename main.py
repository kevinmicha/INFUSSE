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
parser.add_argument('--hidden_channels', type=int, default=128)
parser.add_argument('--lr', type=float, default=5e-3)
parser.add_argument('--epochs', type=int, default=1000)
args = parser.parse_args()

if torch.cuda.is_available():
    device = torch.device('cuda')
#elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
#    device = torch.device('mps')
else:
    device = torch.device('cpu')

class GCNBfDataset(Dataset):
    def __init__(self, edge_indices, edge_attributes, X, Y, pdb, Y_base=None):
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
        pdb = self.pdb[idx]
        #Y_base = self.Y_base[idx]
    
        return Data(x=X, edge_index=edge_index, y=Y, edge_attr=edge_attr)

# Edge data
edge_data = torch.load('edge_data.pt')
edge_indices = edge_data['edge_index']
edge_attributes = edge_data['edge_attr']

#edge_data_proteins = torch.load('edge_data_proteins.pt')
#edge_indices_proteins = edge_data_proteins['edge_index_proteins']
#edge_attributes_proteins = edge_data_proteins['edge_attr_proteins']

# Inputs to GCN 
X = torch.load('gcn_inputs.pt')

# B factor ground truths
Y = torch.load('b_factors.pt')

pdb_codes = np.load('pdb_codes.npy')

#X_proteins = torch.load('gcn_inputs_proteins.pt')
#Y_proteins = torch.load('b_factors_proteins.pt')

# Bulding the dataset and splitting into training and test sets
dataset = GCNBfDataset(edge_indices, edge_attributes, X, Y, pdb_codes)
train_size = int(0.9 * len(dataset))  
test_size = len(dataset) - train_size  
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
train_loader = DataLoader(dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

#dataset_proteins = GCNBfDataset(edge_indices_proteins, edge_attributes_proteins, X_proteins, Y_proteins)
#test_loader_proteins = DataLoader(dataset_proteins, batch_size=1, shuffle=False)
#test_size_proteins = len(dataset_proteins)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels, normalize=True)
        self.conv2 = GCNConv(hidden_channels, hidden_channels, normalize=True)
        #self.conv3 = GCNConv(hidden_channels, hidden_channels, normalize=True)

        self.conv3 = GCNConv(hidden_channels, out_channels, normalize=True)

    def forward(self, x, edge_index, edge_weight=None):
        #x = F.dropout(torch.squeeze(x), p=0.5, training=self.training)
        x = self.conv1(x, torch.squeeze(edge_index), torch.squeeze(edge_weight)).relu()
        #x = F.dropout(x, p=0.02, training=self.training)
        x = self.conv2(x, torch.squeeze(edge_index), torch.squeeze(edge_weight)).relu()

        #x = F.dropout(x, p=0.02, training=self.training)
        x = self.conv3(x, torch.squeeze(edge_index), torch.squeeze(edge_weight))
        #x = self.conv4(x, torch.squeeze(edge_index), torch.squeeze(edge_weight))

        return x

model = GCN(
    in_channels=dataset.num_features,
    hidden_channels=args.hidden_channels,
    out_channels=dataset.out_channels,
).to(device)

#optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
#optimizer = AdaBelief(model.parameters(), lr=0.005, weight_decay=False, eps=1e-16, rectify=False, weight_decouple=False, print_change_log=False) 
optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

print(count_parameters(model))
print(len(X))

def train():
    model.train()
    tr_loss = 0.0
    for loader in train_loader:
        optimizer.zero_grad()
        out = model(loader.x, loader.edge_index, loader.edge_attr)
        #out = (out - out.mean()) / out.std()
        loss = torch.nn.MSELoss(reduction='mean')(torch.squeeze(out), torch.squeeze(loader.y))
        tr_loss += loader.num_graphs * loss.item() / train_size 
        loss.backward()
        optimizer.step()
    return float(tr_loss)


@torch.no_grad()
def test():
    model.eval()
    test_loss = 0.0
    #test_loss_proteins = 0.0
    corr = 0.0
    #corr_proteins = 0.0
    for loader in test_loader:
        pred = model(loader.x, loader.edge_index, loader.edge_attr)
        #pred = (pred - pred.mean()) / pred.std()
        loss = torch.nn.MSELoss(reduction='mean')(torch.squeeze(pred), torch.squeeze(loader.y))
        test_loss += loader.num_graphs * loss.item() / test_size
        #if corr == 0.0:
            #print(np.abs(pred-y))
            #print(pdb)
            #plt.plot(np.arange(len(torch.squeeze(y).numpy())), np.abs(torch.squeeze(pred)-torch.squeeze(y)).numpy())
            #plt.show()
            #print(torch.corrcoef(torch.stack((torch.squeeze(pred), torch.squeeze(y))))[0,1])
        corr += loader.num_graphs * torch.corrcoef(torch.stack((torch.squeeze(pred), torch.squeeze(loader.y))))[0,1] / test_size 
        #print(pred)
        #print('----')
        #print(y)
        #print('++++')

    #for edge_index_proteins, edge_attr_proteins, x_proteins, y_proteins in test_loader_proteins:
    #    pred = model(x_proteins, edge_index_proteins, edge_attr_proteins)
    #   #pred = (pred - pred.mean()) / pred.std()
    #    loss = torch.nn.MSELoss(reduction='mean')(torch.squeeze(pred), torch.squeeze(y_proteins))
    #    test_loss_proteins += loss.item() / test_size_proteins 
    #    corr_proteins += torch.corrcoef(torch.stack((torch.squeeze(pred), torch.squeeze(y_proteins))))[0,1] / test_size_proteins 
    #    #print(pred)
    #    #print('----')
    #    #print(y)
    #    #print('++++')

    return float(test_loss), float(corr)#, float(test_loss_proteins), float(corr_proteins)


best_val_acc = test_acc = 0
times = []
for epoch in range(1, args.epochs + 1):
    start = time.time()
    loss = train()
    tmp_test_acc, corr = test()#, tmp_test_acc_p, corr_p = test()
    #if val_acc > best_val_acc:
    #    best_val_acc = val_acc
    #    test_acc = tmp_test_acc
    #log(Epoch=epoch, Loss=loss, Train=train_acc, Val=val_acc, Test=test_acc)
    log(Epoch=epoch, Loss=loss, Corr=corr, Test=tmp_test_acc)#, Corr_proteins=corr_p, Test_proteins=tmp_test_acc_p)
    times.append(time.time() - start)
print(f'Median time per epoch: {torch.tensor(times).median():.4f}s')