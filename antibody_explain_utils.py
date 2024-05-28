import argparse
import matplotlib.pyplot as plt
import numpy as np
import os.path as osp
import re
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
parser.add_argument('--glob', action='store_true')
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
        #Y_base = self.Y_base[idx]
    
        return Data(x=X, edge_index=edge_index, y=Y, edge_attr=edge_attr, pdb=pdb)

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

checkpoint_path = f'checkpoints/{args.graphs}_feat_lstm_hidden_channels_{args.hidden_channels}_lr_{args.lr}_epochs_{args.epochs}/'

# Edge data
edge_data = torch.load(checkpoint_path+'edge_data.pt')
edge_indices = edge_data['edge_index']
edge_attributes = edge_data['edge_attr']

# Inputs to GCN 
X = torch.load(checkpoint_path+'gcn_inputs.pt')

# B factor ground truths
Y = torch.load(checkpoint_path+'b_factors.pt')

pdb_codes = np.load(checkpoint_path+'pdb_codes.npy')

# Bulding the dataset and splitting into training and test sets
dataset = GCNBfDataset(edge_indices, edge_attributes, X, Y, pdb=pdb_codes)
test_indices = np.load(checkpoint_path+'test_indices.npy')
train_size = int(0.95 * len(dataset))  
test_size = len(test_indices)
train_dataset, test_dataset = [dataset[i] for i in range(len(dataset)) if i not in test_indices], [dataset[i] for i in test_indices]
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

model = torch.load(checkpoint_path+f'model_{args.graphs}.pth', map_location=device)
optimiser = torch.optim.AdamW(model.parameters(), lr=args.lr)

def extract_list_of_residues(file_path):
    pdb = file_path[-8:-4]
    ca_index = 0
    h_res_list = []
    l_res_list = []
    h_idx_list = []
    l_idx_list = []

    with open(file_path, 'r') as pdb_file:
        for line in pdb_file:
            if line.find('HCHAIN') != -1 or line.find('LCHAIN') != -1:
                h_chain = line[line.find('HCHAIN')+len('HCHAIN')+1]
                l_chain = line[line.find('LCHAIN')+len('LCHAIN')+1]

    with open(file_path, 'r') as pdb_file:
        for line in pdb_file:
            fields = re.split(r'\s+', line.strip())
            if fields[2] == 'CA':
                if line.startswith('ATOM') and h_chain.upper() == line[slice(21, 22)] and int(line[slice(23, 26)]) <= 113:
                    h_res_list.append(fields[5])
                    h_idx_list.append(ca_index)
                elif line.startswith('ATOM') and (l_chain.upper() == line[slice(21, 22)]) and int(line[slice(23, 26)]) <= 107:
                    l_res_list.append(fields[5])
                    l_idx_list.append(ca_index)
                ca_index += 1

    return h_res_list, l_res_list, h_idx_list+l_idx_list

folder = '/Users/kevinmicha/Documents/all_structures/chothia_ext/'
b_factor_per_residue = {}
last = False

def find_cdr_positions(heavy_l, light_l):
    heavy_set = ['26', '32', '52', '56', '95', '102']
    light_set = ['24', '34', '50', '56', '89', '97']
    cdr_indices = []
    l = heavy_l + light_l

    def find_next(l, target):
        try:
            return l.index(target)
        except ValueError:
            return -1 
    
    for target in heavy_set:
        idx = find_next(l, target)
        if idx != -1:
            cdr_indices.append(idx)
    
    for target in light_set:
        idx = find_next(l[len(heavy_l):], target)
        if idx != -1:
            cdr_indices.append(len(heavy_l)+idx)
    
    return cdr_indices

def sort_keys(keys):
    def res_id_sorting(key):
        match = re.match(r'(\d+)([A-Z]?)([HL])$', key)
        residue_number = int(match.group(1))
        residue_letter = match.group(2) or ''
        chain_letter = match.group(3)
        
        return (chain_letter == 'L', residue_number, residue_letter or '')

    return sorted(keys, key=res_id_sorting)

@torch.no_grad()
def plot_performance(loader, ca_index, cdr_positions, glob=False, res_dict=None, h_l=None, l_l=None, last=False):
    pred = model(loader.x, loader.edge_index, loader.edge_attr)[ca_index]
    y = loader.y[ca_index]
    l = h_l + l_l

    if not args.glob:
        plt.plot(np.arange(len(torch.squeeze(y).numpy())), ((torch.squeeze(pred)-torch.squeeze(y))**2).numpy())
        for i in range(len(cdr_positions)//2):
            plt.axvspan(cdr_positions[2*i], cdr_positions[2*i+1], alpha=0.1, color='green')
        plt.show()
    else:
        for i, item in enumerate(l):
            if i < len(h_l):
                item += 'H'
            else:
                item += 'L'
            if item in res_dict:
                res_dict[item]['tot_b_factor'] += ((torch.squeeze(pred[i])-torch.squeeze(y[i]))**2).numpy()
                res_dict[item]['count'] += 1
            else:
                res_dict[item] = {'tot_b_factor': ((torch.squeeze(pred[i])-torch.squeeze(y[i]))**2).numpy(), 'count': 1}
        if last: # last PDB
            residue_ids = sort_keys(list(res_dict.keys()))
            cdr_positions = [residue_ids.index(el) for el in ['26H', '32H', '52H', '56H', '95H', '102H']] + [residue_ids.index(el) for el in ['24L', '34L', '50L', '56L', '89L', '97L']]
            b_factors = [res_dict[id_]['tot_b_factor'] / res_dict[id_]['count'] for id_ in residue_ids]
            plt.plot(range(len(residue_ids)), b_factors, marker='o', linestyle='-')
            for i in range(len(cdr_positions)//2):
                plt.axvspan(cdr_positions[2*i], cdr_positions[2*i+1], alpha=0.1, color='green')
            plt.xlabel('Residue index')
            plt.ylabel('MSE')
            plt.show()
    return res_dict

for j, loader in enumerate(test_loader):
    h_l, l_l, ca_index = extract_list_of_residues(folder+loader.pdb[0]+'_stripped.pdb')
    cdr_positions = find_cdr_positions(h_l, l_l)
    if j == len(test_loader) - 1:
        last = True
    b_factor_per_residue = plot_performance(loader, ca_index, cdr_positions, args.glob, b_factor_per_residue, h_l, l_l, last=last)
    