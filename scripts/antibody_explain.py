import argparse
import numpy as np
import pickle
import time
import torch
import os

from torch_geometric.logging import log

from gcn_bf.config import CHECKPOINTS_DIR, DATA_DIR, STRUCTURE_DIR
from gcn_bf.dataset.dataset import GCNBfDataset
from gcn_bf.model.model import GCN
from gcn_bf.utils.biology_utils import extract_list_of_residues, find_cdr_positions
from gcn_bf.utils.torch_utils import count_parameters, get_dataloaders, plot_performance, load_transformer_weights, test, train

parser = argparse.ArgumentParser()
parser.add_argument('--graphs', type=str, default='gnm')
parser.add_argument('--lm', type=str, default='transformer')
parser.add_argument('--hidden_channels', type=int, default=512)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--epochs', type=int, default=500)
parser.add_argument('--glob', action='store_true')
args = parser.parse_args()

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

checkpoint_path = CHECKPOINTS_DIR + f'{args.graphs}_{args.lm}_features_hidden_channels_{args.hidden_channels}_lr_{args.lr}_epochs_{args.epochs}/'

# Getting data, model and optimiser
if args.lm == 'transformer':
    lm = load_transformer_weights(family='general')
    lm_ab = load_transformer_weights(family='antibody', cssp=False)
    train_loader, test_loader, test_size, dataset = get_dataloaders(checkpoint_path, device, lm_ab=lm_ab, lm_ag=lm)
else:
    train_loader, test_loader, test_size, _ = get_dataloaders(checkpoint_path, device)
model = torch.load(checkpoint_path+f'model_{args.graphs}.pth', map_location=device)
optimiser = torch.optim.AdamW(model.parameters(), lr=args.lr)

b_factor_per_residue = {}
last = False
all_errors = []

for j, loader in enumerate(test_loader):
    path = os.path.join(STRUCTURE_DIR, loader.pdb[0]+'.pdb')
    if os.path.exists(path):  
        h_l, l_l, ca_index = extract_list_of_residues(path)
        cdr_positions = find_cdr_positions(h_l, l_l)
        if j == len(test_loader) - 1:
            last = True
        b_factor_per_residue, list_of_errors = plot_performance(model, loader, ca_index, cdr_positions, args.glob, b_factor_per_residue, h_l, l_l, last=last)
        all_errors.append(list_of_errors)

with open(DATA_DIR+'errors_with_antigen.pkl', 'wb') as f:
    pickle.dump(all_errors, f)