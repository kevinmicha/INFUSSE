import argparse
import numpy as np
import time
import torch

from torch_geometric.logging import log

from gcn_bf.config import CHECKPOINTS_DIR, STRUCTURE_DIR
from gcn_bf.dataset.dataset import GCNBfDataset
from gcn_bf.model.model import GCN
from gcn_bf.utils.biology_utils import extract_list_of_residues, find_cdr_positions
from gcn_bf.utils.torch_utils import get_dataloaders, plot_performance

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

checkpoint_path = CHECKPOINTS_DIR + f'{args.graphs}_feat_lstm_hidden_channels_{args.hidden_channels}_lr_{args.lr}_epochs_{args.epochs}/'

# Getting data, model and optimiser
train_loader, test_loader, test_size, _ = get_dataloaders(checkpoint_path, device)
model = torch.load(checkpoint_path+f'model_{args.graphs}.pth', map_location=device)
optimiser = torch.optim.AdamW(model.parameters(), lr=args.lr)

b_factor_per_residue = {}
last = False

for j, loader in enumerate(test_loader):
    h_l, l_l, ca_index = extract_list_of_residues(STRUCTURE_DIR+loader.pdb[0]+'_stripped.pdb')
    cdr_positions = find_cdr_positions(h_l, l_l)
    if j == len(test_loader) - 1:
        last = True
    b_factor_per_residue = plot_performance(model, loader, ca_index, cdr_positions, args.glob, b_factor_per_residue, h_l, l_l, last=last)
    