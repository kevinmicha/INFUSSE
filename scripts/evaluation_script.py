import argparse
import numpy as np
import time
import torch

from torch_geometric.logging import log

from gcn_bf.config import CHECKPOINTS_DIR
from gcn_bf.dataset.dataset import GCNBfDataset
from gcn_bf.model.model import GCN
from gcn_bf.utils.torch_utils import get_dataloaders, test

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

checkpoint_path = CHECKPOINTS_DIR + f'{args.graphs}_feat_lstm_hidden_channels_{args.hidden_channels}_lr_{args.lr}_epochs_{args.epochs}/'

# Getting data, model and optimiser
train_loader, test_loader, test_size, _ = get_dataloaders(checkpoint_path, device)
model = torch.load(checkpoint_path+f'model_{args.graphs}.pth', map_location=device)
optimiser = torch.optim.AdamW(model.parameters(), lr=args.lr)

start = time.time()
tmp_test_acc, corr = test(model, test_loader, test_size)
log(Corr=corr, Test=tmp_test_acc)
eval_time = time.time() - start
print(f'Median time for evaluation: {eval_time:.4f}s')
