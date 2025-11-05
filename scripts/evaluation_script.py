import argparse
import numpy as np
import time
import torch

from torch_geometric.logging import log

from infusse.config import CHECKPOINTS_DIR
from infusse.dataset.dataset import GCNBfDataset
from infusse.model.model import GCN
from infusse.utils.torch_utils import get_dataloaders, load_transformer_weights, test

parser = argparse.ArgumentParser()
parser.add_argument('--graphs', type=str, default='gnm')
parser.add_argument('--lm', type=str, default='transformer')
parser.add_argument('--hidden_channels', type=int, default=512)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--epochs', type=int, default=500)
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

start = time.time()
#tmp_test_acc, corr = test(model, test_loader, test_size)
print(dataset[66])
tmp_test_acc, corr = test(model, dataset[76], 1) #1mlb 66, 1mlc 76
log(Corr=corr, Test=tmp_test_acc)
eval_time = time.time() - start
print(f'Median time for evaluation: {eval_time:.4f}s')
