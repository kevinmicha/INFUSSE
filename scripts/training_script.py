import argparse
import numpy as np
import time
import torch

from torch_geometric.logging import log

from gcn_bf.config import CHECKPOINTS_DIR, DATA_DIR
from gcn_bf.dataset.dataset import GCNBfDataset
from gcn_bf.model.model import GCN
from gcn_bf.utils.torch_utils import count_parameters, get_dataloaders, load_lstm_weights, test, train

parser = argparse.ArgumentParser()
parser.add_argument('--graphs', type=str, default='gnm')
parser.add_argument('--hidden_channels', type=int, default=256)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--epochs', type=int, default=500)
args = parser.parse_args()

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

# LSTM part
input_dim = 26  
hidden_dim = 512 
output_dim = 512 
num_layers = 2
lstm = torch.nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, num_layers=num_layers)
load_lstm_weights(lstm, CHECKPOINTS_DIR+'lstm_lm.hdf5')

# Data
train_loader, test_loader, test_size, dataset = get_dataloaders(DATA_DIR, device, mode='train')

model = GCN(
    in_channels=dataset.num_features,
    hidden_channels=args.hidden_channels,
    out_channels=dataset.out_channels,
    lstm=lstm,
).to(device)

optimiser = torch.optim.AdamW(model.parameters(), lr=args.lr)

print(count_parameters(model))
print(len(dataset))

best_val_acc = test_acc = 0
times = []
for epoch in range(1, args.epochs + 1):
    start = time.time()
    loss = train(model, optimiser, train_loader, len(dataset)-test_size)
    tmp_test_acc, corr = test(model, test_loader, test_size)
    log(Epoch=epoch, Loss=loss, Corr=corr, Test=tmp_test_acc)
    times.append(time.time() - start)
print(f'Median time per epoch: {torch.tensor(times).median():.4f}s')
torch.save(model, CHECKPOINTS_DIR+f'model_{args.graphs}_lstm_features_hidden_channels_{args.hidden_channels}_lr_{args.lr}_epochs_{args.epochs}.pth')
