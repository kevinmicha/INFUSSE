import argparse
import numpy as np
import time
import torch

from torch_geometric.logging import log

from gcn_bf.config import CHECKPOINTS_DIR, DATA_DIR
from gcn_bf.dataset.dataset import GCNBfDataset
from gcn_bf.model.model import GCN
from gcn_bf.utils.torch_utils import count_parameters, get_dataloaders, load_lstm_weights, load_transformer_weights, test, train

parser = argparse.ArgumentParser()
parser.add_argument('--graphs', type=str, default='gnm')
parser.add_argument('--lm', type=str, default='transformer')
parser.add_argument('--cssp', type=bool, default=False)
parser.add_argument('--hidden_channels', type=int, default=256)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--epochs', type=int, default=500)
args = parser.parse_args()

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
lm = None

if args.lm == 'transformer':
    lm_dim = 1024
    lm_ab = load_transformer_weights(family='antibody', cssp=args.cssp)
    lm = load_transformer_weights(family='general')
    train_loader, test_loader, test_size, dataset = get_dataloaders(DATA_DIR, device, mode='train', lm_ab=lm_ab, lm_ag=lm)
elif args.lm == 'lstm':
    input_dim = 26  
    lm_dim = 512 
    num_layers = 2
    lm = torch.nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, num_layers=num_layers)
    lm = load_lstm_weights(lm, CHECKPOINTS_DIR+'lstm_lm.hdf5')
    train_loader, test_loader, test_size, dataset = get_dataloaders(DATA_DIR, device, mode='train')
else:
    print('#TODO. Implement here the no-LM option')

model = GCN(
    in_channels=dataset.num_features,
    hidden_channels=args.hidden_channels,
    out_channels=dataset.out_channels,
    lm_dim=lm_dim,
    lm=lm,
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
torch.save(model, CHECKPOINTS_DIR+f'model_{args.graphs}_{args.lm}_features_hidden_channels_{args.hidden_channels}_lr_{args.lr}_epochs_{args.epochs}.pth')
