import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch

from torch_geometric.loader import DataLoader
from torch.utils.data import random_split
from transformers import RoFormerForMaskedLM

from gcn_bf.dataset.dataset import GCNBfDataset
from gcn_bf.utils.biology_utils import sort_keys

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def get_dataloaders(path, device, mode='test', train_size=0.95, lm=None):
    if mode == 'test':
        shuffle = False
        batch_size = 1
    else:
        shuffle = True
        batch_size = 8
        
    edge_data = torch.load(path+'edge_data.pt')
    edge_indices = edge_data['edge_index']
    edge_attributes = edge_data['edge_attr']
    X = torch.load(path+'gcn_inputs.pt')
    Y = torch.load(path+'b_factors.pt')
    pdb_codes = np.load(path+'pdb_codes.npy')

    dataset = GCNBfDataset(edge_indices, edge_attributes, X, Y, device=device, pdb=pdb_codes, lm=lm)
    if mode == 'test':
        test_indices = np.load(path+'test_indices.npy')
    train_size = int(train_size * len(dataset))  
    test_size = len(dataset) - train_size

    if mode == 'test':
        train_dataset, test_dataset = [dataset[i] for i in range(len(dataset)) if i not in test_indices], [dataset[i] for i in test_indices]
    else:
        train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    test_loader = DataLoader(test_dataset, batch_size=1)

    if mode != 'test':
        np.save(path+'test_indices.npy', np.array(test_dataset.indices))

    return train_loader, test_loader, test_size, dataset

def load_lstm_weights(model, path):
    with h5py.File(path, 'r') as f:
        weight_map = {
            'lstm.weight_ih_l0': 'model_weights/LSTM1/LSTM1/kernel:0',
            'lstm.weight_hh_l0': 'model_weights/LSTM1/LSTM1/recurrent_kernel:0',
            'lstm.bias_ih_l0': 'model_weights/LSTM1/LSTM1/bias:0',
            'lstm.bias_hh_l0': 'model_weights/LSTM1/LSTM1/bias:0',
            'lstm.weight_ih_l1': 'model_weights/LSTM2/LSTM2/kernel:0',
            'lstm.weight_hh_l1': 'model_weights/LSTM2/LSTM2/recurrent_kernel:0',
            'lstm.bias_ih_l1': 'model_weights/LSTM2/LSTM2/bias:0',
            'lstm.bias_hh_l1': 'model_weights/LSTM2/LSTM2/bias:0',
        }

        for name, param in model.named_parameters():
            if name in weight_map:
                weight_path = weight_map[name]
                weight_data = np.array(f[weight_path])

                if 'weight_ih' in name or 'weight_hh' in name:
                    param.data = torch.from_numpy(weight_data.T)
                elif 'bias' in name:
                    if '_ih' in name:
                        param.data = torch.from_numpy(weight_data[:len(weight_data)//2])
                    else:
                        param.data = torch.from_numpy(weight_data[len(weight_data)//2:])
                param.requires_grad = False # Freezing parameters

    model.eval()

    return model

def load_transformer_weights(cssp=False):
    if cssp:
        model = RoFormerForMaskedLM.from_pretrained('alchemab/antiberta2-cssp')
    else:
        model = RoFormerForMaskedLM.from_pretrained('alchemab/antiberta2')

    for name, param in model.named_parameters():
        param.requires_grad = False # Freezing parameters
    model.eval()

    return model

@torch.no_grad()
def plot_performance(model, loader, ca_index, cdr_positions, glob=False, res_dict=None, h_l=None, l_l=None, last=False):
    pred = model(loader.x, loader.edge_index, loader.edge_attr)[ca_index]
    y = loader.y[ca_index]
    l = h_l + l_l

    if not glob:
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


@torch.no_grad()
def test(model, test_loader, test_size):
    model.eval()
    test_loss = 0.0
    corr = 0.0
    for loader in test_loader:
        pred = model(loader.x, loader.edge_index, loader.edge_attr)
        loss = torch.nn.MSELoss(reduction='mean')(torch.squeeze(pred), torch.squeeze(loader.y))
        test_loss += loader.num_graphs * loss.item() / test_size
        print(loader.pdb)
        print(loader.num_graphs * torch.corrcoef(torch.stack((torch.squeeze(pred), torch.squeeze(loader.y))))[0,1])
        corr += loader.num_graphs * torch.corrcoef(torch.stack((torch.squeeze(pred), torch.squeeze(loader.y))))[0,1] / test_size 

    return float(test_loss), float(corr)

def train(model, optimiser, train_loader, train_size):
    model.train()
    tr_loss = 0.0
    for loader in train_loader:
        optimiser.zero_grad()
        out = model(loader.x, loader.edge_index, loader.edge_attr)
        loss = torch.nn.MSELoss(reduction='mean')(torch.squeeze(out), torch.squeeze(loader.y))
        tr_loss += loader.num_graphs * loss.item() / train_size 
        loss.backward()
        optimiser.step()
    return float(tr_loss)
