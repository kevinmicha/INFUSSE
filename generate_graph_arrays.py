import glob
import json
import numpy as np
import os
import scipy
import torch

from torch_geometric.utils.convert import from_scipy_sparse_matrix

directory = '/Users/kevinmicha/Documents/PhD/GCN-Bf/'
#input_folder = '/Users/kevinmicha/Documents/all_structures/adjacencies_sparse/'
input_folder = '/Users/kevinmicha/Documents/all_structures/contact_maps/'
pdb_codes = np.load(directory+'pdb_codes.npy')
ei_list = []
ea_list = []
inv_laplacian_list = []
X = torch.load('gcn_inputs.pt')
Y = torch.load('b_factors.pt')


for i, pdb in enumerate(pdb_codes):
    adjacency = scipy.sparse.load_npz(input_folder+pdb+'.npz')
    edge_index, edge_attr = from_scipy_sparse_matrix(adjacency)
    edge_attr = torch.tensor(edge_attr, dtype=torch.float) 
    if edge_index.shape[-1] != edge_attr.shape[0] or X[i].shape[0] != Y[i].shape[0] or X[i].shape[0] != adjacency.toarray().shape[0]:
        print('Error')
        print(pdb)
        print(X[i].shape[0])
        print(adjacency.toarray().shape[0])
    ei_list.append(edge_index)
    ea_list.append(edge_attr)
torch.save({'edge_index': ei_list, 'edge_attr': ea_list}, 'edge_data.pt')
