import argparse
import glob
import numpy as np
import os
import scipy
import torch
import re

from gcn_bf.config import DATA_DIR
from gcn_bf.utils.biology_utils import encode_line, generate_one_hot_matrix, get_tokenised_sequence

parser = argparse.ArgumentParser()
parser.add_argument('--lm', type=str, default='transformer')
parser.add_argument('--cssp', type=bool, default=False)
args = parser.parse_args()

directory = DATA_DIR
input_folder = directory + 'strides_outputs/'
output_folder = directory + 'gcn_inputs/'
pdb_codes = np.load(directory+'pdb_codes.npy')
X_list = []
print(len(pdb_codes))

for pdb in pdb_codes:
    if args.lm == 'transformer':
        X = get_tokenised_sequence(input_folder+pdb+'.txt', args.cssp)
        X_list.append(X)
    else:
        X = generate_one_hot_matrix(input_folder+pdb+'.txt')
        X_list.append(torch.from_numpy(X))

torch.save(X_list, directory+'gcn_inputs.pt')
