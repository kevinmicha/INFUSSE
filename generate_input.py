import glob
import numpy as np
import os
import scipy
import torch
import re

amino_acids = ['ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLU', 'GLN', 'GLY', 'HIS', 'ILE',
                     'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL']

secondary_structure = ['310Helix', 'AlphaHelix', 'Bridge', 'Coil', 'Strand', 'Turn']

aa_mapping = {aa: i for i, aa in enumerate(amino_acids)}
ss_mapping = {ss: i + len(amino_acids) for i, ss in enumerate(secondary_structure)}

# Generating one-hot encoded entry
def encode_line(line):
    # From STRIDES output
    aa_pos = 1
    ss_pos = 6

    split_line = line.split()
    aa = split_line[aa_pos] 
    ss = split_line[ss_pos]

    #one_hot = np.zeros(len(amino_acids)+len(secondary_structure))
    one_hot = np.zeros(len(amino_acids))
    if aa in aa_mapping:
        one_hot[aa_mapping[aa]] = 1
    #if ss in ss_mapping:
    #    one_hot[ss_mapping[ss]] = 1

    #one_hot[-3] = float(split_line[7]) / 360
    #one_hot[-2] = float(split_line[8]) / 180
    #one_hot[-1] = float(split_line[9]) 
    
    return one_hot

# Reading and extracting valid lines
def generate_one_hot_matrix(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        valid_lines = [line for line in lines if line.startswith('ASG')]
        #valid_lines = [line for line in valid_lines if int(''.join(filter(str.isdigit, line.split()[3])))<=107]
        #X = np.zeros((len(valid_lines), len(amino_acids)+len(secondary_structure)), dtype=np.float32)
        X = np.zeros((len(valid_lines), len(amino_acids)), dtype=np.float32)
        for i, line in enumerate(valid_lines):
            X[i] = encode_line(line)
    return X

directory = '/Users/kevinmicha/Documents/PhD/GCN-Bf/'
input_folder = directory + 'strides_outputs/'
output_folder = directory + 'gcn_inputs/'
pdb_codes = np.load(directory+'pdb_codes.npy')
X_list = []
print(len(pdb_codes))

for pdb in pdb_codes:
    X = generate_one_hot_matrix(input_folder+pdb+'.txt')
    X_list.append(torch.from_numpy(X))

torch.save(X_list, 'gcn_inputs.pt')
