import glob
import numpy as np
import os
import scipy
import torch
import re

# The one-hot encoding for amino acids is generated based on the following sequence:
#['A', 'V', 'F', 'I', 'L','D','E','K','S','T','Y','C','N','Q', 'P','M', 'R', 'H', 'W', 'G', 'X']

amino_acids = ['ALA', 'VAL', 'PHE', 'ILE', 'LEU', 'ASP', 'GLU', 'LYS', 'SER', 'THR', 'TYR', 'CYS', 'ASN', 'GLN', 'PRO', 'MET', 'ARG', 'HIS', 'TRP', 'GLY']

secondary_structure = ['AlphaHelix', 'Strand', 'Coil']

aa_mapping = {aa: i for i, aa in enumerate(amino_acids)}
ss_mapping = {ss: i + len(amino_acids) + 1 for i, ss in enumerate(secondary_structure)}

def get_ca_coordinates(file_path):
    pattern = r'([+-]?\d+\.\d+)\s*([+-]?\d+\.\d+)\s*([+-]?\d+\.\d+)' # of line in PDB file

    with open(file_path, 'r') as pdb_file:
        for line in pdb_file:
            if line.find('AGCHAIN') != -1 or line.find('HCHAIN') != -1 or line.find('LCHAIN') != -1:
                if line[line.find('AGCHAIN')+len('AGCHAIN')+1:line.find('AGCHAIN')+len('AGCHAIN')+5] != 'NONE':
                    ag_chain = line[line.find('AGCHAIN')+len('AGCHAIN')+1]
                    if line[line.find('AGCHAIN')+len('AGCHAIN')+2] == ';':
                        ag_chain_2 = line[line.find('AGCHAIN')+len('AGCHAIN')+3]
                        if line[line.find('AGCHAIN')+len('AGCHAIN')+4] == ';':
                            ag_chain_3 = line[line.find('AGCHAIN')+len('AGCHAIN')+5]
                        else: 
                            ag_chain_3 = None
                    else:
                        ag_chain_2 = None
                        ag_chain_3 = None
                else:
                    ag_chain = None
                    ag_chain_2 = None
                    ag_chain_3 = None
                h_chain = line[line.find('HCHAIN')+len('HCHAIN')+1]
                l_chain = line[line.find('LCHAIN')+len('LCHAIN')+1]
    
    x_list = []
    y_list = []
    z_list = []

    with open(file_path, 'r') as pdb_file:
        for line in pdb_file:
            if line.startswith('ATOM') and (h_chain.upper() == line[slice(21, 22)] or l_chain.upper() == line[slice(21, 22)] or ag_chain == line[slice(21, 22)] or ag_chain_2 == line[slice(21, 22)] or ag_chain_3 == line[slice(21, 22)]):
                fields = re.split(r'\s+', line.strip())
                if fields[2] == 'CA':# and ((h_chain.upper() == line[slice(21, 22)] and int(line[slice(23, 26)]) >= 1 and int(line[slice(23, 26)]) <= 107) or (l_chain.upper() == line[slice(21, 22)] and int(line[slice(23, 26)]) >= 1 and int(line[slice(23, 26)]) <= 107)):
                    matches = re.search(pattern, line)
                    x_list.append(float(matches.group(1)))
                    y_list.append(float(matches.group(2)))
                    z_list.append(float(matches.group(3)))
                    #print(fields[6])
                    #x_list.append(fields[6])
                    #y_list.append(fields[7])
                    #z_list.append(fields[8])

                elif line.startswith('ENDMDL'):
                    break
                
    return x_list, y_list, z_list

# Generating one-hot encoded entry
def encode_line(line, x, y, z, end_chain):
    # From STRIDES output
    aa_pos = 1
    ss_pos = 6

    split_line = line.split()
    aa = split_line[aa_pos] 
    ss = split_line[ss_pos]

    one_hot = np.zeros(len(amino_acids)+1+len(secondary_structure)+4)
    if aa in aa_mapping:
        one_hot[aa_mapping[aa]] = 1
    else:
        one_hot[len(amino_acids)+1] = 1 # non-standard amino acid
    if ss in ss_mapping:
        one_hot[ss_mapping[ss]] = 1

    one_hot[-4] = x
    one_hot[-3] = y
    one_hot[-2] = z
    one_hot[-1] = end_chain

    return one_hot

# Reading and extracting valid lines
def generate_one_hot_matrix(file_path, x_list, y_list, z_list):
    with open(file_path, 'r') as file:
        x_list = np.array(x_list)
        y_list = np.array(y_list)
        z_list = np.array(z_list)
        x_list = (x_list - np.mean(x_list)) / np.std(x_list)
        y_list = (y_list - np.mean(y_list)) / np.std(y_list)
        z_list = (z_list - np.mean(z_list)) / np.std(z_list)

        lines = file.readlines()
        valid_lines = [line for line in lines if line.startswith('ASG')]
        end_chain = [1 if line[2] == 1 else 0 for line in valid_lines][1:-1]
        end_chain = [1] + [1 if (i < len(valid_lines) - 1 and (valid_lines[i+1][2] == 1 or valid_lines[i][2] ==1)) else 0 for i, line in enumerate(valid_lines)] + [1]
        X = np.zeros((len(valid_lines), len(amino_acids)+1+len(secondary_structure)+4), dtype=np.float32)
        for i, line in enumerate(valid_lines):
            X[i] = encode_line(line, x_list[i], y_list[i], z_list[i], end_chain[i])
    return X

def get_first_digit(filename):
    for char in filename:
        if char.isdigit():
            return int(char)
    return -1 

folder = '/Users/kevinmicha/Documents/all_structures/chothia_unbound'
folder_2 = '/Users/kevinmicha/Documents/all_structures/chothia_ext'
file_list = sorted([file for file in glob.glob(os.path.join(folder, '*stripped.pdb')) if '_H' not in file]+[file for file in glob.glob(os.path.join(folder_2, '*stripped.pdb')) if '_H' not in file], key=get_first_digit)

directory = '/Users/kevinmicha/Documents/PhD/GCN-Bf/'
input_folder = directory + 'strides_outputs/'
output_folder = directory + 'gcn_inputs/'
pdb_codes = np.load(directory+'pdb_codes.npy')
X_list = []
print(len(pdb_codes))

for pdb in pdb_codes:
    print(pdb)
    file_path = [file_name for file_name in file_list if file_name[-17:-13] == pdb][0]
    x_list, y_list, z_list = get_ca_coordinates(file_path)
    X = generate_one_hot_matrix(input_folder+pdb+'.txt', x_list, y_list, z_list)
    print(X.shape)
    X_list.append(torch.from_numpy(X))

torch.save(X_list, 'gcn_inputs_sequence_based.pt')
