import glob
import os
import numpy as np
import re
import torch

def parse_pdb(file_path):
    amino_acids = {}
    current_aa = None

    with open(file_path, 'r') as pdb_file:
        for line in pdb_file:
            if line.find('AGCHAIN') != -1:
                ag_chain = line[line.find('AGCHAIN')+len('AGCHAIN')+1]
                h_chain = line[line.find('HCHAIN')+len('HCHAIN')+1]
                l_chain = line[line.find('LCHAIN')+len('LCHAIN')+1]
                if line[line.find('AGCHAIN')+len('AGCHAIN')+2] == ';':
                    ag_chain_2 = line[line.find('AGCHAIN')+len('AGCHAIN')+3]
                else:
                    ag_chain_2 = None
                #if ag_chain == h_chain or ag_chain == l_chain:
                #    ag_chain = ' '
                #break
    
    with open(file_path, 'r') as pdb_file:
        for line in pdb_file:
            if line.startswith('ATOM') and (h_chain == line[slice(21, 22)] or l_chain == line[slice(21, 22)] or ag_chain == line[slice(21, 22)] or ag_chain_2 == line[slice(21, 22)]):
                fields = re.split(r'\s+', line.strip())
                if fields[2] == 'CA':# and int(line[slice(23, 26)]) >= 2 and int(line[slice(23, 26)]) <= 107:
                    fields[-2] = '.'.join(fields[-2].split('.')[-2:]) if fields[-2].count('.') == 2 else fields[-2]
                    b_factor = float(fields[-2])
    
                    # Extract amino acid information
                    res_id = fields[5]
                    chain_id = fields[4]
                    if len(chain_id) == 1:
                        aa_identifier = f"{chain_id}_{res_id}"
                    else:
                        aa_identifier = f"{chain_id[0]}_{chain_id[1:]}"
    
                    # Check if the amino acid has been encountered before
                    if aa_identifier not in amino_acids:
                        amino_acids[aa_identifier] = {
                            'b_factors': [],
                            'residue_id': fields[5],
                            'chain_id:': fields[4],
                        }
    
                    amino_acids[aa_identifier]['b_factors'].append(b_factor)
                elif line.startswith('ENDMDL'):
                    break
                
    return amino_acids

def compute_average_b_factors(amino_acids):
    averages = {}
    unp = False
    for aa_id, data in amino_acids.items():
        avg_b_factor = (data['b_factors'])[0]
        if avg_b_factor > 100:
            unp = True
        averages[aa_id] = avg_b_factor
    mean_ = np.mean(list(averages.values()))
    std_dev_ = np.std(list(averages.values()))

    standardised_averages = {aa_id: (avg - mean_) / std_dev_ for aa_id, avg in averages.items()}

    if std_dev_ == 0:
        raise Exception('All the atoms have the same B factor') 

    return standardised_averages, unp

folder = '/Users/kevinmicha/Documents/all_structures/chothia_ext'
file_list = sorted([file for file in glob.glob(os.path.join(folder, '*stripped.pdb')) if '_H' not in file])
pdb_codes = []
b_factors = []
pathological = ['1i8m', '1zea', '2fr4', '2r8s', '3eys', '3vw3', '4kze', '5e08', '6b14', '6b3k', '6db9', '6df1', '6df2', '8hrh']

for file in file_list:
    print(file[-17:-13])
    if os.path.isfile(f'/Users/kevinmicha/Documents/all_structures/adjacencies_sparse/{file[-17:-13]}.npz') and file[-17:-13] not in pathological:
        try:
            amino_acids_data = parse_pdb(file)
            avg_b_factors, unp = compute_average_b_factors(amino_acids_data)
            if unp == False:
                b_factors.append(torch.Tensor(list(avg_b_factors.values())))
                pdb_codes.append(file[-17:-13])
        except:
            pass  

if b_factors:
    torch.save(b_factors, '/Users/kevinmicha/Documents/PhD/GCN-Bf/b_factors.pt')
    np.save('/Users/kevinmicha/Documents/PhD/GCN-Bf/pdb_codes.npy', pdb_codes)