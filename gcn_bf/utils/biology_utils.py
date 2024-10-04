import numpy as np
import re
import torch

from gcn_bf.config import STRUCTURE_DIR
from transformers import RoFormerTokenizer

def compute_average_b_factors(b_amino_acids, b_factor_thr=100):
    unp = False
    if any(b_fact_element > b_factor_thr or b_fact_element < 0 for b_fact_element in b_amino_acids):
        unp = True
    averages = b_amino_acids
    mean_ = np.mean(averages)
    std_dev_ = np.std(averages)
    if std_dev_ == 0:
        raise Exception('All the atoms have the same B factor') 

    return (np.array(averages) - mean_) / std_dev_, unp

def encode_line(line, amino_acids, secondary_structure):
    '''
    Generating one-hot encoded entry
    '''

    aa_mapping = {aa: i for i, aa in enumerate(amino_acids)}
    ss_mapping = {ss: i + len(amino_acids) for i, ss in enumerate(secondary_structure)}

    # From STRIDES output
    aa_pos = 1
    ss_pos = 6

    split_line = line.split()
    aa = split_line[aa_pos] 
    #ss = split_line[ss_pos]

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

def extract_list_of_residues(file_path):
    pdb = file_path[-8:-4]
    ca_index = 0
    h_res_list = []
    l_res_list = []
    h_idx_list = []
    l_idx_list = []

    with open(file_path, 'r') as pdb_file:
        for line in pdb_file:
            if line.find('HCHAIN') != -1 or line.find('LCHAIN') != -1:
                h_chain = line[line.find('HCHAIN')+len('HCHAIN')+1]
                l_chain = line[line.find('LCHAIN')+len('LCHAIN')+1]

    with open(file_path, 'r') as pdb_file:
        for line in pdb_file:
            fields = re.split(r'\s+', line.strip())
            if fields[2] == 'CA':
                if line.startswith('ATOM') and h_chain.upper() == line[slice(21, 22)] and int(line[slice(23, 26)]) <= 113:
                    h_res_list.append(fields[5])
                    h_idx_list.append(ca_index)
                    ca_index += 1
                elif line.startswith('ATOM') and (l_chain.upper() == line[slice(21, 22)]) and int(line[slice(23, 26)]) <= 107:
                    l_res_list.append(fields[5])
                    l_idx_list.append(ca_index)
                    ca_index += 1

    return h_res_list, l_res_list, h_idx_list+l_idx_list

def find_cdr_positions(heavy_l, light_l):
    heavy_set = ['26', '32', '52', '56', '95', '102']
    light_set = ['24', '34', '50', '56', '89', '97']
    cdr_indices = []
    l = heavy_l + light_l

    def find_next(l, target):
        try:
            return l.index(target)
        except ValueError:
            return -1 
    
    for target in heavy_set:
        idx = find_next(l, target)
        if idx != -1:
            cdr_indices.append(idx)
    
    for target in light_set:
        idx = find_next(l[len(heavy_l):], target)
        if idx != -1:
            cdr_indices.append(len(heavy_l)+idx)
    
    return cdr_indices

def format_sequence(sequence):
    seq_with_spaces = ' '.join(sequence)
    final_sequence = seq_with_spaces.replace('-', '[PAD]')
    final_sequence = final_sequence.replace(':', '[SEP]')
    final_sequence = final_sequence.replace('?', '[UNK]')
    
    return final_sequence

def generate_one_hot_matrix(file_path):
    '''
    Reading and extracting valid lines
    '''

    amino_acids = [' ', 'ASP', 'GLY', 'SEC', 'LEU', 'ASN', 'THR', 'LYS', 'HIS', 'TYR', 'TRP', 'CYS', 'PRO', 'VAL', 'SER', 'PYL', 'ILE', 'GLU', 'PHE', 'XAA', 'GLN', 'ALA', 'ASX', 'GLX', 'ARG', 'MET']
    secondary_structure = ['310Helix', 'AlphaHelix', 'Bridge', 'Coil', 'Strand', 'Turn']

    with open(file_path, 'r') as file:
        lines = file.readlines()
        valid_lines = [line for line in lines if line.startswith('ASG')]
        #valid_lines = [line for line in valid_lines if int(''.join(filter(str.isdigit, line.split()[3])))<=107]
        #X = np.zeros((len(valid_lines), len(amino_acids)+len(secondary_structure)), dtype=np.float32)
        X = np.zeros((len(valid_lines), len(amino_acids)), dtype=np.float32)
        for i, line in enumerate(valid_lines):
            X[i] = encode_line(line, amino_acids, secondary_structure)
    return X

def get_tokenised_sequence(file_path, cssp=False):
    aa_pos = 1
    chain_pos = 2
    amino_acid_dictionary = {
    'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C',
    'GLU': 'E', 'GLN': 'Q', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
    'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P',
    'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V',
    'ASX': 'B', 'GLX': 'Z', 'SEC': 'U', 'PYL': 'O', 'XAA': 'X',
    ' ': ' ', 
    }
    h_chain_seq = ''
    l_chain_seq = ''
    ag_chain_seq = ''
    input_folder = STRUCTURE_DIR    
    
    with open(input_folder+file_path[-8:-4]+'.pdb', 'r') as pdb_file:
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

    if cssp:
        tokeniser = RoFormerTokenizer.from_pretrained('alchemab/antiberta2-cssp')
    else:
        tokeniser = RoFormerTokenizer.from_pretrained('alchemab/antiberta2')

    with open(input_folder+file_path[-8:-4]+'.pdb', 'r') as pdb_file:
        for line in pdb_file:
            if line.startswith('ATOM'):
                atom_type = line[12:16].strip()
                if atom_type == 'CA':
                    chain_id = line[21] 
                    residue_name = line[17:20].strip()  
                    residue_number = line[22:26].strip() 

                    if residue_name in amino_acid_dictionary:
                        amino_acid = amino_acid_dictionary[residue_name]
                        
                        if chain_id == h_chain:
                            h_chain_seq += amino_acid
                        elif (chain_id == l_chain and h_chain != l_chain) or (chain_id == l_chain.lower() and h_chain == l_chain):
                            l_chain_seq += amino_acid
                        elif chain_id in [ag_chain, ag_chain_2, ag_chain_3]:
                            ag_chain_seq += amino_acid
    if ag_chain_seq != '':
        X = f'{h_chain_seq}:{l_chain_seq}:{ag_chain_seq}'
    else:
        X = f'{h_chain_seq}:{l_chain_seq}'
    input_seq = format_sequence(X)
    inputs = tokeniser(input_seq, return_tensors='pt')

    return inputs

def get_first_digit(filename):
    for char in filename:
        if char.isdigit():
            return int(char)
    return -1 

def parse_pdb(file_path):
    b_factors_heavy_chain = []
    b_factors_light_chain = []
    b_factors_antigens = []

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

    with open(file_path, 'r') as pdb_file:
        for line in pdb_file:
            if line.startswith('ATOM'):
                chain_id = line[slice(21, 22)].strip()  
                fields = re.split(r'\s+', line.strip())
                
                if fields[2] == 'CA': 
                    fields[-2] = '.'.join(fields[-2].split('.')[-2:]) if fields[-2].count('.') == 2 else fields[-2]
                    b_factor = float(fields[-2])

                    # Process heavy chain first
                    if h_chain.upper() == chain_id.upper():
                        b_factors_heavy_chain.append(b_factor)

                    # Then light chain 
                    elif l_chain.upper() == chain_id.upper():
                        b_factors_light_chain.append(b_factor)

                    # Finally antigen chains
                    elif chain_id in [ag_chain, ag_chain_2, ag_chain_3]:
                        b_factors_antigens.append(b_factor)

            elif line.startswith('ENDMDL'):
                break

                
    return b_factors_heavy_chain + b_factors_light_chain + b_factors_antigens

def sort_keys(keys):
    def res_id_sorting(key):
        match = re.match(r'(\d+)([A-Z]?)([HL])$', key)
        residue_number = int(match.group(1))
        residue_letter = match.group(2) or ''
        chain_letter = match.group(3)
        
        return (chain_letter == 'L', residue_number, residue_letter or '')

    return sorted(keys, key=res_id_sorting)