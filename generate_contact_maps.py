import glob
import numpy as np
import os
import scipy
import sys

from Bio.PDB import PDBParser

def compute_distance_matrix(file_path):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('protein', file_path)
    
    with open(file, 'r') as pdb_file:
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

    # Alpha-C coordinates 
    alpha_carbon_coordinates = {}
    for model in structure:
        for chain in model:
            if chain.id.upper() in [h_chain, l_chain, ag_chain, ag_chain_2, ag_chain_3]:
                for residue in chain:
                    if 'CA' in residue and residue.id[0] == ' ':# and ((h_chain == chain.id.upper() and residue.id[1] >= 1 and residue.id[1] <= 107) or (l_chain == chain.id.upper() and residue.id[1] >= 1 and residue.id[1] <= 107)):
                        alpha_carbon_coordinates[(chain.id, residue.id)] = residue['CA'].get_coord()
    
    # Compute distances and create matrix
    num_residues = len(alpha_carbon_coordinates)
    distance_matrix = np.zeros((num_residues, num_residues))

    for i, (_, coord1) in enumerate(alpha_carbon_coordinates.items()):
        for j, (_, coord2) in enumerate(alpha_carbon_coordinates.items()):
            distance_matrix[i, j] = np.linalg.norm(coord1 - coord2)

    return distance_matrix

# Saving
def save_distance_matrix(distance_matrix, output_file):
    scipy.sparse.save_npz(output_file, scipy.sparse.csr_matrix(distance_matrix))

folder = '/Users/kevinmicha/Documents/all_structures/chothia_unbound'
folder_2 = '/Users/kevinmicha/Documents/all_structures/chothia_ext'
folder_3 = '/Users/kevinmicha/Documents/all_structures/chothia_ext_ext'
file_list = sorted([file for file in glob.glob(os.path.join(folder, '*stripped.pdb')) if '_H' not in file]+[file for file in glob.glob(os.path.join(folder_2, '*stripped.pdb')) if '_H' not in file]+[file for file in glob.glob(os.path.join(folder_3, '*stripped.pdb')) if '_H' not in file])
threshold = 10.0

for file in file_list:
    print(file[-17:-13])
    distance_matrix = compute_distance_matrix(file)
    contact_map = np.where(distance_matrix<=threshold, 1, 0)
    save_distance_matrix(contact_map, f'/Users/kevinmicha/Documents/all_structures/contact_maps/{file[-17:-13]}.npz')