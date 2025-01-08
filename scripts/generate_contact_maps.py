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
    if h_chain == l_chain == ag_chain:
        ag_chain = None

    # Alpha-C coordinates 
    alpha_carbon_coordinates = []
    if h_chain:
        for model in structure:
            for chain in model:
                if (chain.id == h_chain or (chain.id.upper() == h_chain and h_chain != l_chain and h_chain != ag_chain)):
                    for residue in chain:
                        if 'CA' in residue and residue.id[0] == ' ':# and int(residue.id[1]) <= 112:
                            alpha_carbon_coordinates.append(residue['CA'].get_coord())
        
    # Process light chain next
    if l_chain:
        for model in structure:
            for chain in model:
                if (chain.id.upper() == l_chain and h_chain != l_chain) or (chain.id == l_chain.lower() and h_chain == l_chain):
                    for residue in chain:
                        if 'CA' in residue and residue.id[0] == ' ':# and int(residue.id[1]) <= 107:
                            alpha_carbon_coordinates.append(residue['CA'].get_coord())

    # Process antigen chains last
    if ag_chain:
        for model in structure:
            for chain in model:
                if (chain.id.upper() == ag_chain and l_chain != ag_chain and h_chain != ag_chain and ag_chain != ag_chain_2 and ag_chain != ag_chain_3) or (chain.id == ag_chain.lower() and (l_chain == ag_chain or h_chain == ag_chain)) or (chain.id == ag_chain and l_chain != ag_chain and h_chain != ag_chain and (ag_chain == ag_chain_2 or ag_chain == ag_chain_3)):
                    for residue in chain:
                        if 'CA' in residue and residue.id[0] == ' ':
                            alpha_carbon_coordinates.append(residue['CA'].get_coord())
    
    if ag_chain_2:
        for model in structure:
            for chain in model:
                if (chain.id.upper() == ag_chain_2 and ag_chain != ag_chain_2 and l_chain != ag_chain_2 and h_chain != ag_chain_2) or (chain.id == ag_chain_2.lower() and (l_chain == ag_chain_2 or h_chain == ag_chain_2 or ag_chain == ag_chain_2)):
                    for residue in chain:
                        if 'CA' in residue and residue.id[0] == ' ':
                            alpha_carbon_coordinates.append(residue['CA'].get_coord())
    if ag_chain_3:
        for model in structure:
            for chain in model:
                if (chain.id.upper() == ag_chain_3 and ag_chain != ag_chain_3 and ag_chain_2 != ag_chain_3 and l_chain != ag_chain_3 and h_chain != ag_chain_3) or (chain.id == ag_chain_3.lower() and (l_chain == ag_chain_3 or h_chain == ag_chain_3 or ag_chain == ag_chain_3 or ag_chain_2 == ag_chain_3)):
                    for residue in chain:
                        if 'CA' in residue and residue.id[0] == ' ':
                            alpha_carbon_coordinates.append(residue['CA'].get_coord())

    # Compute distances and create matrix
    num_residues = len(alpha_carbon_coordinates)
    distance_matrix = np.zeros((num_residues, num_residues))

    for i, coord1 in enumerate(alpha_carbon_coordinates):
        for j, coord2 in enumerate(alpha_carbon_coordinates):
            distance_matrix[i, j] = np.linalg.norm(coord1 - coord2)

    return distance_matrix

# Saving
def save_distance_matrix(distance_matrix, output_file):
    scipy.sparse.save_npz(output_file, scipy.sparse.csr_matrix(distance_matrix))

folder = '/Users/kevinmicha/Documents/all_structures/chothia_gcn_mice'
file_list = sorted([file for file in glob.glob(os.path.join(folder, '*.pdb')) if '_H' not in file])
threshold = 8.0 # 10.0

for file in file_list:
    print(file[-8:-4])
    file ='/Users/kevinmicha/Documents/all_structures/chothia_gcn/8ek5.pdb'
    distance_matrix = compute_distance_matrix(file)
    contact_map = np.where(distance_matrix<=threshold, 1, 0)
    print(contact_map.shape)
    #save_distance_matrix(np.where(distance_matrix == 0, 0, 1 / np.where(distance_matrix == 0, 1, distance_matrix)**2), f'/Users/kevinmicha/Documents/all_structures/distance_matrices/{file[-8:-4]}.npz')
    #save_distance_matrix(np.exp(-distance_matrix**2/64), f'/Users/kevinmicha/Documents/all_structures/distance_matrices/{file[-8:-4]}.npz')
    save_distance_matrix(contact_map, f'/Users/kevinmicha/Documents/all_structures/contact_maps/{file[-8:-4]}.npz')