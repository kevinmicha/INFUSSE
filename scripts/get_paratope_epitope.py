import os
import glob
import numpy as np
import torch

from Bio.PDB import PDBParser
from Bio.PDB.Residue import Residue
from Bio.PDB.Chain import Chain
from Bio.PDB.Structure import Structure

from infusse.config import DATA_DIR, STRUCTURE_DIR
from infusse.utils.biology_utils import get_first_digit

def extract_paratope_epitope(pdb_file_path, pdb_codes):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('protein', pdb_file_path)

    antigen_chains = []
    with open(pdb_file_path, 'r') as pdb_file:
        for line in pdb_file:
            if line.find('AGCHAIN') != -1 or line.find('HCHAIN') != -1 or line.find('LCHAIN') != -1:
                if line[line.find('AGCHAIN')+len('AGCHAIN')+1:line.find('AGCHAIN')+len('AGCHAIN')+5] != 'NONE':
                    ag_chain_1 = line[line.find('AGCHAIN')+len('AGCHAIN')+1]
                    if line[line.find('AGCHAIN')+len('AGCHAIN')+2] == ';':
                        ag_chain_2 = line[line.find('AGCHAIN')+len('AGCHAIN')+3]
                        if line[line.find('AGCHAIN')+len('AGCHAIN')+4] == ';':
                            ag_chain_3 = line[line.find('AGCHAIN')+len('AGCHAIN')+5]
                            antigen_chains = [ag_chain_1, ag_chain_2, ag_chain_3]
                        else: 
                            ag_chain_3 = None
                            antigen_chains = [ag_chain_1, ag_chain_2]
                    else:
                        ag_chain_2 = None
                        ag_chain_3 = None
                        antigen_chains = [ag_chain_1]
                else:
                    ag_chain_1 = None
                    ag_chain_2 = None
                    ag_chain_3 = None
                    antigen_chains = []
                h_chain = line[line.find('HCHAIN')+len('HCHAIN')+1]
                l_chain = line[line.find('LCHAIN')+len('LCHAIN')+1]

    def get_interaction_residues(chain_1, antigen_chains, small_antigen=False):
        ab_interaction_residues = []
        ag_interaction_residues = []  
        current_offset_ag = 0  
        residue_numbering = {}  

        for antigen_chain in antigen_chains:
            current_offset_ab = -1
            for residue_2 in antigen_chain.get_residues():
                if not isinstance(residue_2, Residue):
                    continue
                residue_id = (antigen_chain.id+str(residue_2.id[1]) + residue_2.id[2]).strip()
                if residue_2.id[0] == ' ' and residue_id not in residue_numbering:
                    residue_numbering[residue_id] = current_offset_ag
                    current_offset_ag += 1

            for residue_1 in chain_1.get_residues():
                if not isinstance(residue_1, Residue):
                    continue
                if residue_1.id[0] == ' ':
                    current_offset_ab += 1
                for residue_2 in antigen_chain.get_residues():
                    if not isinstance(residue_2, Residue):
                        continue
                    for atom_1 in residue_1:
                        for atom_2 in residue_2:
                            distance = atom_1 - atom_2
                            if distance < 5.0:
                                res_ab = current_offset_ab# (str(residue_1.id[1]) + residue_1.id[2]).strip()
                                if residue_1.id[0] == ' ' and res_ab not in ab_interaction_residues and (not small_antigen or residue_2.id[0] != ' ') and residue_2.resname != 'HOH':
                                    ab_interaction_residues.append(res_ab)

                                if not small_antigen:
                                    absolute_residue_number = residue_numbering.get((antigen_chain.id+str(residue_2.id[1]) + residue_2.id[2]).strip())
                                    if residue_1.id[0] == ' ' and absolute_residue_number is not None and absolute_residue_number not in ag_interaction_residues:
                                        ag_interaction_residues.append(absolute_residue_number)
                                break
        return ab_interaction_residues, ag_interaction_residues

    # Fixing potential issues with chain IDs
    if h_chain == l_chain:
        l_chain = l_chain.lower()
    small_antigen = any(h_chain.upper() == ag_chain.upper() or l_chain.upper() == ag_chain.upper() for ag_chain in antigen_chains)

    try:
        heavy_chain = structure[0][h_chain]
        light_chain = structure[0][l_chain]
        antigen_chains = [structure[0][ag_chain] for ag_chain in antigen_chains]
    except KeyError: # strange case in which all chain IDs are actually lowercase
        heavy_chain = structure[0][h_chain.lower()]
        light_chain = structure[0][l_chain.lower()]
        antigen_chains = [structure[0][ag_chain.lower()] for ag_chain in antigen_chains]

    heavy_paratope, heavy_epitope = get_interaction_residues(heavy_chain, antigen_chains, small_antigen=small_antigen)
    light_paratope, light_epitope = get_interaction_residues(light_chain, antigen_chains, small_antigen=small_antigen)
    epitope = sorted(set(heavy_epitope + light_epitope), key=int)

    #print('Small antigen? '+str(small_antigen))
    if epitope:
        print(epitope)
    elif small_antigen:
        print(heavy_paratope)
        print(light_paratope)

    return heavy_paratope, light_paratope, epitope, small_antigen

def process_pdb_files():
    pdb_codes = np.load(os.path.join(DATA_DIR, 'pdb_codes.npy'))
    file_list = list(dict.fromkeys(sorted([
        file for folder in STRUCTURE_DIR
        for file in glob.glob(os.path.join(folder, '*.pdb'))
        if '_stripped' not in file
    ], key=get_first_digit)))

    result = {}

    for pdb_file_path in file_list:
       #pdb_file_path = '/Users/kevinmicha/Documents/all_structures/chothia_gcn/6wm9.pdb'
        pdb_id = os.path.basename(pdb_file_path).split('.')[0]
        print(pdb_id)
        if pdb_id not in pdb_codes:
            continue

        heavy_paratope, light_paratope, epitope, small_antigen = extract_paratope_epitope(pdb_file_path, pdb_codes)

        if not heavy_paratope and not light_paratope and (not epitope or small_antigen): # unbound cases should fall here
            continue
        result[pdb_id] = {
            'heavy_paratope': list(heavy_paratope),
            'light_paratope': list(light_paratope),
            'epitope': list(epitope),
        }

    #torch.save(result, os.path.join(DATA_DIR, 'paratope_epitope.pt'))

if __name__ == '__main__':
    process_pdb_files()
