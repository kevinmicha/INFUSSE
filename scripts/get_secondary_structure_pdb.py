import os
import urllib.request
import numpy as np
import sys
import pickle

from gcn_bf.config import DATA_DIR, STRUCTURE_DIR

def fetch_pdb_from_web(pdb_code):
    url = f'https://files.rcsb.org/download/{pdb_code}.pdb'
    try:
        response = urllib.request.urlopen(url)
        pdb_content = response.read().decode('utf-8')
        return pdb_content
    except Exception as e:
        print(f'Error fetching PDB {pdb_code}: {e}')
        return None

def parse_antibody_chains(structure_dir, pdb_code):
    h_chain = None
    l_chain = None

    for folder in STRUCTURE_DIR:
        file_path = os.path.join(folder, f'{pdb_code}.pdb')
        if os.path.exists(file_path):
            with open(file_path, 'r') as pdb_file:
                pdb_content = pdb_file.read()
                for line in pdb_content.splitlines():
                    if 'HCHAIN' in line:
                        h_chain = line[line.find('HCHAIN') + len('HCHAIN') + 1]
                    if 'LCHAIN' in line:
                        l_chain = line[line.find('LCHAIN') + len('LCHAIN') + 1]
            break
    return h_chain, l_chain

def classify_residues_for_chains(pdb_content, h_chain, l_chain):
    helix_residues, strand_residues = parse_pdb_content_for_ss(pdb_content, h_chain, l_chain)

    heavy_chain_structure = []
    light_chain_structure = []

    for line in pdb_content.splitlines():
        record_type = line[:6].strip()
        if record_type == 'ATOM':
            atom_name = line[12:16].strip()
            if atom_name == 'CA': 
                chain_id = line[21]
                try:
                    res_seq = int(line[22:26].strip())
                    if chain_id == h_chain:
                        if res_seq in helix_residues[h_chain]:
                            heavy_chain_structure.append(0)
                        elif res_seq in strand_residues[h_chain]:
                            heavy_chain_structure.append(1)
                        else:
                            heavy_chain_structure.append(2)
                except ValueError:
                    continue

    for line in pdb_content.splitlines():
        record_type = line[:6].strip()
        if record_type == 'ATOM':
            atom_name = line[12:16].strip()
            if atom_name == 'CA': 
                chain_id = line[21]
                try:
                    res_seq = int(line[22:26].strip())
                    if chain_id == l_chain:
                        if res_seq in helix_residues[l_chain]:
                            light_chain_structure.append(0)
                        elif res_seq in strand_residues[l_chain]:
                            light_chain_structure.append(1)
                        else:
                            light_chain_structure.append(2)
                except ValueError:
                    continue
    return heavy_chain_structure + light_chain_structure

def parse_pdb_content_for_ss(pdb_content, h_chain, l_chain):
    helix_residues = {h_chain: set(), l_chain: set()}
    strand_residues = {h_chain: set(), l_chain: set()}

    for line in pdb_content.splitlines():
        record_type = line[:6].strip()
        if record_type == 'HELIX':
            chain_id = line[19].strip()
            start_residue = int(line[21:25].strip())
            end_residue = int(line[33:37].strip())
            for res in range(start_residue, end_residue + 1):
                if chain_id in [h_chain, l_chain]:
                    helix_residues[chain_id].add(res)
        elif record_type == 'SHEET':
            chain_id = line[21].strip()
            start_residue = int(line[22:26].strip())
            end_residue = int(line[33:37].strip())
            for res in range(start_residue, end_residue + 1):
                if chain_id in [h_chain, l_chain]:
                    strand_residues[chain_id].add(res)

    return helix_residues, strand_residues

def process_test_set(test_indices_file, pdb_codes_file, output_file):
    test_indices = np.load(test_indices_file)
    pdb_codes = np.load(pdb_codes_file)

    overall_structure = []

    for index in test_indices:
        pdb_code = pdb_codes[index]
        pdb_content = fetch_pdb_from_web(pdb_code)

        h_chain, l_chain = parse_antibody_chains(STRUCTURE_DIR, pdb_code)

        if h_chain or l_chain:
            structure = classify_residues_for_chains(pdb_content, h_chain, l_chain)
            overall_structure.append(structure)

    with open(output_file, 'wb') as f:
        pickle.dump(overall_structure, f)

def main():
    test_indices_file = DATA_DIR + 'test_indices.npy'
    pdb_codes_file = DATA_DIR + 'pdb_codes.npy'
    output_file = DATA_DIR + 'secondary_.pkl'

    process_test_set(test_indices_file, pdb_codes_file, output_file)

if __name__ == '__main__':
    main()