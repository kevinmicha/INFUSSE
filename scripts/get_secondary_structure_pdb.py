import os
import urllib.request
import numpy as np
import sys
import pickle

from infusse.config import DATA_DIR, STRUCTURE_DIR

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
    ag_chain = None
    ag_chain_2 = None
    ag_chain_3 = None

    for folder in STRUCTURE_DIR:
        file_path = os.path.join(folder, f'{pdb_code}.pdb')
        if os.path.exists(file_path):
            with open(file_path, 'r') as pdb_file:
                pdb_content = pdb_file.read()
                for line in pdb_content.splitlines():
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
            break
    if h_chain == l_chain:
        l_chain = l_chain.lower()
    if h_chain == ag_chain or l_chain == ag_chain:
        ag_chain = ag_chain.lower()

    return h_chain, l_chain, ag_chain, ag_chain_2, ag_chain_3

def classify_residues_for_chains(pdb_content, h_chain, l_chain, ag_chain, ag_chain_2, ag_chain_3):
    helix_residues, strand_residues = parse_pdb_content_for_ss(pdb_content, h_chain, l_chain, ag_chain, ag_chain_2, ag_chain_3)

    heavy_chain_structure = []
    light_chain_structure = []
    ag_chain_structure = []

    seen_residues = set()

    for line in pdb_content.splitlines():
        if line.startswith('ATOM'):
            atom_name = line[12:16].strip()
            if atom_name == 'CA':
                chain_id = line[21]
                res_seq = line[22:26].strip()
                ins_code = line[26].strip()
                res_id = f"{res_seq}{ins_code}"

                if chain_id == h_chain and (chain_id, res_id) not in seen_residues:
                    if res_id in helix_residues[h_chain]:
                        heavy_chain_structure.append(0)
                    elif res_id in strand_residues[h_chain]:
                        heavy_chain_structure.append(1)
                    else:
                        heavy_chain_structure.append(2)
                    seen_residues.add((chain_id, res_id))

    for line in pdb_content.splitlines():
        if line.startswith('ATOM'):
            atom_name = line[12:16].strip()
            if atom_name == 'CA':
                chain_id = line[21]
                res_seq = line[22:26].strip()
                ins_code = line[26].strip()
                res_id = f"{res_seq}{ins_code}"

                if chain_id == l_chain and (chain_id, res_id) not in seen_residues:
                    if res_id in helix_residues[l_chain]:
                        light_chain_structure.append(0)
                    elif res_id in strand_residues[l_chain]:
                        light_chain_structure.append(1)
                    else:
                        light_chain_structure.append(2)
                    seen_residues.add((chain_id, res_id))

    for line in pdb_content.splitlines():
        if line.startswith('ATOM'):
            atom_name = line[12:16].strip()
            if atom_name == 'CA':
                chain_id = line[21]
                res_seq = line[22:26].strip()
                ins_code = line[26].strip()
                res_id = f"{res_seq}{ins_code}"

                if chain_id in [ag_chain, ag_chain_2, ag_chain_3] and (chain_id, res_id) not in seen_residues:
                    if res_id in helix_residues[chain_id]:
                        ag_chain_structure.append(0)
                    elif res_id in strand_residues[chain_id]:
                        ag_chain_structure.append(1)
                    else:
                        ag_chain_structure.append(2)
                    seen_residues.add((chain_id, res_id))
    print(len(heavy_chain_structure + light_chain_structure + ag_chain_structure))
    return heavy_chain_structure + light_chain_structure + ag_chain_structure


def parse_pdb_content_for_ss(pdb_content, h_chain, l_chain, ag_chain, ag_chain_2, ag_chain_3):
    from string import ascii_uppercase

    helix_residues = {h_chain: set(), l_chain: set(), ag_chain: set(), ag_chain_2: set(), ag_chain_3: set()}
    strand_residues = {h_chain: set(), l_chain: set(), ag_chain: set(), ag_chain_2: set(), ag_chain_3: set()}

    def generate_res_ids(start_num, start_ins, end_num, end_ins):
        """Generate list of residue ids like '100H', '100I', ..., '100L'"""
        if start_num != end_num:
            # fallback to numeric range if numbers differ
            return [f"{i}" for i in range(start_num, end_num + 1)]
        else:
            start_index = ascii_uppercase.index(start_ins)
            end_index = ascii_uppercase.index(end_ins)
            return [f"{start_num}{ascii_uppercase[i]}" for i in range(start_index, end_index + 1)]

    for line in pdb_content.splitlines():
        record_type = line[:6].strip()
        if record_type == 'HELIX':
            chain_id = line[19].strip()
            start_res_num = int(line[21:25].strip())
            start_ins_code = line[25].strip()
            end_res_num = int(line[33:37].strip())
            end_ins_code = line[37].strip()

            if chain_id in helix_residues:
                for res in generate_res_ids(start_res_num, start_ins_code, end_res_num, end_ins_code):
                    helix_residues[chain_id].add(res)

        elif record_type == 'SHEET':
            chain_id = line[21].strip()
            start_res_num = int(line[22:26].strip())
            start_ins_code = line[26].strip()
            end_res_num = int(line[33:37].strip())
            end_ins_code = line[37].strip()

            if chain_id in strand_residues:
                for res in generate_res_ids(start_res_num, start_ins_code, end_res_num, end_ins_code):
                    strand_residues[chain_id].add(res)

    return helix_residues, strand_residues


def process_test_set(test_indices_file, pdb_codes_file, output_file):
    test_indices = np.load(test_indices_file)
    pdb_codes = np.load(pdb_codes_file)

    overall_structure = []

    for index in test_indices:
        pdb_code = pdb_codes[index]
        print(pdb_code)
        pdb_content = fetch_pdb_from_web(pdb_code)

        h_chain, l_chain, ag_chain, ag_chain_2, ag_chain_3 = parse_antibody_chains(STRUCTURE_DIR, pdb_code)

        if h_chain or l_chain or ag_chain or ag_chain_2 or ag_chain_3:
            structure = classify_residues_for_chains(pdb_content, h_chain, l_chain, ag_chain, ag_chain_2, ag_chain_3)
            overall_structure.append(structure)

    with open(output_file, 'wb') as f:
        pickle.dump(overall_structure, f)

def main():
    test_indices_file = DATA_DIR + 'test_indices.npy'
    pdb_codes_file = DATA_DIR + 'pdb_codes.npy'
    output_file = DATA_DIR + 'secondary_full.pkl'
    
    process_test_set(test_indices_file, pdb_codes_file, output_file)

if __name__ == '__main__':
    main()