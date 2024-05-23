import glob
import os
import numpy as np
import re
import torch

def parse_pdb(file_path):
    #amino_acids = {}
    b_amino_acids = []
    #current_aa = None

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
    
    with open(file_path, 'r') as pdb_file:
        for line in pdb_file:
            if line.startswith('ATOM') and (h_chain.upper() == line[slice(21, 22)] or l_chain.upper() == line[slice(21, 22)] or ag_chain == line[slice(21, 22)] or ag_chain_2 == line[slice(21, 22)] or ag_chain_3 == line[slice(21, 22)]):
                fields = re.split(r'\s+', line.strip())
                if fields[2] == 'CA':# and ((h_chain.upper() == line[slice(21, 22)] and int(line[slice(23, 26)]) >= 1 and int(line[slice(23, 26)]) <= 107) or (l_chain.upper() == line[slice(21, 22)] and int(line[slice(23, 26)]) >= 1 and int(line[slice(23, 26)]) <= 107)):
                    fields[-2] = '.'.join(fields[-2].split('.')[-2:]) if fields[-2].count('.') == 2 else fields[-2]
                    b_factor = float(fields[-2])
    
                    # Extract amino acid information
                    #res_id = fields[5]
                    #chain_id = fields[4]
                    #if len(chain_id) == 1:
                    #    aa_identifier = f"{chain_id}_{res_id}"
                    #else:
                    #    aa_identifier = f"{chain_id[0]}_{chain_id[1:]}"
    
                    # Check if the amino acid has been encountered before
                    #if aa_identifier not in amino_acids:
                    #    amino_acids[aa_identifier] = {
                    #        'b_factors': [],
                    #        'residue_id': fields[5],
                    #        'chain_id:': fields[4],
                    #    }
    
                    #amino_acids[aa_identifier]['b_factors'].append(b_factor)
                    b_amino_acids.append(b_factor)
                elif line.startswith('ENDMDL'):
                    break
                
    return b_amino_acids

def compute_average_b_factors(b_amino_acids):
    #averages = {}
    #unp = False
    #for aa_id, data in amino_acids.items():
    #    avg_b_factor = (data['b_factors'])[0]
    #    if avg_b_factor > 100:
    #        unp = True
    #    averages[aa_id] = avg_b_factor
    unp = False
    if any(b_fact_element > 120 or b_fact_element < 0 for b_fact_element in b_amino_acids):
        unp = True
    averages = b_amino_acids
    mean_ = np.mean(averages)
    std_dev_ = np.std(averages)
    if std_dev_ == 0:
        raise Exception('All the atoms have the same B factor') 

    return (np.array(averages) - mean_) / std_dev_, unp

def get_first_digit(filename):
    for char in filename:
        if char.isdigit():
            return int(char)
    return -1 

folder = '/Users/kevinmicha/Documents/all_structures/chothia_unbound'
folder_2 = '/Users/kevinmicha/Documents/all_structures/chothia_ext'
folder_3 = '/Users/kevinmicha/Documents/all_structures/chothia_ext_ext'
file_list = list(dict.fromkeys(sorted([file for file in glob.glob(os.path.join(folder, '*stripped.pdb')) if '_H' not in file]+[file for file in glob.glob(os.path.join(folder_2, '*stripped.pdb')) if '_H' not in file]+[file for file in glob.glob(os.path.join(folder_3, '*stripped.pdb')) if '_H' not in file], key=get_first_digit)))

pdb_codes = []
b_factors = []
pathological = ['1i8m', '1zea', '2fr4', '2r8s', '3eys', '3vw3', '4kze', '5e08', '5xli', '6b14', '6b3k', '6db9', '6df1', '6df2', '7bem', '7kmh', '7t0w', '7vux', '8dp3', '8e8r', '8e8s', '8e8x', '8gb8', '8hrh', '8jgg', '8sh5']
pathological += ['1rzg', '2vq1', '4ncc', '5drn', '6vor', '7kgu', '7pa7', '7rdk'] # unbound
pathological += ['1oay', '1oau', '4nm8', '4bkl', '7bep', '7yud', '8hrx', '8fja', '8st0'] + ['8gsf', '8gse', '8gsc', '8gsd'] # when putting bound and unbound together
pathological += ['2ok0', '4z8f', '5ds8', '5dub', '5fgb', '6db8' ,'6u8d', '6x1s', '6x1u', '6x1w', '6xjq', '6xjw', '7t86', '7v5n', '8d29'] # ext_ext
#pathological += ['7uja', '8fr6', '8g85', '8gas', '7ur6', '8heb', '8hec', '8hed', '7umn', '7uow', '7xj6', '7xj8', '7xj9', '7yac', '7yae', '8ct6', '8dto', '8dy6', '8g94', '8g9w', '8g9x', '8g9y', '8saq', '8sar', '8sas', '8sau', '8sav', '8saw', '8sax', '8say', '8saz', '8sb0', '8sb1', '8sb2', '8sb3', '8sb4', '8sb5', '8d21', '8dmh', '8gzz', '8h00', '8h01', '7upa', '7upb', '7upd', '7upk', '7ys6', '8d7e', '8fmz', '8fn0', '8fn1', '8il3', '7x93', '7x94', '7x95', '7x96', '7yon', '7yoo', '8bcz', '8hbd', '8hcq', '8hcx', '8ek1', '8eka', '8id3', '8id4', '8id6', '8id8', '8id9', '7ru8', '7ru5', '7e9o', '7ru4', '7x9y', '7vkt', '7ryc', '7t3m', '7ru3', '7xa3', '8hsc', '8g59', '8hs2', '8hs3', '8c7h', '7u0q', '7u0x', '7wsc', '7xcz', '7xda', '7xdb', '7xdk', '7yk6', '7yk7', '7xk2', '7urv', '7urx', '7wwi', '7wwj', '7wwk', '8f6e', '8f6f', '8f6h', '8f6i', '8f6j', '7wr9', '8hhx', '8hhy', '8hhz', '7ts0', '7tn9', '8de6', '7t5o', '8d48', '8dl6', '8dm3', '8dm4', '8epa', '8hc2', '8hc3', '8hc4', '8hc5', '8hc6', '8hc7', '8hc8', '8hc9', '8hca', '8hcb', '7wh8', '7whb', '7whd', '7zlg', '7zlh', '7zli', '7zlj', '7yar', '7wck', '7wcp', '7wcu', '7xw9', '7ykj', '7wti', '7wur', '8hii', '8hij', '8hik', '8gsf', '7wtf', '7wtg', '7wth', '7wtj', '7wtk', '8diu', '8gsc', '8gsd', '8gse', '7t2h', '7uz4', '7uz5', '7uz6', '7uz7', '7uz8', '7uz9', '7uza', '7uzb', '7x8w', '7x8y', '7x8z', '7x90', '7x91', '7x92', '8dl7', '8dl8', '8dw2', '8dw3', '7wos', '7wop', '7woq', '7wor', '7wou', '7wov', '7wow', '8e9w', '8e9x', '8e9y', '8e9z', '7t17', '7uvl', '7sww', '7swx', '7u9o', '7u9p', '7umm', '8dvd', '7six', '7sj0', '8dim', '7try', '7uot', '7uov', '7z12', '8dwc', '8dwg', '8dua', '7u8g', '7wbh', '7yr1', '7y24', '7y26', '7y27', '7yr0', '7y89', '7zce', '7zcf', '7vif', '7wrj', '7wry', '8e7m', '7zlk', '7srr', '7vfx', '7xco', '8dke', '8dkw', '8dkx', '7r0c', '7tjq', '7tl0', '7x5h', '7xxl', '7sjn', '7sjo', '7x1t', '7x1u', '7xat', '7xau', '7xav', '7xck', '7xcp', '8dad', '8dlr', '8dls', '8dlw', '8dpf', '8dpg', '8dph', '8dpi', '8dzh', '8dzi', '7usl', '7x7t', '7xq8', '7xw7', '7zbu', '8a1e', '7upl', '7u0p', '7z3a', '7upx', '7v23', '7v24', '7v27', '7wcd', '8cw9', '7t9n', '7qti', '7qtj', '7qtk', '7vgr', '7vgs', '7xms', '7xmt', '8dt3', '7wjy', '7sk7', '7ums', '7t0w', '7t0z', '7sk3', '7sk4', '7sk5', '7sk8', '7sk9', '7um5', '7um6', '7um7', '7ura', '7urc', '7urd', '7ure', '7urf', '7wo4', '7wo5', '7wo7', '7woa', '7wob', '7woc', '7wog', '7wp0', '7xow', '7xox', '7y12', '7y15', '7wj5', '7wjz', '7wk0', '7x6a', '7xbd', '7zyi', '7x1m', '7ran', '7wro', '7wrz', '7wr8', '7xjl', '7xjk', '7wrl']
for file in file_list:
    print(file[-17:-13])
    if os.path.isfile(f'/Users/kevinmicha/Documents/all_structures/adjacencies_sparse/{file[-17:-13]}.npz') and file[-17:-13] not in pathological:
        try:
            amino_acids_data = parse_pdb(file)
            avg_b_factors, unp = compute_average_b_factors(amino_acids_data)
            if unp == False:
                b_factors.append(torch.Tensor(avg_b_factors))
                pdb_codes.append(file[-17:-13])
        except:
            pass  

if b_factors:
    torch.save(b_factors, '/Users/kevinmicha/Documents/PhD/GCN-Bf/b_factors.pt')
    np.save('/Users/kevinmicha/Documents/PhD/GCN-Bf/pdb_codes.npy', pdb_codes)