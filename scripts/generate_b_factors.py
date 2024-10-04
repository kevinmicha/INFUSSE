import argparse
import glob
import os
import numpy as np
import torch

from gcn_bf.config import ADJACENCIES_DIR, DATA_DIR, STRUCTURE_DIR
from gcn_bf.utils.biology_utils import compute_average_b_factors, get_first_digit, parse_pdb

parser = argparse.ArgumentParser()
parser.add_argument('--thr', type=int, default=100)
args = parser.parse_args()

folder = STRUCTURE_DIR
file_list = list(dict.fromkeys(sorted([file for file in glob.glob(os.path.join(folder, '*.pdb')) if '_H' not in file], key=get_first_digit)))

pdb_codes = []
b_factors = []
pathological = ['1i8m', '1zea', '2fr4', '2r8s', '3eys', '3vw3', '4kze', '5e08', '5xli', '6b14', '6b3k', '6db9', '6df1', '6df2', '7bem', '7kmh', '7t0w', '7vux', '7ums', '7xq8', '8dp3', '8e8r', '8e8s', '8e8x', '8gb8', '8hrh', '8jgg', '8sh5']
pathological += ['1rzg', '2vq1', '4ncc', '5drn', '6vor', '7kgu', '7pa7', '7rdk'] # unbound
pathological += ['1oay', '1oau', '4nm8', '4bkl', '7bep', '7yud', '8hrx', '8fja', '8st0'] + ['8gsf', '8gse', '8gsc', '8gsd'] # when putting bound and unbound together
pathological += ['2ok0', '4z8f', '5ds8', '5dub', '5fgb', '6db8' ,'6u8d', '6x1s', '6x1u', '6x1w', '6xjq', '6xjw', '7t86', '7v5n', '8d29', '3v6o', '8de4', '7yar'] # ext_ext
pathological += ['3lqa', '4hlz', '4xi5', '5dwu', '5cjq', '6ehg'] #multi-antigen. These can be fixed

for file in file_list:
    print(file[-8:-4])
    if os.path.isfile(ADJACENCIES_DIR+f'{file[-8:-4]}.npz') and file[-8:-4] not in pathological:
        try:
            amino_acids_data = parse_pdb(file)
            avg_b_factors, unp = compute_average_b_factors(amino_acids_data, b_factor_thr=args.thr)
            if unp == False:
                b_factors.append(torch.Tensor(avg_b_factors[:250]))
                pdb_codes.append(file[-8:-4])
                #pdb_codes.append(file[-17:-13])
        except:
            pass  

if b_factors:
    torch.save(b_factors, DATA_DIR+'b_factors.pt')
    np.save(DATA_DIR+'pdb_codes.npy', pdb_codes)