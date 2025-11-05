import glob
import os
import subprocess

from infusse.config import DATA_DIR, STRUCTURE_DIR
from infusse.utils.biology_utils import get_first_digit

input_folder = STRUCTURE_DIR
file_list = list(dict.fromkeys(sorted([file for input_folder in STRUCTURE_DIR for file in glob.glob(os.path.join(input_folder, '*.pdb')) if '_stripped' not in file], key=get_first_digit)))
stride_path = '/Users/kevinmicha/Documents/PhD/stride-master/src/'
output_path = DATA_DIR + 'strides_outputs/'
pathological = ['1i8m', '1zea', '2fr4', '2r8s', '3eys', '3vw3', '4kze', '5e08', '5xli', '6b14', '6b3k', '6db9', '6df1', '6df2', '7bem', '7kmh', '7t0w', '7vux', '7ums', '8dp3', '8e8r', '8e8s', '8e8x', '8gb8', '8hrh', '8jgg', '8sh5']
pathological += ['2dko', '2j32', '3mea', '3mgn']
pathological += ['2ok0', '4z8f', '5ds8', '5dub', '5fgb', '6db8' ,'6u8d', '6x1s', '6x1u', '6x1w', '6xjq', '6xjw', '7t86', '7v5n', '8d29'] 
test_set = [
    '3u30', '1dqq', '1u95', '6n32', '1u8k', '7v05', '6e63', '2r1y', '6wm9', '2adf',
    '1mlb', '2uyl', '3ls5', '3d9a', '6oc7', '2w9d', '3okm', '5ebw', '1mjj', '1q0y',
    '6idg', '3whx', '7nx2', '2ipt', '3okd', '6aod', '7kkz', '4yhl', '6bb4', '6ucf',
    '3hnt', '3t77', '3gjf', '2orb', '8gby', '3cfb', '2otw', '4okv', '6wos', '7v64',
    '2r2b', '3hc4', '8hyl', '2xza', '4h88', '2dqt', '8gbw', '1ct8', '4hij', '5nh3',
    '6pa0', '7ufq', '2ykl', '4etq', '3cmo', '3ijh', '5a2j', '6b0g', '4ut7', '5a2l',
    '6o3a', '8ib1', '6q1g', '6was', '1f8t', '7dr4', '1igj', '3hc3', '7mu4', '5w24',
    '1f4x', '4hwe', '2ai0', '4lqf', '8ek5', '3t4y']


for file in file_list:
    pdb_code = file[-8:-4] 
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
                            ag_chain_3 = ''
                    else:
                        ag_chain_2 = ''
                        ag_chain_3 = ''
                else:
                    ag_chain = ''
                    ag_chain_2 = ''
                    ag_chain_3 = ''
                h_chain = line[line.find('HCHAIN')+len('HCHAIN')+1]
                l_chain = line[line.find('LCHAIN')+len('LCHAIN')+1]
    if pdb_code not in pathological:
        output_file = os.path.join(output_path, f'{pdb_code}.txt')  
        subprocess.call([os.path.join(stride_path, 'stride'), file, '-r'+h_chain+l_chain, '-f' + output_file], stdout=open(os.devnull, 'wb'))