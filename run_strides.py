import glob
import os
import subprocess

input_folder = '/Users/kevinmicha/Documents/all_structures/chothia_vhh'

file_list = [file for file in glob.glob(os.path.join(input_folder, '*stripped.pdb')) if '_H' not in file]
stride_path = '/Users/kevinmicha/Documents/PhD/stride-master/src/'
output_path = 'strides_outputs/'
pathological = ['1i8m', '1zea', '2fr4', '2r8s', '3eys', '3vw3', '4kze', '5e08', '5xli', '6b14', '6b3k', '6db9', '6df1', '6df2', '7bem', '7kmh', '7t0w', '7vux', '8dp3', '8e8r', '8e8s', '8e8x', '8gb8', '8hrh', '8jgg', '8sh5']
pathological += ['2dko', '2j32', '3mea', '3mgn']

for file in file_list:
    pdb_code = file[-17:-13] 

    if pdb_code not in pathological:
        output_file = os.path.join(output_path, f'{pdb_code}.txt')  
        subprocess.call([os.path.join(stride_path, 'stride'), file, '-f' + output_file], stdout=open(os.devnull, 'wb'))