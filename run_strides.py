import glob
import os
import subprocess

input_folder = '/Users/kevinmicha/Documents/all_structures/chothia_ext'
file_list = [file for file in glob.glob(os.path.join(input_folder, '*stripped.pdb')) if '_H' not in file]
stride_path = '/Users/kevinmicha/Documents/PhD/stride-master/src/'
output_path = 'strides_outputs/'
pathological = ['1i8m', '1zea', '2fr4', '2r8s', '3eys', '3vw3', '4kze', '5e08', '6b14', '6b3k', '6db9', '6df1', '6df2', '8hrh']

for file in file_list:
    pdb_code = file[-17:-13] 

    if pdb_code not in pathological:
        output_file = os.path.join(output_path, f'{pdb_code}.txt')  
        subprocess.call([os.path.join(stride_path, 'stride'), file, '-f' + output_file], stdout=open(os.devnull, 'wb'))