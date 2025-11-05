import logomaker
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt 
import numpy as np
import pandas as pd
import pickle
import re
import torch

from anarci import run_anarci
from collections import defaultdict
from infusse.config import DATA_DIR

sequences = torch.load(DATA_DIR+'sequences.pt')
test_indices = np.load(DATA_DIR+'test_indices.npy')
pdb_codes = np.load(DATA_DIR+'pdb_codes.npy')
print(test_indices)
numbered_sequences = []

def sort_keys(items):
    def res_id_sorting(item):
        key = item[0]
        match = re.match(r'([HL])(\d+)([A-Z]?)$', key)
        chain = match.group(1)  # H or L
        residue_number = int(match.group(2))  
        insertion_code = match.group(3) or ''  # insertion code 
        
        return (chain, residue_number, insertion_code)

    return sorted(items, key=res_id_sorting)


for i, seq in enumerate(sequences):
    heavy_seq, light_seq = seq.split(':')

    heavy_results = run_anarci([('heavy', heavy_seq)], scheme='chothia')
    if not heavy_results[2][0]:
        print(f'Skipping heavy chain {pdb_codes[i]}')
        continue
    heavy_parsed = {
        f'H{pos[0]}{pos[1].strip()}': ('□' if residue == '-' else residue)
        for pos, residue in heavy_results[1][0][0][0]
    }

    light_results = run_anarci([('light', light_seq)], scheme='chothia')
    if not light_results[2][0]:
        print(f'Skipping light chain {pdb_codes[i]}')
        continue    
    light_parsed = {
        f'L{pos[0]}{pos[1].strip()}': ('□' if residue == '-' else residue)
        for pos, residue in light_results[1][0][0][0]
    }
    
    sequence = {**heavy_parsed, **light_parsed}
    numbered_sequences.append(sequence)
with open(DATA_DIR+'sequences_anarci.pkl', 'wb') as f:
    pickle.dump(numbered_sequences, f)
test_sequences = [numbered_sequences[i] for i in test_indices if i < len(numbered_sequences)]
numbered_sequences = [seq for i, seq in enumerate(numbered_sequences) if i not in test_indices]

position_counts = defaultdict(lambda: defaultdict(int))

for seq in numbered_sequences:
    for position, residue in seq.items():
        position_counts[position][residue] += 1

position_entropy = {}
for position, residues in position_counts.items():
    total_count = sum(residues.values())
    probabilities = np.array([count / total_count for count in residues.values()])
    entropy = -np.sum(probabilities * np.log(probabilities))
    position_entropy[position] = entropy

positions = []
entropies = []
for position, entropy in sort_keys(position_entropy.items()):
    print(f'Position {position}: Entropy = {entropy:.3f}')
    positions.append(position)
    entropies.append(entropy)
plt.plot(positions, entropies, '-')
print(positions)
print(entropies)
plt.show()

test_sequences = numbered_sequences + test_sequences # uncomment to only show test set in logo representation

test_conservation_scores = []
for seq in test_sequences:
    sequence_scores = {}
    for position, residue in seq.items():
        if position in position_entropy:
            sequence_scores[position] = position_entropy[position]
    test_conservation_scores.append(sequence_scores)

#with open(DATA_DIR+'conservation_scores.pkl', 'wb') as f:
#    pickle.dump(test_conservation_scores, f)

# All that follows is the sequence logo generation for the test set

colour_scheme = {
    'V': 'black', 'L': 'black', 'I': 'black', 'M': 'black', # Aliphatic hydrophobic
    'F': '#bcbd22', 'W': '#bcbd22', 'Y': '#bcbd22', # Aromatic
    'S': 'purple', 'T': 'purple', 'N': 'purple', 'Q': 'purple',  # Polar
    'R': 'blue', 'H': 'blue', 'K': 'blue',  # Positive (Basic)
    'D': 'red', 'E': 'red',  # Negative (Acidic)
    'A': 'gray', 'G': 'gray', 'P': 'gray', '□': 'gray', # Tiny
    'C': 'green',  # Cysteine
}

test_position_counts = defaultdict(lambda: defaultdict(int))
for seq in test_sequences:
    for position in position_entropy:
        if position not in seq:
            seq[position] = '□'
        residue = seq[position]
        residue = '□' if residue == '-' else residue
        test_position_counts[position][residue] += 1

test_logo_data = []
positions = []
s = 20
n_sequences = len(test_sequences)

for position, residues in sort_keys(test_position_counts.items()):
    total_count = sum(residues.values())
    frequencies = {res: count / total_count for res, count in residues.items()}
    entropy = -sum(f * np.log2(f) for f in frequencies.values() if f > 0)
    small_sample_correction = (1 / np.log(2)) * ((s - 1) / (2 * n_sequences))
    info_content = np.log2(s) - (entropy + small_sample_correction)
    positions.append(position)
    logo_row = {res: freq * info_content for res, freq in frequencies.items()}# if res != '-'}  
    test_logo_data.append(logo_row)

residues = sorted(set(res for freq in test_position_counts.values() for res in freq.keys()))# if res != '-'))
sorted_positions = [pos for pos, _ in sort_keys(test_position_counts.items())]
position_map = {pos: i for i, pos in enumerate(sorted_positions)}
numeric_indices = [position_map[pos] for pos in sorted_positions]

logo_df = pd.DataFrame(test_logo_data, index=numeric_indices, columns=residues).fillna(0)

plt.figure(figsize=(12, 6))
logo = logomaker.Logo(
    logo_df,
    shade_below=0.5,
    color_scheme=colour_scheme,
    font_name='Arial',
)

logo.ax.set_ylim(0, np.log2(s))
logo.ax.set_ylabel('Bits', fontsize=12)
logo.ax.set_xlabel('Position', fontsize=12)
logo.ax.set_xticks(range(len(positions)))
logo.ax.set_xticklabels(positions, rotation=90, fontsize=8)
plt.show()


