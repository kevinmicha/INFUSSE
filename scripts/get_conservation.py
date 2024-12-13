import numpy as np
import pickle
import torch

from anarci import run_anarci
from collections import defaultdict
from gcn_bf.config import DATA_DIR

sequences = torch.load(DATA_DIR+'sequences.pt')
test_indices = np.load(DATA_DIR+'test_indices.npy')
pdb_codes = np.load(DATA_DIR+'pdb_codes.npy')
print(test_indices)
numbered_sequences = []

for i, seq in enumerate(sequences):
    heavy_seq, light_seq = seq.split(':')

    heavy_results = run_anarci([('heavy', heavy_seq)], scheme='chothia')
    if not heavy_results[2][0]:
        print(f'Skipping heavy chain {pdb_codes[i]}')
        continue
    heavy_parsed = {f'H{pos[0]}{pos[1].strip()}': residue for pos, residue in heavy_results[1][0][0][0]}

    light_results = run_anarci([('light', light_seq)], scheme='chothia')
    if not light_results[2][0]:
        print(f'Skipping light chain {pdb_codes[i]}')
        continue    
    light_parsed = {f'L{pos[0]}{pos[1].strip()}': residue for pos, residue in light_results[1][0][0][0]}
    
    sequence = {**heavy_parsed, **light_parsed}
    numbered_sequences.append(sequence)

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

for position, entropy in sorted(position_entropy.items()):
    print(f'Position {position}: Entropy = {entropy:.3f}')

test_conservation_scores = []
for seq in test_sequences:
    sequence_scores = {}
    for position, residue in seq.items():
        if position in position_entropy and residue != '-':
            sequence_scores[position] = position_entropy[position]
    #print(sequence_scores)
    test_conservation_scores.append(sequence_scores)

with open(DATA_DIR+'conservation_scores.pkl', 'wb') as f:
    pickle.dump(test_conservation_scores, f)