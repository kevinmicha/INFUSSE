"""
Module containing paths to the data, logs, scripts and checkpoints.

"""
import os

ADJACENCIES_DIR = os.environ.get('ADJACENCIES_DIR', '/Users/kevinmicha/Documents/all_structures/adjacencies_sparse/')
CHECKPOINTS_DIR = os.environ.get('CHECKPOINTS_DIR', '../checkpoints/')
CM_DIR = os.environ.get('CM_DIR', '/Users/kevinmicha/Documents/all_structures/contact_maps/')
DATA_DIR = os.environ.get('DATA_DIR', '../data/')
STRUCTURE_DIR = os.environ.get('STRUCTURE_DIR', '/Users/kevinmicha/Documents/all_structures/chothia_ext/')