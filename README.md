# GCN-Bf

Request four folders of data from the authors:
- `adjacencies_sparse` (go to `gcn_bf/config.py` and update `ADJACENCIES_DIR` accordingly)
- `contact_maps` (go to `gcn_bf/config.py` and update `CM_DIR` accordingly)
- `chothia_ext` (go to `gcn_bf/config.py` and update `STRUCTURE_DIR` accordingly)
- `strides_outputs` (place this folder in `gcn_bf/data/`) 

Go to `scripts` and run in the following order (graphs can be `gmn` or `bagpype`):
- `python generate_b_factors.py [--thr THR]` 
- `python generate_input.py` 
- `python generate_graph_arrays.py [--graphs GRAPHS]` 
- `python training_script.py [--graphs GRAPHS] [--lr LR] [--hidden_channels HIDDEN_CHANNELS] [--epochs EPOCHS]` 

For evaluation and explainability, once models and data are saved in the `checkpoints` folder:
- `python evaluation_script.py [--graphs GRAPHS] [--lr LR] [--hidden_channels HIDDEN_CHANNELS] [--epochs EPOCHS]` 
- `python antibody_explain.py [--glob] [--graphs GRAPHS] [--lr LR] [--hidden_channels HIDDEN_CHANNELS] [--epochs EPOCHS]` 