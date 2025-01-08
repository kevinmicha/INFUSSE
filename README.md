# GCN-Bf

1. Create a conda environment, install all the dependencies, and then: 
```bash
git clone https://github.com/kevinmicha/GCN-Bf
cd GCN-Bf
pip install .
```

List of dependencies:
* `matplotlib`
* `numpy`
* `pandas`
* `requests`
* `scikit-learn`
* `scipy`
* `torch`
* `torch_geometric`
* `transformers`

2. Request three folders of data from the authors:
- `adjacencies_sparse` (go to `gcn_bf/config.py` and update `ADJACENCIES_DIR` accordingly)
- `contact_maps` (go to `gcn_bf/config.py` and update `CM_DIR` accordingly)
- `chothia_ext` (go to `gcn_bf/config.py` and update `STRUCTURE_DIR` accordingly)

3. Go to `scripts` and run in the following order (graphs can be `gnm` or `bagpype`):
- `python generate_b_factors.py [--thr THR]` 
- `python generate_input.py` 
- `python generate_graph_arrays.py [--graphs GRAPHS]` 
- `python training_script.py [--graphs GRAPHS] [--lr LR] [--hidden_channels HIDDEN_CHANNELS] [--epochs EPOCHS] [--seq_only]` 

4. For evaluation and interpretability, once models and data are saved in the `checkpoints` folder:
- `python evaluation_script.py [--graphs GRAPHS] [--lr LR] [--hidden_channels HIDDEN_CHANNELS] [--epochs EPOCHS]` 
- `python antibody_explain.py [--glob] [--graphs GRAPHS] [--lr LR] [--hidden_channels HIDDEN_CHANNELS] [--epochs EPOCHS]` 

5. Further insights can be found in the notebook [Interpretability](https://github.com/kevinmicha/GCN-Bf/blob/main/notebooks/Interpretability.ipynb).