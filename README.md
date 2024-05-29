# GCN-Bf

Go to `scripts` and run in the following order:
- `python generate_b_factors.py [--thr THR]` 
- `python run_strides.py` 
- `python generate_input.py` 
- `python generate_graph_arrays.py [--graphs GRAPHS]` 
- `python training_script.py [--graphs GRAPHS] [--lr LR] [--hidden_channels HIDDEN_CHANNELS] [--epochs EPOCHS]` 

For evaluation and explainability, once models and data are saved in the `checkpoints` folder:
- `python evaluation_script.py [--graphs GRAPHS] [--lr LR] [--hidden_channels HIDDEN_CHANNELS] [--epochs EPOCHS]` 
- `python antibody_explain.py [--glob] [--graphs GRAPHS] [--lr LR] [--hidden_channels HIDDEN_CHANNELS] [--epochs EPOCHS]` 