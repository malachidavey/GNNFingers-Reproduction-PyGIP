# GraphMI — DGL Implementation (PyGIP)

This folder contains a fully-correct, paper-aligned implementation of
**Model Inversion Attacks Against Graph Neural Networks**.

### Training

python scripts/train_dgl_tu_gcn.py --dataset ENZYMES


Produces:


dgl_checkpoints/ENZYMES_gcn.pt


### Attack



python scripts/run_graphmi.py
--dataset ENZYMES
--ckpt dgl_checkpoints/ENZYMES_gcn.pt


### Evaluation



python scripts/eval_graphmi.py


This reproduces the paper’s GraphMI baseline.