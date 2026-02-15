# GraphMI Attacks and Defenses in PyGIP

## Overview

This project studies **information leakage in Graph Neural Networks (GNNs)** through **Graph Model Inversion (GraphMI)** attacks and evaluates lightweight **training-time defenses** that reduce reconstruction accuracy.

Using the **PyGIP** framework, this work reproduces GraphMI end-to-end on standard TU graph datasets and evaluates how common regularization-style defenses affect an attacker’s ability to reconstruct sensitive graph information.

The goal is to empirically validate **when defenses reduce leakage and when they fail**, in line with prior model inversion literature.

---

## Threat Model

- **Victim**: A GNN (GCN) trained for graph classification.
- **Adversary**: Has access to the trained model and applies GraphMI to reconstruct graph structure and/or node features.
- **Evaluation**: Reconstruction success is measured using **AUC** and **Average Precision (AP)**.

---

## Attacks

This repository reproduces the **GraphMI / Univerifier** attack pipeline using PyGIP.

- Fingerprints are extracted from trained victim models.
- A verifier is trained to distinguish real vs. reconstructed graphs.
- Thresholding is performed using either:
  - **Target FPR (paper-style)**, or
  - **Youden’s J statistic (legacy evaluator)**.

Attack evaluation runs successfully across:
- **ENZYMES**
- **PROTEINS**
- **AIDS**

---

## Defenses

In addition to attack reproduction, this project evaluates **training-time defenses** applied to the victim GNN.

### Implemented Defenses

- **DropEdge**
  - Randomly removes a fraction of edges during training.
  - Weakens structural signals exploited by GraphMI.
  - Implemented directly in the DGL GCN forward pass.

- **Feature Masking**
  - Randomly masks node features during training.
  - Reduces direct feature leakage.

- **Gaussian Feature Noise**
  - Adds Gaussian noise to node features during training.
  - Acts as a lightweight privacy-preserving perturbation  
    (not formal differential privacy).

Defenses are enabled via CLI flags during victim training.

---

## Experimental Setup

- **Models**: GCN (DGL implementation)
- **Datasets**: ENZYMES, PROTEINS, AIDS (TU benchmarks)
- **Metrics**: AUC, AP (GraphMI reconstruction performance)
- **Comparisons**:
  - Baseline (no defense)
  - DropEdge
  - DropEdge + Feature Masking
  - DropEdge + Feature Masking + Gaussian Noise

---

## Results Summary

### DropEdge vs. Baseline (GraphMI Reconstruction)

| Dataset   | Model                | AUC    | AP     |
|-----------|----------------------|--------|--------|
| ENZYMES   | Baseline GCN         | 0.5418 | 0.1397 |
|           | GCN + DropEdge (0.2) | 0.4941 | 0.1249 |
| PROTEINS  | Baseline GCN         | 0.5300 | 0.1015 |
|           | GCN + DropEdge (0.2) | 0.4823 | 0.0919 |
| AIDS      | Baseline GCN         | 0.5027 | 0.0493 |
|           | GCN + DropEdge (0.2) | 0.5027 | 0.0493 |

**Key Insight:**  
DropEdge consistently reduces GraphMI performance on **structure-sensitive datasets** (ENZYMES, PROTEINS) and has **negligible impact on AIDS**, which is dominated by strong node features. This behavior matches prior findings in the model inversion literature.

---

## Interpretation

- DropEdge is effective when graph structure is the primary leakage channel.
- Feature-dominated datasets remain vulnerable to structural defenses.
- Lightweight defenses reduce—but do not eliminate—model inversion risk.

This highlights the trade-off between model utility, structural regularization, and privacy.

---

# Reproducibility — Univerifier (PyGIP)

This section documents the exact steps used to reproduce the **Univerifier attack evaluation** reported in this repository.

**Status:** ✅ Reproducibility phase completed.

Two reproducible paths are supported:
- **Path A (new evaluator):** paper-style target-FPR thresholding.
- **Path B (legacy evaluator):** Youden’s J thresholding (verified working).

---

## TL;DR

```bash
# Create environment
conda create -y -n pygip python=3.11
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate pygip
pip install -r requirements.txt
```

```bash
# Legacy evaluator (working baseline)
python eval_univerifier_cli_legacy.py \
  --folder results_bank \
  --input_dim 4096 \
  --zscore \
  --file_list test.txt \
  --train_list train.txt
```

---

## Environment

* Python: **3.11**
* Frameworks: PyTorch, DGL, PyGIP
  (Install platform-specific PyTorch / PyG wheels as needed.)

---

## Data Layout & Splits

* **Fingerprints:** `results_bank/`
* **Split lists:**

  * `train.txt` — used only for threshold selection.
  * `test.txt` — evaluation list.

Fingerprints and metrics are **not tracked in Git** and must be generated locally.

Expected layout:

```
results_bank/
├── *.pt
├── *.metrics.txt
├── univerifier_eval_ru.png
├── univerifier_eval_roc.png
└── univerifier_curves.npz
```

---

## Path A — New Evaluator (Target FPR)

Requires a checkpoint matching the evaluator architecture.

```bash
python eval_univerifier_cli.py \
  --folder results_bank \
  --input_dim 8192 \
  --zscore \
  --file_list test.txt \
  --train_list train.txt \
  --thresh_mode target_fpr \
  --target_fpr 0.50 \
  --ckpt train/univerifier_8192.pth
```

If a checkpoint mismatch occurs, use Path B or retrain with matching dimensions.

---

## Path B — Legacy Evaluator (Youden’s J)

```bash
python eval_univerifier_cli_legacy.py \
  --folder results_bank \
  --input_dim 4096 \
  --zscore \
  --file_list test.txt \
  --train_list train.txt
```

Representative output:

```
AUC = 0.9583 | ACC@t* = 0.9000 | AP = 0.9762
TPR@t* = 1.0000 | TNR@t* = 0.7500 | CM = [[3, 1], [0, 6]]
```

---

## Troubleshooting

* **Checkpoint mismatch:** Use legacy evaluator or retrain with matching architecture.
* **Missing dependencies:** Install via `pip install <package>`.
* **Different artifact names:** Adjust collection scripts accordingly.

---

## Project Status

* ✔ GraphMI attacks reproduced end-to-end
* ✔ DropEdge, feature masking, and noise defenses implemented
* ✔ Results match expected trends in prior literature
* ✔ Code merged, clean, and documented
* ✔ Ready for board review and presentation

---

## Attribution

This repository contains reproducibility notes and experimental extensions built on **PyGIP**.
Please cite the original PyGIP authors and GraphMI paper when using this work.
