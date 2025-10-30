# Reproducibility — Univerifier (PyGIP)

This README documents how to reproduce my Univerifier evaluation inside **PyGIP**. It includes exact commands, the dataset split files used for thresholding/evaluation, and the current results I obtained.

**Status:** ✅ Repro phase completed. I include two reproducible paths:
- **Path A (new evaluator):** supports `--thresh_mode target_fpr` (paper-style operating point). Requires a checkpoint that matches the evaluator’s architecture.
- **Path B (legacy evaluator):** uses **Youden’s J** for thresholding. This is what I just ran successfully with my current checkpoint.

---

## TL;DR

```bash
# (One-time) Create/activate env + deps
conda create -y -n pygip python=3.11
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate pygip
pip install -r requirements.txt    # If torch/pyg need special wheels, install per their docs.

# Verify fingerprints + lists
# - results_bank/   (fingerprints live here)
# - train.txt       (used to select threshold)
# - test.txt        (evaluation list)

# Path B — Legacy eval (works now; Youden J threshold)
python eval_univerifier_cli_legacy.py   --folder results_bank   --input_dim 4096   --zscore   --file_list test.txt   --train_list train.txt

# Collect plots into a summary archive
mkdir -p results_summary
for f in univerifier_eval_ru.png univerifier_eval_roc.png; do
  [ -f "results_bank/$f" ] && cp "results_bank/$f" results_summary/
done
tar -czf results_summary.tgz results_summary
```

---

## Environment

- Python: **3.11** (conda env `pygip`)
- Install base deps from `requirements.txt`. If your machine requires specific **PyTorch**/**PyG** wheels (CPU, CUDA, Apple MPS), follow their quick-start pages to install the correct versions.

---

## Data Layout & Splits

- **Fingerprints:** `results_bank/` (already present in this repo).
- **Split lists:**
  - `train.txt` — used **only** to pick a threshold (target-FPR or Youden J, depending on evaluator).
  - `test.txt` — evaluation list.
- Each list contains one path per line (relative or absolute). Update if your paths differ.

---

## Path A — New evaluator (paper-style target FPR)

This path uses `eval_univerifier_cli.py` with **target-FPR** thresholding. It requires a **checkpoint that matches the evaluator’s architecture** (e.g., input dim 8192, hidden sizes, etc.). If your current checkpoint doesn’t match, either locate a matching one or retrain.

```bash
# Example — adjust flags to match your checkpoint and script help
python eval_univerifier_cli.py   --folder results_bank   --input_dim 8192   --zscore   --file_list test.txt   --train_list train.txt   --thresh_mode target_fpr   --target_fpr 0.50   --ckpt train/univerifier_8192.pth
```

> If you see a `state_dict` mismatch (missing/unexpected keys or size mismatch), your checkpoint/model shapes differ. Either switch to the legacy evaluator (Path B) or retrain a checkpoint with `--input_dim` that matches your fingerprints.

---

## Path B — Legacy evaluator (Youden J) — **Working baseline**

Command:
```bash
python eval_univerifier_cli_legacy.py   --folder results_bank   --input_dim 4096   --zscore   --file_list test.txt   --train_list train.txt
```

Representative output (from my successful run):
```
Loaded 10 fingerprints  (F+: 6, F-: 4)
Computing threshold from TRAIN set (Youden J)...
Chosen threshold t* = 0.1331
✅ ARUC/AUC = 0.9583 | ACC@t* = 0.9000 | AP = 0.9762
    TPR@t* = 1.0000 | TNR@t* = 0.7500 | CM = [[3, 1], [0, 6]]
Plots: results_bank/univerifier_eval_ru.png , results_bank/univerifier_eval_roc.png
Curves saved to results_bank/univerifier_curves.npz
```

**Recorded metrics:**
- **AUC**: `0.9583`
- **ACC@t\***: `0.9000`
- **AP**: `0.9762`
- **TPR@t\***: `1.0000`
- **TNR@t\***: `0.7500`
- **t\*** (Youden J): `0.1331`
- **CM**: `[[3, 1], [0, 6]]`
- **Artifacts**: `results_bank/univerifier_eval_ru.png`, `results_bank/univerifier_eval_roc.png`, `results_bank/univerifier_curves.npz`

---

## Optional — Get operating point near FPR = 0.50 from saved curves

After the legacy run, you can compute the closest operating point to a target FPR using the saved ROC:

```bash
python scripts/compute_op_at_fpr.py --npz results_bank/univerifier_curves.npz --fpr 0.50
```

The script prints the nearest FPR/TPR and the associated threshold. If you also saved raw scores, you can re-bin at that threshold to recover a confusion matrix.

---

## Artifact collection

```bash
mkdir -p results_summary
for f in univerifier_eval_ru.png univerifier_eval_roc.png; do
  [ -f "results_bank/$f" ] && cp "results_bank/$f" results_summary/
done
tar -czf results_summary.tgz results_summary
```

---

## Troubleshooting

- **ModuleNotFoundError (e.g., matplotlib):** `pip install matplotlib`
- **Torch/PyG mismatch or missing wheels:** install per your platform’s quick-start page.
- **Checkpoint shape mismatch:** either (a) switch to legacy evaluator (Path B), or (b) retrain/find a checkpoint that matches `eval_univerifier_cli.py`’s architecture (e.g., `--input_dim 8192`), then pass it via `--ckpt`.
- **Different plot filenames:** scripts may emit `s{seed}_f0.50_...png`; adapt the collector to grab those variants as needed.

---

## License / Attribution

This repo contains my reproducibility notes and glue instructions. Please refer to the upstream **PyGIP** project for implementation details and cite the original authors/paper.
