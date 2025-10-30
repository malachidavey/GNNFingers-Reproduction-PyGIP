#!/usr/bin/env bash
set -e

# === BASELINES ===
python train_tu_baseline.py --dataset ENZYMES  --arch GIN --epochs 300 --batch_size 128 --lr 0.01 --seed 3
python train_tu_baseline.py --dataset PROTEINS --arch GIN --epochs 300 --batch_size 128 --lr 0.01 --seed 2
python train_tu_baseline.py --dataset AIDS     --arch GIN --epochs 200 --batch_size 128 --lr 0.01 --seed 0

# === POSITIVE MODELS (F+) ===
# ENZYMES
python gen_pos_models.py --dataset ENZYMES  --arch GIN --baseline_ckpt results_tu/enzymes_gin_s3.pt  --mode ft_last  --epochs 50 --lr 1e-3 --seed 0
python gen_pos_models.py --dataset ENZYMES  --arch GIN --baseline_ckpt results_tu/enzymes_gin_s3.pt  --mode ft_all   --epochs 50 --lr 1e-3 --seed 1
python gen_pos_models.py --dataset ENZYMES  --arch GIN --baseline_ckpt results_tu/enzymes_gin_s3.pt  --mode prune    --prune_pct 0.2 --epochs 30 --lr 1e-3 --seed 2
python gen_pos_models.py --dataset ENZYMES  --arch GIN --baseline_ckpt results_tu/enzymes_gin_s3.pt  --mode distill  --T 2.0 --epochs 60 --lr 1e-3 --seed 3

# PROTEINS
python gen_pos_models.py --dataset PROTEINS --arch GIN --baseline_ckpt results_tu/proteins_gin_s2.pt --mode ft_last  --epochs 50 --lr 1e-3 --seed 0
python gen_pos_models.py --dataset PROTEINS --arch GIN --baseline_ckpt results_tu/proteins_gin_s2.pt --mode ft_all   --epochs 50 --lr 1e-3 --seed 1
python gen_pos_models.py --dataset PROTEINS --arch GIN --baseline_ckpt results_tu/proteins_gin_s2.pt --mode prune    --prune_pct 0.2 --epochs 30 --lr 1e-3 --seed 2
python gen_pos_models.py --dataset PROTEINS --arch GIN --baseline_ckpt results_tu/proteins_gin_s2.pt --mode distill  --T 2.0 --epochs 60 --lr 1e-3 --seed 3

# AIDS
python gen_pos_models.py --dataset AIDS     --arch GIN --baseline_ckpt results_tu/aids_gin_s0.pt     --mode ft_last  --epochs 30 --lr 1e-3 --seed 0
python gen_pos_models.py --dataset AIDS     --arch GIN --baseline_ckpt results_tu/aids_gin_s0.pt     --mode ft_all   --epochs 30 --lr 1e-3 --seed 1
python gen_pos_models.py --dataset AIDS     --arch GIN --baseline_ckpt results_tu/aids_gin_s0.pt     --mode prune    --prune_pct 0.2 --epochs 20 --lr 1e-3 --seed 2
python gen_pos_models.py --dataset AIDS     --arch GIN --baseline_ckpt results_tu/aids_gin_s0.pt     --mode distill  --T 2.0 --epochs 40 --lr 1e-3 --seed 3

# === NEGATIVE MODELS (F-) ===
python train_tu_baseline.py --dataset ENZYMES  --arch GIN --epochs 300 --batch_size 128 --lr 0.01 --seed 5 --out_dir results_bank
python train_tu_baseline.py --dataset ENZYMES  --arch GIN --epochs 300 --batch_size 128 --lr 0.01 --seed 6 --out_dir results_bank
python train_tu_baseline.py --dataset PROTEINS --arch GIN --epochs 300 --batch_size 128 --lr 0.01 --seed 7 --out_dir results_bank
python train_tu_baseline.py --dataset PROTEINS --arch GIN --epochs 300 --batch_size 128 --lr 0.01 --seed 8 --out_dir results_bank
python train_tu_baseline.py --dataset AIDS     --arch GIN --epochs 200 --batch_size 128 --lr 0.01 --seed 8 --out_dir results_bank
python train_tu_baseline.py --dataset AIDS     --arch GIN --epochs 200 --batch_size 128 --lr 0.01 --seed 9 --out_dir results_bank

# === UNIVERIFIER =
python train_univerifier.py
python eval_univerifier.py
