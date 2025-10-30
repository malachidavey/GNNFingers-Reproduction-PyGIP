#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

FOLDER="${1:-results_bank}"
TRAIN_SEEDS="${2:-0,1,5,6,7}"
TEST_SEEDS="${3:-2,3,8,9}"
DIM="${4:-8192}"

python make_split_lso.py --folder "$FOLDER" --train_seeds "$TRAIN_SEEDS" --test_seeds "$TEST_SEEDS" --shuffle
python train_univerifier_cli.py --folder "$FOLDER" --train_list train.txt --test_list test.txt \
  --input_dim "$DIM" --zscore --epochs 20 --batch_size 32 --wd 2e-3
python eval_univerifier_cli.py --folder "$FOLDER" --input_dim "$DIM" --zscore

