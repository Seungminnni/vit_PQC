#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
../.venv_lwe18/bin/python train_binary.py \
  --preset stage3_h3 \
  --topK 8 \
  --use_pair_filter \
  --pair_budget 16 \
  --run_name binary_pair_residual \
  "$@"

