#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

../.venv_lwe18/bin/python train_binary.py \
  --preset stage3_h3 \
  --n 50 \
  --M 1000 \
  --q 127 \
  --h 3 \
  --secret_distribution bernoulli \
  --p_nonzero 0.06 \
  --sigma_e 2.0 \
  --steps 5000 \
  --batch_size 8 \
  --eval_batches 10 \
  --eval_every 200 \
  --loss_pos_weight_mode prior \
  --hfree_uncertain_K 12 \
  --hfree_threshold 0.5 \
  --amp \
  --amp_dtype bf16 \
  "$@"
