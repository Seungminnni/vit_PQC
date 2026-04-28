#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

../.venv_lwe18/bin/python train_binary.py \
  --preset stage3_h3 \
  --n 50 \
  --M 1600 \
  --q 127 \
  --h 3 \
  --sigma_e 0 \
  --steps 5000 \
  --batch_size 8 \
  --topK 20 \
  --secret_split \
  --train_secret_fraction 0.8 \
  --amp \
  --amp_dtype bf16 \
  --seed 0 \
  "$@"
