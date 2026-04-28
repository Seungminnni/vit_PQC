#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
../.venv_lwe18/bin/python train_binary.py \
  --preset stage3_h3 \
  --secret_split \
  --train_secret_fraction 0.8 \
  --run_name binary_new_secret_split \
  "$@"

