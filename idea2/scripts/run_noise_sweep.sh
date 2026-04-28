#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
for sigma in 0.5 1.0 1.5 2.0; do
  ../.venv_lwe18/bin/python train_binary.py --preset noise --sigma_e "$sigma" --run_name "noise_sigma_${sigma}" "$@"
done

