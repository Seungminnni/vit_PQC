#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
../.venv_lwe18/bin/python train_binary.py --preset stage2_h2 --steps 1500 --topK 6 "$@"

