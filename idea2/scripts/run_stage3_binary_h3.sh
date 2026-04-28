#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
../.venv_lwe18/bin/python train_binary.py --preset stage3_h3 --steps 2000 --topK 8 "$@"

