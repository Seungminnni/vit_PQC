#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
../.venv_lwe18/bin/python train_binary.py --preset stage1_h1 --steps 1000 --topK 4 "$@"

