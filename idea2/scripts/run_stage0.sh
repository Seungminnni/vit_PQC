#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
../.venv_lwe18/bin/python train_binary.py --preset stage0 --steps 200 --topK 4 "$@"

