#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
../.venv_lwe18/bin/python ablation.py --experiment feature --preset stage3_h3 --steps 500 "$@"

