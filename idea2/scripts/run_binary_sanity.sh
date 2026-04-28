#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
../.venv_lwe18/bin/python binary_sanity_suite.py --preset stage3_h3 --batch_size 32 --random_trials 20 "$@"

