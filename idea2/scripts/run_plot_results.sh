#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
../.venv_lwe18/bin/python plot_results.py "$@"
