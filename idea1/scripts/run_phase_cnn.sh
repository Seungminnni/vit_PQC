#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
../.venv_lwe18/bin/python -m src.train --config configs/cnn_phase_binary.yaml
