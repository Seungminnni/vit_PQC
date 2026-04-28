#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
../.venv_lwe18/bin/python -m src.train --config configs/transformer_rhie_binary.yaml
