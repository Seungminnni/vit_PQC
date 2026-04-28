#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
../.venv_lwe18/bin/python generate_run_artifacts.py "$@"
