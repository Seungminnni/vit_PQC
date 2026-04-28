#!/usr/bin/env bash
set -euo pipefail
if [ "$#" -lt 1 ]; then
  echo "usage: $0 CHECKPOINT [extra analyze_binary.py args...]" >&2
  exit 2
fi
checkpoint="$1"
shift
cd "$(dirname "$0")/.."
../.venv_lwe18/bin/python analyze_binary.py --checkpoint "$checkpoint" "$@"
