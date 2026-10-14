#!/usr/bin/env bash
# Check real CUDA execution before opening the app; no named Conda environment is required.
set -euo pipefail
cd -- "$(dirname -- "$0")"
bash run.sh --check --require-cuda
exec bash run.sh "$@"
