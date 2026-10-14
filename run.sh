#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
if [ -n "${VIRTUAL_ENV:-}${CONDA_PREFIX:-}" ] && command -v python >/dev/null 2>&1; then
  exec python app.py "$@"
elif [ -x .venv/bin/python ]; then
  exec .venv/bin/python app.py "$@"
elif command -v python3 >/dev/null 2>&1; then
  exec python3 app.py "$@"
else
  echo "Install Python 3.10-3.13, then run: python3 tools/setup_environment.py"
  exit 1
fi
