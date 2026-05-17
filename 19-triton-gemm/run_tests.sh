#!/usr/bin/env bash
set -euo pipefail
TASK_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "${TASK_DIR}"

VENV="${1:-.venv}"
if [[ ! -d "${VENV}" ]]; then
  python3 -m venv "${VENV}"
fi
"${VENV}/bin/pip" install -q -r requirements-test.txt
"${VENV}/bin/python" -m pytest test.py -v
exec "${VENV}/bin/python" benchmark.py
