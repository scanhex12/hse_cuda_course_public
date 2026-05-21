#!/usr/bin/env bash
set -euo pipefail
TASK_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "${TASK_DIR}"

VENV="${1:-.venv}"
if [[ ! -d "${VENV}" ]]; then
  python3 -m venv "${VENV}"
fi
"${VENV}/bin/pip" install -q -r requirements-test.txt
"${VENV}/bin/python" - <<'PY'
import sys
import torch

if not torch.cuda.is_available():
    print("CUDA is not available", file=sys.stderr)
    print("torch:", torch.__version__, file=sys.stderr)
    raise SystemExit(1)

print("torch", torch.__version__, "cuda", torch.version.cuda, torch.cuda.get_device_name(0))
PY
"${VENV}/bin/python" -m pytest test.py -v
exec "${VENV}/bin/python" benchmark.py
