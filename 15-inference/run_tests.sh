#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
VENV="${1:-.venv}"
if [[ ! -d "$VENV" ]]; then
  python3 -m venv "$VENV"
fi
"$VENV/bin/pip" install -q -r requirements-test.txt
exec "$VENV/bin/python" -m unittest discover -s tests -v
