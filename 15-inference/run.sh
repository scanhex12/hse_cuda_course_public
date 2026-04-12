#!/usr/bin/env bash
set -euo pipefail
if [[ -z "${TENSORRT_ROOT:-}" ]]; then
  echo "Укажите TENSORRT_ROOT." >&2
  exit 1
fi
export LD_LIBRARY_PATH="${TENSORRT_ROOT}/lib:${LD_LIBRARY_PATH:-}"
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
exec "${ROOT}/build/trt_server" "$@"
