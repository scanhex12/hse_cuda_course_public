#!/usr/bin/env bash
set -euo pipefail
if [[ -z "${TENSORRT_ROOT:-}" ]]; then
  echo "Укажите TENSORRT_ROOT." >&2
  exit 1
fi
export LD_LIBRARY_PATH="${TENSORRT_ROOT}/lib:${LD_LIBRARY_PATH:-}"
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
BUILD="${ROOT}/build"
TASK_DIR="${ROOT}/15-inference"
cd "${BUILD}"

if [[ ! -f trt_server ]]; then
  echo "Соберите trt_server: cmake --build ${ROOT}/build --target trt_server" >&2
  exit 1
fi

ONNX="${TASK_DIR}/resnet50.onnx"
if [[ ! -f "${ONNX}" ]]; then
  echo "Нет ${ONNX}. Запустите onnx_model.py в каталоге задачи." >&2
  exit 1
fi

echo "Запуск gRPC InferenceService на порту 50051..."
exec ./trt_server 50051 "${ONNX}"
