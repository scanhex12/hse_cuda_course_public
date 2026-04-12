#!/usr/bin/env bash
set -euo pipefail
TASK_DIR="$(cd "$(dirname "$0")" && pwd)"
if [[ -z "${TENSORRT_ROOT:-}" ]]; then
  shopt -s nullglob
  for _d in "${TASK_DIR}"/TensorRT-*; do
    if [[ -f "${_d}/include/NvInfer.h" ]]; then
      TENSORRT_ROOT="${_d}"
      break
    fi
  done
  shopt -u nullglob
fi
if [[ -z "${TENSORRT_ROOT:-}" ]]; then
  echo "Set TENSORRT_ROOT or unpack SDK to ${TASK_DIR}/TensorRT-*" >&2
  exit 1
fi
export LD_LIBRARY_PATH="${TENSORRT_ROOT}/lib:${LD_LIBRARY_PATH:-}"
ROOT="$(cd "${TASK_DIR}/.." && pwd)"
BUILD="${ROOT}/build"
cd "${BUILD}"

if [[ ! -f trt_server ]]; then
  echo "Build trt_server: cmake --build ${ROOT}/build --target trt_server" >&2
  exit 1
fi

ONNX="${TASK_DIR}/resnet50.onnx"
if [[ ! -f "${ONNX}" ]]; then
  echo "No ${ONNX}. Run onnx_model.py in task directory." >&2
  exit 1
fi

echo "Starting gRPC InferenceService on port 50051..."
exec ./trt_server 50051 "${ONNX}"
