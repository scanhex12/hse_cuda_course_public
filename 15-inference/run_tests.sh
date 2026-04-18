#!/usr/bin/env bash
set -euo pipefail
TASK_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "${TASK_DIR}"

VENV="${1:-.venv}"
if [[ ! -d "$VENV" ]]; then
  python3 -m venv "$VENV"
fi
"${VENV}/bin/pip" install -q -r requirements-test.txt

if [[ "${INFERENCE_TEST_NO_E2E:-}" == "1" ]]; then
  exec "${VENV}/bin/python" -m unittest discover -s tests -v
fi

ROOT="$(cd "${TASK_DIR}/.." && pwd)"
BUILD="${ROOT}/build"
ONNX="${TASK_DIR}/resnet50.onnx"
SERVER_PID=""

cleanup() {
  if [[ -n "${SERVER_PID}" ]]; then
    kill "${SERVER_PID}" 2>/dev/null || true
    wait "${SERVER_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT

if [[ "${INFERENCE_TEST_EXTERNAL_SERVER:-}" != "1" ]]; then
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

  if [[ ! -f "${BUILD}/trt_server" ]]; then
    echo "run_tests.sh: нет ${BUILD}/trt_server (соберите trt_server) или задайте INFERENCE_TEST_NO_E2E=1." >&2
    exit 1
  fi
  if [[ ! -f "${ONNX}" ]]; then
    echo "run_tests.sh: нет ${ONNX}" >&2
    exit 1
  fi
  if [[ -z "${TENSORRT_ROOT:-}" || ! -f "${TENSORRT_ROOT}/include/NvInfer.h" ]]; then
    echo "run_tests.sh: задайте TENSORRT_ROOT или распакуйте TensorRT в ${TASK_DIR}/TensorRT-*" >&2
    exit 1
  fi

  export LD_LIBRARY_PATH="${TENSORRT_ROOT}/lib:${LD_LIBRARY_PATH:-}"
  E2E_PORT="${INFERENCE_E2E_PORT:-50051}"
  "${BUILD}/trt_server" "${E2E_PORT}" "${ONNX}" &
  SERVER_PID=$!
  sleep 15
fi

E2E_PORT="${INFERENCE_E2E_PORT:-50051}"
INFERENCE_E2E=1 INFERENCE_BENCHMARK_E2E=1 INFERENCE_E2E_PORT="${E2E_PORT}" \
  "${VENV}/bin/python" -m unittest discover -s tests -v
