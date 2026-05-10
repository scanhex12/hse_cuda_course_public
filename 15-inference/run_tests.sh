#!/usr/bin/env bash
set -euo pipefail
TASK_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "${TASK_DIR}"

VENV="${1:-.venv}"
if [[ ! -d "$VENV" ]]; then
  python3 -m venv "$VENV"
fi
"${VENV}/bin/pip" install -q -r requirements-test.txt

"${VENV}/bin/python" -m grpc_tools.protoc \
  -I "${TASK_DIR}" \
  --python_out="${TASK_DIR}" \
  --grpc_python_out="${TASK_DIR}" \
  "${TASK_DIR}/inference.proto"

if [[ "${TRT15_UNIT_ONLY:-${INFERENCE_TEST_NO_E2E:-}}" == "1" ]]; then
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

if [[ "${TRT15_EXTERNAL:-${INFERENCE_TEST_EXTERNAL_SERVER:-}}" != "1" ]]; then
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
    echo "run_tests.sh: нет ${BUILD}/trt_server (соберите trt_server) или TRT15_UNIT_ONLY=1." >&2
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
  E2E_PORT="${TRT15_PORT:-${INFERENCE_E2E_PORT:-50051}}"
  "${BUILD}/trt_server" "${E2E_PORT}" "${ONNX}" &
  SERVER_PID=$!

  WAIT_STEP_S="${TRT15_WAIT_STEP:-${INFERENCE_SERVER_WAIT_STEP_S:-5}}"
  WAIT_MAX_S="${TRT15_WAIT_MAX:-${INFERENCE_SERVER_WAIT_MAX_S:-3600}}"
  echo "Waiting for trt_server on 127.0.0.1:${E2E_PORT} (up to ${WAIT_MAX_S}s; first build is slow)..."
  _deadline=$((SECONDS + WAIT_MAX_S))
  while true; do
    if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
      wait "${SERVER_PID}" || true
      echo "run_tests.sh: trt_server exited before opening ${E2E_PORT}; see logs above." >&2
      exit 1
    fi
    if python3 -c "import socket; s=socket.socket(); s.settimeout(1); s.connect(('127.0.0.1', int('${E2E_PORT}'))); s.close()" 2>/dev/null; then
      echo "trt_server is accepting connections on port ${E2E_PORT}."
      break
    fi
    if [ "${SECONDS}" -ge "${_deadline}" ]; then
      echo "run_tests.sh: timeout waiting for port ${E2E_PORT} (increase TRT15_WAIT_MAX?)." >&2
      exit 1
    fi
    sleep "${WAIT_STEP_S}"
  done
fi

E2E_PORT="${TRT15_PORT:-${INFERENCE_E2E_PORT:-50051}}"
export TRT15_E2E=1 INFERENCE_E2E=1 TRT15_BENCH=1 INFERENCE_BENCHMARK_E2E=1
export TRT15_PORT="${E2E_PORT}" INFERENCE_E2E_PORT="${E2E_PORT}"
"${VENV}/bin/python" -m unittest discover -s tests -v
