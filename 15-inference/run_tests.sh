#!/usr/bin/env bash
set -euo pipefail
TASK_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "${TASK_DIR}"

VENV="${1:-.venv}"
PORT_UTIL="${TASK_DIR}/port_util.py"

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

if [[ "${TRT15_EXTERNAL:-${INFERENCE_TEST_EXTERNAL_SERVER:-}}" == "1" ]]; then
  echo "run_tests.sh: TRT15_EXTERNAL игнорируется — сервер всегда поднимается этим скриптом." >&2
fi

ROOT="$(cd "${TASK_DIR}/.." && pwd)"
BUILD="${ROOT}/build"
ONNX="${TASK_DIR}/resnet50.onnx"
SERVER_PID=""

py_port() {
  python3 "${PORT_UTIL}" "$@"
}

pick_free_port() {
  py_port pick
}

server_listening() {
  py_port listening "$1" "$2"
}

stop_server() {
  local pid="${1:-}"
  [[ -n "${pid}" ]] || return 0
  if ! kill -0 "${pid}" 2>/dev/null; then
    return 0
  fi
  kill -TERM "${pid}" 2>/dev/null || true
  for _ in $(seq 1 20); do
    if ! kill -0 "${pid}" 2>/dev/null; then
      wait "${pid}" 2>/dev/null || true
      return 0
    fi
    sleep 0.25
  done
  kill -KILL "${pid}" 2>/dev/null || true
  wait "${pid}" 2>/dev/null || true
}

E2E_PORT="$(pick_free_port)"
echo "run_tests.sh: port ${E2E_PORT}." >&2

cleanup() {
  if [[ -n "${SERVER_PID}" ]]; then
    stop_server "${SERVER_PID}"
    SERVER_PID=""
  fi
}
trap cleanup EXIT INT TERM

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
"${BUILD}/trt_server" "${E2E_PORT}" "${ONNX}" &
SERVER_PID=$!

WAIT_STEP_S="${TRT15_WAIT_STEP:-${INFERENCE_SERVER_WAIT_STEP_S:-5}}"
WAIT_MAX_S="${TRT15_WAIT_MAX:-${INFERENCE_SERVER_WAIT_MAX_S:-3600}}"
echo "Waiting for trt_server pid ${SERVER_PID} on 127.0.0.1:${E2E_PORT} (up to ${WAIT_MAX_S}s)..."
_deadline=$((SECONDS + WAIT_MAX_S))
while true; do
  if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
    wait "${SERVER_PID}" || true
    echo "run_tests.sh: trt_server exited before listening on ${E2E_PORT}." >&2
    exit 1
  fi
  if server_listening "${SERVER_PID}" "${E2E_PORT}"; then
    echo "trt_server pid ${SERVER_PID} listens on 127.0.0.1:${E2E_PORT}."
    break
  fi
  if [ "${SECONDS}" -ge "${_deadline}" ]; then
    echo "run_tests.sh: timeout waiting for pid ${SERVER_PID} to listen on ${E2E_PORT}." >&2
    exit 1
  fi
  sleep "${WAIT_STEP_S}"
done

export TRT15_E2E=1 INFERENCE_E2E=1 TRT15_BENCH=1 INFERENCE_BENCHMARK_E2E=1
export TRT15_PORT="${E2E_PORT}" INFERENCE_E2E_PORT="${E2E_PORT}"
"${VENV}/bin/python" -m unittest discover -s tests -v
