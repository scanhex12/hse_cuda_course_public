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

# Port helpers via /proc — работают без root, в отличие от ss -p / fuser на CI.
py_port() {
  python3 - "$@" <<'PY'
import glob
import os
import re
import signal
import socket
import subprocess
import sys
import time

cmd = sys.argv[1]


def port_hex(port: int) -> str:
    # В /proc/net/tcp{,6} порт записан как hex-значение без htons (50051 -> C383).
    return f"{port:04X}"


def listening_inodes(port: int) -> set[str]:
    want = port_hex(port)
    inodes: set[str] = set()
    for path in ("/proc/net/tcp", "/proc/net/tcp6"):
        try:
            with open(path, encoding="ascii") as fh:
                next(fh, None)
                for line in fh:
                    cols = line.split()
                    if len(cols) < 10:
                        continue
                    local, state, inode = cols[1], cols[3], cols[9]
                    if state != "0A":
                        continue
                    if local.endswith(":" + want):
                        inodes.add(inode)
        except OSError:
            pass
    return inodes


def pids_on_port(port: int) -> list[int]:
    inodes = listening_inodes(port)
    pids: set[int] = set()
    if inodes:
        for proc in glob.glob("/proc/[0-9]*"):
            fd_dir = os.path.join(proc, "fd")
            try:
                pid = int(os.path.basename(proc))
            except ValueError:
                continue
            try:
                for fd in os.listdir(fd_dir):
                    try:
                        link = os.readlink(os.path.join(fd_dir, fd))
                    except OSError:
                        continue
                    if not link.startswith("socket:["):
                        continue
                    if link[8:-1] in inodes:
                        pids.add(pid)
            except OSError:
                continue

    if pids:
        return sorted(pids)

    for cmd in (
        ["fuser", f"{port}/tcp"],
        ["lsof", "-t", "-i", f":{port}", "-sTCP:LISTEN"],
    ):
        try:
            out = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                check=False,
            )
        except OSError:
            continue
        for token in re.split(r"\s+", out.stdout.strip()):
            if token.isdigit():
                pids.add(int(token))

    if pids:
        return sorted(pids)

    try:
        out = subprocess.run(
            ["ss", "-tlnp"],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            check=False,
        )
        for line in out.stdout.splitlines():
            if f":{port} " not in line:
                continue
            for match in re.finditer(r"pid=(\d+)", line):
                pids.add(int(match.group(1)))
    except OSError:
        pass

    return sorted(pids)


def port_is_open(port: int) -> bool:
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=1.0):
            return True
    except OSError:
        return False


def stop_pid(pid: int) -> None:
    try:
        os.kill(pid, 0)
    except OSError:
        return
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.kill(pid, sig)
        except OSError:
            return
        for _ in range(20):
            try:
                os.kill(pid, 0)
            except OSError:
                return
            time.sleep(0.25)


def free_port(port: int) -> list[int]:
    remaining: list[int] = []
    for _ in range(3):
        pids = pids_on_port(port)
        if not pids:
            break
        for pid in pids:
            stop_pid(pid)
        time.sleep(0.5)
        remaining = pids_on_port(port)
        if not remaining:
            break
    return remaining


def pick_free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


if cmd == "pids":
    for pid in pids_on_port(int(sys.argv[2])):
        print(pid)
elif cmd == "open":
    raise SystemExit(0 if port_is_open(int(sys.argv[2])) else 1)
elif cmd == "free":
    left = free_port(int(sys.argv[2]))
    for pid in left:
        print(pid)
    raise SystemExit(1 if left else 0)
elif cmd == "owned":
    port = int(sys.argv[2])
    want = int(sys.argv[3])
    raise SystemExit(0 if want in pids_on_port(port) else 1)
elif cmd == "pick":
    print(pick_free_port())
else:
    raise SystemExit(f"unknown command: {cmd}")
PY
}

port_pids() {
  py_port pids "$1"
}

port_is_open() {
  py_port open "$1"
}

port_owned_by() {
  py_port owned "$1" "$2"
}

pick_free_port() {
  py_port pick
}

free_port() {
  local port="$1"
  local left=""
  left="$(py_port free "${port}" 2>&1 || true)"
  if [[ -n "${left}" ]]; then
    echo "run_tests.sh: не удалось освободить порт ${port}, процессы: ${left//$'\n'/ }" >&2
    return 1
  fi
  return 0
}

stop_server() {
  local pid="${1:-}"
  [[ -n "${pid}" ]] || return 0
  python3 - "${pid}" <<'PY'
import os, signal, sys, time
pid = int(sys.argv[1])
for sig in (signal.SIGTERM, signal.SIGKILL):
    try:
        os.kill(pid, sig)
    except OSError:
        break
    for _ in range(20):
        try:
            os.kill(pid, 0)
        except OSError:
            raise SystemExit(0)
        time.sleep(0.25)
PY
  wait "${pid}" 2>/dev/null || true
}

E2E_PORT_EXPLICIT=0
if [[ -n "${TRT15_PORT:-}" || -n "${INFERENCE_E2E_PORT:-}" ]]; then
  E2E_PORT_EXPLICIT=1
  E2E_PORT="${TRT15_PORT:-${INFERENCE_E2E_PORT:-50051}}"
else
  E2E_PORT=50051
fi

cleanup() {
  if [[ -n "${SERVER_PID}" ]]; then
    stop_server "${SERVER_PID}"
    SERVER_PID=""
  fi
  py_port free "${E2E_PORT}" >/dev/null 2>&1 || true
}
trap cleanup EXIT INT TERM

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

  if ! free_port "${E2E_PORT}"; then
    _port_busy=1
  elif port_is_open "${E2E_PORT}"; then
    _port_busy=1
  else
    _port_busy=0
  fi

  if [[ "${_port_busy}" == "1" ]]; then
    stale="$(port_pids "${E2E_PORT}" | tr '\n' ' ')"
    if [[ -n "${stale}" ]]; then
      echo "run_tests.sh: порт ${E2E_PORT} занят процессами: ${stale}" >&2
    else
      echo "run_tests.sh: порт ${E2E_PORT} занят (pid владельца неизвестен — другой пользователь?)" >&2
    fi
    if [[ "${E2E_PORT_EXPLICIT}" == "1" ]]; then
      echo "run_tests.sh: задан TRT15_PORT=${E2E_PORT}, переключиться на другой порт нельзя." >&2
      exit 1
    fi
    E2E_PORT="$(pick_free_port)"
    echo "run_tests.sh: используем свободный порт ${E2E_PORT}." >&2
  fi

  "${BUILD}/trt_server" "${E2E_PORT}" "${ONNX}" &
  SERVER_PID=$!

  WAIT_STEP_S="${TRT15_WAIT_STEP:-${INFERENCE_SERVER_WAIT_STEP_S:-5}}"
  WAIT_MAX_S="${TRT15_WAIT_MAX:-${INFERENCE_SERVER_WAIT_MAX_S:-3600}}"
  echo "Waiting for trt_server (pid ${SERVER_PID}) on 127.0.0.1:${E2E_PORT} (up to ${WAIT_MAX_S}s; first build is slow)..."
  _deadline=$((SECONDS + WAIT_MAX_S))
  while true; do
    if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
      wait "${SERVER_PID}" || true
      echo "run_tests.sh: trt_server exited before opening ${E2E_PORT}; see logs above." >&2
      exit 1
    fi
    if port_owned_by "${E2E_PORT}" "${SERVER_PID}"; then
      echo "trt_server (pid ${SERVER_PID}) is listening on port ${E2E_PORT}."
      break
    fi
    if port_is_open "${E2E_PORT}"; then
      stale="$(port_pids "${E2E_PORT}" | tr '\n' ' ')"
      echo "run_tests.sh: порт ${E2E_PORT} занят чужим процессом (${stale:-?}), ждём наш pid ${SERVER_PID}..." >&2
    fi
    if [ "${SECONDS}" -ge "${_deadline}" ]; then
      stale="$(port_pids "${E2E_PORT}" | tr '\n' ' ')"
      echo "run_tests.sh: timeout; порт ${E2E_PORT} слушает ${stale:-?}, наш pid ${SERVER_PID} так и не занял порт." >&2
      exit 1
    fi
    sleep "${WAIT_STEP_S}"
  done
else
  if ! port_is_open "${E2E_PORT}"; then
    echo "run_tests.sh: TRT15_EXTERNAL=1, но на 127.0.0.1:${E2E_PORT} никто не слушает." >&2
    exit 1
  fi
  echo "run_tests.sh: используем trt_server на 127.0.0.1:${E2E_PORT} (TRT15_EXTERNAL=1)." >&2
fi

export TRT15_E2E=1 INFERENCE_E2E=1 TRT15_BENCH=1 INFERENCE_BENCHMARK_E2E=1
export TRT15_PORT="${E2E_PORT}" INFERENCE_E2E_PORT="${E2E_PORT}"
"${VENV}/bin/python" -m unittest discover -s tests -v
