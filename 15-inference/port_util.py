#!/usr/bin/env python3
"""Port helpers for run_tests.sh (no extra deps)."""
from __future__ import annotations

import glob
import os
import re
import signal
import socket
import subprocess
import sys
import time


def port_hex(port: int) -> str:
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
                    if state == "0A" and local.endswith(":" + want):
                        inodes.add(inode)
        except OSError:
            pass
    return inodes


def server_pid_listening(pid: int, port: int) -> bool:
    inodes = listening_inodes(port)
    if not inodes:
        return False
    fd_dir = f"/proc/{pid}/fd"
    try:
        for fd in os.listdir(fd_dir):
            try:
                link = os.readlink(os.path.join(fd_dir, fd))
            except OSError:
                continue
            if link.startswith("socket:[") and link[8:-1] in inodes:
                return True
    except OSError:
        return False
    return False


def pids_on_port(port: int) -> list[int]:
    inodes = listening_inodes(port)
    pids: set[int] = set()
    if inodes:
        for proc in glob.glob("/proc/[0-9]*"):
            fd_dir = os.path.join(proc, "fd")
            try:
                proc_pid = int(os.path.basename(proc))
            except ValueError:
                continue
            try:
                for fd in os.listdir(fd_dir):
                    try:
                        link = os.readlink(os.path.join(fd_dir, fd))
                    except OSError:
                        continue
                    if link.startswith("socket:[") and link[8:-1] in inodes:
                        pids.add(proc_pid)
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

    try:
        out = subprocess.run(
            ["ss", "-tlnp"],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            check=False,
        )
        for line in out.stdout.splitlines():
            if not re.search(rf":{port}(?:\s|$)", line):
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
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def main() -> None:
    cmd = sys.argv[1]
    if cmd == "pick":
        print(pick_free_port())
        return
    if cmd == "open":
        raise SystemExit(0 if port_is_open(int(sys.argv[2])) else 1)
    if cmd == "pids":
        for pid in pids_on_port(int(sys.argv[2])):
            print(pid)
        return
    if cmd == "free":
        left = free_port(int(sys.argv[2]))
        for pid in left:
            print(pid)
        raise SystemExit(1 if left else 0)
    if cmd == "listening":
        ok = server_pid_listening(int(sys.argv[2]), int(sys.argv[3]))
        raise SystemExit(0 if ok else 1)
    raise SystemExit(f"unknown command: {cmd}")


if __name__ == "__main__":
    main()
