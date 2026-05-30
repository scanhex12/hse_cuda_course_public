#!/usr/bin/env python3
from __future__ import annotations

import os
import sys
import time

import torch

from kernel import matmul_int8, quantize_int8, quantize_int8_perrow

HIDDEN = 2048
INTER = 11008
BS = 32
PREFILL_LEN = 512
DECODE_LEN = 1
TP = 1

ITERS_QUANT = int(os.environ.get("HSE_PERF_ITERS_QUANT", "256"))
ITERS_MATMUL = int(os.environ.get("HSE_PERF_ITERS_MATMUL", "512"))


def perf_quantize(M: int, K: int, iters: int = ITERS_QUANT, thr: float = 1.3) -> tuple[float, float]:
    torch.manual_seed(0)
    a = torch.randn((M, K), device="cuda", dtype=torch.float16)

    for _ in range(10):
        quantize_int8(a, axis=1)
        quantize_int8_perrow(a)
    torch.cuda.synchronize()

    t0 = time.time()
    for _ in range(iters):
        quantize_int8_perrow(a)
    torch.cuda.synchronize()
    t1 = time.time()
    for _ in range(iters):
        quantize_int8(a, axis=1)
    torch.cuda.synchronize()
    t2 = time.time()

    torch_time = (t2 - t1) / iters
    triton_time = (t1 - t0) / iters
    speedup = torch_time / triton_time
    if speedup <= thr:
        raise AssertionError(
            f"quantize: need {thr}x speedup, got {speedup:.3f}x "
            f"(torch={torch_time:.6f}s triton={triton_time:.6f}s)"
        )
    print(f"  quantize M={M} K={K}: {speedup:.2f}x (thr {thr})")
    return triton_time, torch_time


def perf_matmul_int8(
    M: int,
    K: int,
    N: int,
    iters: int = ITERS_MATMUL,
    thr: float = 0.95,
) -> tuple[float, float, float]:
    torch.manual_seed(0)
    a = torch.randn((M, K), device="cuda", dtype=torch.float16)
    b = torch.randn((K, N), device="cuda", dtype=torch.float16).contiguous()
    int_b, scale_b = quantize_int8(b, axis=0)

    for _ in range(10):
        int_a, a_scale = quantize_int8_perrow(a)
        matmul_int8(int_a, a_scale, int_b, scale_b)
    torch.cuda.synchronize()

    t0 = time.time()
    for _ in range(iters):
        int_a, a_scale = quantize_int8_perrow(a)
    torch.cuda.synchronize()
    qt1 = time.time()
    for _ in range(iters):
        matmul_int8(int_a, a_scale, int_b, scale_b)
    torch.cuda.synchronize()
    t1 = time.time()

    quant_time = qt1 - t0
    triton_time = t1 - qt1

    for _ in range(10):
        torch.matmul(a, b)
    torch.cuda.synchronize()

    t2 = time.time()
    for _ in range(iters):
        torch.matmul(a, b)
    torch.cuda.synchronize()
    t3 = time.time()
    torch_time = t3 - t2

    speedup = torch_time / triton_time
    if speedup <= thr:
        raise AssertionError(
            f"matmul M={M} K={K} N={N}: need {thr}x speedup, got {speedup:.3f}x "
            f"(torch={torch_time / iters:.6f}s triton={triton_time / iters:.6f}s)"
        )
    print(f"  matmul M={M} K={K} N={N}: {speedup:.2f}x (thr {thr})")
    return triton_time, torch_time, quant_time


def perf_model_layer(
    bs: int,
    seq_len: int,
    hidden: int,
    inter: int,
    tp: int,
    thr: float = 0.95,
    label: str = "",
) -> None:
    st1 = 0.0
    st2 = 0.0
    m = bs * seq_len
    print(f"model_layer {label} (bs={bs} seq={seq_len} thr={thr}):")

    t1, t2, _ = perf_matmul_int8(m, hidden, hidden * 3 // tp, thr=thr)
    perf_quantize(m, hidden, thr=thr)
    st1 += t1
    st2 += t2

    t1, t2, _ = perf_matmul_int8(m, hidden // tp, hidden, thr=thr)
    perf_quantize(m, hidden // tp, thr=thr)
    st1 += t1
    st2 += t2

    t1, t2, _ = perf_matmul_int8(m, hidden, inter * 2 // tp, thr=thr)
    st1 += t1
    st2 += t2

    t1, t2, _ = perf_matmul_int8(m, inter // tp, hidden, thr=thr)
    perf_quantize(m, inter // tp, thr=thr)
    st1 += t1
    st2 += t2

    speedup = st2 / st1
    if speedup <= thr:
        raise AssertionError(
            f"layer {label}: need {thr}x speedup, got {speedup:.3f}x (triton={st1} torch={st2})"
        )
    print(f"  layer total: {speedup:.2f}x (thr {thr})")


def run_all() -> None:
    print("benchmark: quantize (layer shape)")
    perf_quantize(BS * PREFILL_LEN, HIDDEN)

    print("benchmark: matmul (QKV shape)")
    perf_matmul_int8(BS * PREFILL_LEN, HIDDEN, HIDDEN * 3 // TP)

    print("benchmark: model layer prefill")
    perf_model_layer(BS, PREFILL_LEN, HIDDEN, INTER, TP, thr=1.0, label="prefill")

    print("benchmark: model layer decode")
    perf_model_layer(BS, DECODE_LEN, HIDDEN, INTER, TP, thr=0.1, label="decode")


def main():
    run_all()


if __name__ == "__main__":
    sys.exit(main())
