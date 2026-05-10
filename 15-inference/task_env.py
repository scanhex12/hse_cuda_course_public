from __future__ import annotations

import os
from dataclasses import dataclass


def _pick(new_key: str, legacy_key: str | None, default: str) -> str:
    v = os.environ.get(new_key)
    if v is not None and v != "":
        return v
    if legacy_key:
        v = os.environ.get(legacy_key)
        if v is not None and v != "":
            return v
    return default


@dataclass(frozen=True)
class E2ESettings:
    host: str
    port: int


def e2e_settings() -> E2ESettings:
    return E2ESettings(
        host=_pick("TRT15_HOST", "INFERENCE_E2E_HOST", "localhost"),
        port=int(_pick("TRT15_PORT", "INFERENCE_E2E_PORT", "50051")),
    )


@dataclass(frozen=True)
class BenchmarkSettings:
    duration_s: float
    threads: int
    load_base: float
    max_avg_latency_s: float
    max_max_latency_s: float


_AUTHOR_AVG = 0.0095
_AUTHOR_MAX = 0.0185


def benchmark_settings() -> BenchmarkSettings:
    return BenchmarkSettings(
        duration_s=float(_pick("TRT15_BENCH_SECONDS", "INFERENCE_BENCHMARK_DURATION_S", "1.0")),
        threads=int(_pick("TRT15_BENCH_THREADS", "INFERENCE_BENCHMARK_THREADS", "2")),
        load_base=float(_pick("TRT15_BENCH_LOAD", "INFERENCE_BENCHMARK_LOAD_BASE", "6.0")),
        max_avg_latency_s=float(
            _pick(
                "TRT15_BENCH_MAX_AVG_S",
                "INFERENCE_BENCHMARK_MAX_AVG_LATENCY_S",
                str(_AUTHOR_AVG * 2.0),
            )
        ),
        max_max_latency_s=float(
            _pick(
                "TRT15_BENCH_MAX_TAIL_S",
                "INFERENCE_BENCHMARK_MAX_MAX_LATENCY_S",
                str(_AUTHOR_MAX * 2.0),
            )
        ),
    )


def unit_tests_only() -> bool:
    return _pick("TRT15_UNIT_ONLY", "INFERENCE_TEST_NO_E2E", "0") in ("1", "true", "yes")


def e2e_tests_enabled() -> bool:
    return _pick("TRT15_E2E", "INFERENCE_E2E", "0") in ("1", "true", "yes")


def benchmark_e2e_enabled() -> bool:
    return _pick("TRT15_BENCH", "INFERENCE_BENCHMARK_E2E", "0") in ("1", "true", "yes")


def external_server() -> bool:
    return _pick("TRT15_EXTERNAL", "INFERENCE_TEST_EXTERNAL_SERVER", "0") in (
        "1",
        "true",
        "yes",
    )


def server_wait_step_s() -> int:
    return int(_pick("TRT15_WAIT_STEP", "INFERENCE_SERVER_WAIT_STEP_S", "5"))


def server_wait_max_s() -> int:
    return int(_pick("TRT15_WAIT_MAX", "INFERENCE_SERVER_WAIT_MAX_S", "3600"))
