import sys
import unittest
from pathlib import Path

_TASK_ROOT = Path(__file__).resolve().parents[1]
if str(_TASK_ROOT) not in sys.path:
    sys.path.insert(0, str(_TASK_ROOT))

import client_benchmark
import task_env


def _flat_load_params(base: float) -> dict:
    return {
        'A': 0.0,
        'B': 0.0,
        'C': 0.0,
        'D': 0.0,
        'f1': 0.01,
        'f2': 0.01,
        'f3': 0.01,
        'tau': 100.0,
        'base': base,
        'noise': 0.0,
    }


@unittest.skipUnless(
    task_env.benchmark_e2e_enabled(),
    'set TRT15_BENCH=1 (or INFERENCE_BENCHMARK_E2E=1) and start trt_server',
)
class TestBenchmarkServerE2E(unittest.TestCase):
    def test_server_handles_load_without_errors_and_latency_sla(self):
        endpoint = task_env.e2e_settings()
        bench = task_env.benchmark_settings()

        png = _TASK_ROOT / 'image.png'
        image_paths = [str(png)] if png.is_file() else None

        out = client_benchmark.run_benchmark(
            endpoint.host,
            endpoint.port,
            bench.duration_s,
            bench.threads,
            image_paths,
            _flat_load_params(bench.load_base),
            quiet=True,
        )

        print(
            "\n--- benchmark results ---\n"
            f"  endpoint:         {endpoint.host}:{endpoint.port}\n"
            f"  duration_s:        {bench.duration_s}\n"
            f"  threads:           {bench.threads}\n"
            f"  load_base (RPS):   {bench.load_base}\n"
            f"  success / errors:  {out['success']} / {out['errors']}\n"
            f"  avg_latency_ms:    {out['avg_latency_s'] * 1000.0:.4f} "
            f"(limit < {bench.max_avg_latency_s * 1000.0:.4f})\n"
            f"  max_latency_ms:    {out['max_latency_s'] * 1000.0:.4f} "
            f"(limit < {bench.max_max_latency_s * 1000.0:.4f})\n"
            "---",
            flush=True,
        )

        self.assertEqual(
            out['errors'],
            0,
            msg=f'server returned errors or timeouts: {out!r}',
        )
        self.assertGreater(
            out['success'],
            0,
            msg='no successful requests — server unreachable or failing immediately',
        )
        self.assertLess(
            out['avg_latency_s'],
            bench.max_avg_latency_s,
            msg=(
                f"avg latency {out['avg_latency_s']*1000:.1f} ms >= limit "
                f"{bench.max_avg_latency_s*1000:.1f} ms"
            ),
        )
        self.assertLess(
            out['max_latency_s'],
            bench.max_max_latency_s,
            msg=(
                f"max latency {out['max_latency_s']*1000:.1f} ms >= limit "
                f"{bench.max_max_latency_s*1000:.1f} ms"
            ),
        )


if __name__ == '__main__':
    unittest.main()
