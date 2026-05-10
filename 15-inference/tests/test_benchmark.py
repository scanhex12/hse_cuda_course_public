import os
import sys
import unittest
from pathlib import Path

_TASK_ROOT = Path(__file__).resolve().parents[1]
if str(_TASK_ROOT) not in sys.path:
    sys.path.insert(0, str(_TASK_ROOT))

import client_benchmark

_AUTHOR_REF_AVG_LATENCY_S = 0.0095
_AUTHOR_REF_MAX_LATENCY_S = 0.0185


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
    os.environ.get('INFERENCE_BENCHMARK_E2E'),
    'set INFERENCE_BENCHMARK_E2E=1 and start trt_server (same host/port as INFERENCE_E2E_*)',
)
class TestBenchmarkServerE2E(unittest.TestCase):
    def test_server_handles_load_without_errors_and_latency_sla(self):
        host = os.environ.get('INFERENCE_E2E_HOST', 'localhost')
        port = int(os.environ.get('INFERENCE_E2E_PORT', '50051'))
        duration = float(os.environ.get('INFERENCE_BENCHMARK_DURATION_S', '2.5'))
        num_threads = int(os.environ.get('INFERENCE_BENCHMARK_THREADS', '3'))
        base = float(os.environ.get('INFERENCE_BENCHMARK_LOAD_BASE', '12.0'))
        max_avg_s = float(
            os.environ.get(
                'INFERENCE_BENCHMARK_MAX_AVG_LATENCY_S',
                str(_AUTHOR_REF_AVG_LATENCY_S * 2.0),
            )
        )
        max_worst_s = float(
            os.environ.get(
                'INFERENCE_BENCHMARK_MAX_MAX_LATENCY_S',
                str(_AUTHOR_REF_MAX_LATENCY_S * 2.0),
            )
        )

        png = _TASK_ROOT / 'image.png'
        image_paths = [str(png)] if png.is_file() else None

        out = client_benchmark.run_benchmark(
            host,
            port,
            duration,
            num_threads,
            image_paths,
            _flat_load_params(base),
            quiet=True,
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
            max_avg_s,
            msg=f"avg latency {out['avg_latency_s']*1000:.1f} ms >= limit {max_avg_s*1000:.1f} ms",
        )
        self.assertLess(
            out['max_latency_s'],
            max_worst_s,
            msg=f"max latency {out['max_latency_s']*1000:.1f} ms >= limit {max_worst_s*1000:.1f} ms",
        )


if __name__ == '__main__':
    unittest.main()
