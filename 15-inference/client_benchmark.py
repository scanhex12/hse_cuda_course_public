#!/usr/bin/env python3

import math
import random
import sys
import threading
import time
from collections import deque
from pathlib import Path

import grpc
import numpy as np
from PIL import Image

import inference_pb2
import inference_pb2_grpc

_GRPC_CHANNEL_OPTIONS = (
    ("grpc.keepalive_time_ms", 30_000),
    ("grpc.keepalive_timeout_ms", 10_000),
)


def generate_random_image():
    arr = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
    img = Image.fromarray(arr)
    return img

def load_image_or_generate(image_path):
    if image_path and Path(image_path).exists():
        img = Image.open(image_path).convert("RGB")
        img = img.resize((224, 224))
        return img
    return generate_random_image()

def send_request(host, port, image_data):
    target = f"{host}:{port}"
    channel = grpc.insecure_channel(target, options=list(_GRPC_CHANNEL_OPTIONS))
    try:
        stub = inference_pb2_grpc.InferenceServiceStub(channel)
        return classify_once(stub, image_data)
    finally:
        channel.close()


def classify_once(stub, image_data, timeout_s=45.0):
    req = inference_pb2.InferRequest()
    req.width = 224
    req.height = 224
    req.rgb_image = image_data
    transient = {
        grpc.StatusCode.UNAVAILABLE,
        grpc.StatusCode.RESOURCE_EXHAUSTED,
        grpc.StatusCode.ABORTED,
        grpc.StatusCode.DEADLINE_EXCEEDED,
    }
    for attempt in range(3):
        try:
            resp = stub.Classify(req, timeout=timeout_s)
            if len(resp.top5) != 5:
                return None
            indices = tuple(p.class_index for p in resp.top5)
            scores = tuple(p.score for p in resp.top5)
            return (indices, scores)
        except grpc.RpcError as e:
            if e.code() not in transient or attempt + 1 >= 3:
                return None
            time.sleep(0.03 * (attempt + 1))
        except Exception:
            return None
    return None

def load_function(t, params):
    A = params.get('A', 5.0)
    B = params.get('B', 3.0)
    C = params.get('C', 2.0)
    D = params.get('D', 1.5)
    f1 = params.get('f1', 0.1)
    f2 = params.get('f2', 0.15)
    f3 = params.get('f3', 0.05)
    tau = params.get('tau', 20.0)
    base = params.get('base', 2.0)
    noise_amplitude = params.get('noise', 0.3)
    
    term1 = A * math.sin(2 * math.pi * f1 * t) * math.cos(2 * math.pi * f2 * t)
    term2 = B * math.exp(-t / tau) * math.sin(2 * math.pi * f3 * t)
    term3 = C * math.sin(2 * math.pi * (f1 + f2) * t) * math.cos(2 * math.pi * f3 * t)
    term4 = D * math.sin(2 * math.pi * f1 * t) ** 2 * math.cos(2 * math.pi * f2 * t)
    noise = noise_amplitude * (random.random() - 0.5) * 2
    
    load = base + term1 + term2 + term3 + term4 + noise
    return max(0.1, load)

def worker_thread(host, port, images, stats, stop_event, load_func, load_params):
    target = f"{host}:{port}"
    channel = grpc.insecure_channel(target, options=list(_GRPC_CHANNEL_OPTIONS))
    stub = inference_pb2_grpc.InferenceServiceStub(channel)
    request_times = deque(maxlen=1000)

    try:
        while not stop_event.is_set():
            t = time.time()
            load = load_func(t, load_params)

            delay = 1.0 / load if load > 0 else 1.0
            time.sleep(delay)

            if stop_event.is_set():
                break

            img = random.choice(images)
            arr = np.array(img, dtype=np.uint8)
            image_data = arr.tobytes()

            start_time = time.time()
            result = classify_once(stub, image_data)
            elapsed = time.time() - start_time

            with stats['lock']:
                if result is not None:
                    stats['success'] += 1
                    request_times.append(elapsed)
                    stats['total_time'] += elapsed
                    stats['max_latency'] = max(stats['max_latency'], elapsed)
                    stats['avg_latency'] = (
                        stats['total_time'] / stats['success']
                        if stats['success'] > 0
                        else 0.0
                    )
                    stats['min_latency'] = min(stats['min_latency'], elapsed)
                else:
                    stats['errors'] += 1
    finally:
        channel.close()


def run_benchmark(
    host,
    port,
    duration,
    num_threads,
    image_paths,
    load_params,
    *,
    quiet=False,
):
    """Запуск бенчмарка; возвращает агрегаты для тестов (quiet=True без вывода в stdout)."""
    if not quiet:
        print("Starting benchmark:")
        print(f"  Host: {host}")
        print(f"  Port: {port}")
        print(f"  Duration: {duration}s")
        print(f"  Threads: {num_threads}")
        print(f"  Load function parameters: {load_params}")
        print()

    images = []
    if image_paths:
        for path in image_paths:
            img = load_image_or_generate(path)
            images.append(img)
    else:
        for _ in range(10):
            images.append(generate_random_image())

    if not images:
        images = [generate_random_image()]

    stats = {
        'success': 0,
        'errors': 0,
        'total_time': 0.0,
        'avg_latency': 0.0,
        'max_latency': 0.0,
        'min_latency': float('inf'),
        'lock': threading.Lock(),
    }

    stop_event = threading.Event()
    threads = []

    start_time = time.time()
    base_time = start_time

    def load_func(t, params):
        elapsed = t - base_time
        return load_function(elapsed, params)

    for _ in range(num_threads):
        t = threading.Thread(
            target=worker_thread,
            args=(host, port, images, stats, stop_event, load_func, load_params),
        )
        t.daemon = True
        t.start()
        threads.append(t)

    last_report = start_time
    report_interval = 2.0

    try:
        while time.time() - start_time < duration:
            time.sleep(0.1)

            if quiet:
                continue

            current_time = time.time()
            if current_time - last_report >= report_interval:
                elapsed = current_time - start_time
                current_load = load_function(elapsed, load_params)

                with stats['lock']:
                    success = stats['success']
                    errors = stats['errors']
                    rate = success / elapsed if elapsed > 0 else 0
                    avg_lat = stats['avg_latency']
                    max_lat = stats['max_latency']
                    min_lat = (
                        stats['min_latency']
                        if stats['min_latency'] != float('inf')
                        else 0.0
                    )

                print(
                    f"[{elapsed:.1f}s] Load: {current_load:.2f} req/s | "
                    f"Success: {success} | Errors: {errors} | "
                    f"Rate: {rate:.2f} req/s | "
                    f"Latency: avg={avg_lat*1000:.1f}ms, min={min_lat*1000:.1f}ms, max={max_lat*1000:.1f}ms"
                )
                last_report = current_time
    except KeyboardInterrupt:
        if not quiet:
            print("\nInterrupted by user")

    stop_event.set()
    for t in threads:
        t.join(timeout=8.0)

    final_time = time.time() - start_time

    with stats['lock']:
        success = stats['success']
        errors = stats['errors']
        total = success + errors
        rate = success / final_time if final_time > 0 else 0
        avg_lat = stats['avg_latency']
        max_lat = stats['max_latency']
        min_lat = stats['min_latency'] if stats['min_latency'] != float('inf') else 0.0
        total_time = stats['total_time']

    if not quiet:
        print("\n" + "=" * 60)
        print("Final Results:")
        print("=" * 60)
        print(f"Duration: {final_time:.2f}s")
        print(f"Total requests: {total}")
        print(f"Successful: {success}")
        print(f"Errors: {errors}")
        print(f"Success rate: {success / total * 100:.2f}%" if total > 0 else "N/A")
        print(f"Average rate: {rate:.2f} req/s")
        print("Latency statistics:")
        print(f"  Average: {avg_lat * 1000:.2f} ms")
        print(f"  Min: {min_lat * 1000:.2f} ms")
        print(f"  Max: {max_lat * 1000:.2f} ms")
        print("=" * 60)

    avg_latency_s = (total_time / success) if success > 0 else 0.0
    max_latency_s = max_lat
    return {
        'errors': errors,
        'success': success,
        'avg_latency_s': avg_latency_s,
        'max_latency_s': max_latency_s,
    }


def benchmark(host, port, duration, num_threads, image_paths, load_params):
    run_benchmark(
        host,
        port,
        duration,
        num_threads,
        image_paths,
        load_params,
        quiet=False,
    )

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python client_benchmark.py <duration> [options]")
        print("Options:")
        print("  --host HOST          Server host (default: localhost)")
        print("  --port PORT          gRPC server port (default: 50051)")
        print("  --threads N          Number of threads (default: 4)")
        print("  --images PATH ...    Image paths (default: generate random)")
        print("  --load-params A,B,C,D,f1,f2,f3,tau,base,noise")
        print("                      Custom load function parameters")
        sys.exit(1)
    
    duration = float(sys.argv[1])
    host = "localhost"
    port = 50051
    num_threads = 4
    image_paths = []
    load_params = {}
    
    i = 2
    while i < len(sys.argv):
        if sys.argv[i] == "--host" and i + 1 < len(sys.argv):
            host = sys.argv[i + 1]
            i += 2
        elif sys.argv[i] == "--port" and i + 1 < len(sys.argv):
            port = int(sys.argv[i + 1])
            i += 2
        elif sys.argv[i] == "--threads" and i + 1 < len(sys.argv):
            num_threads = int(sys.argv[i + 1])
            i += 2
        elif sys.argv[i] == "--images":
            i += 1
            while i < len(sys.argv) and not sys.argv[i].startswith("--"):
                image_paths.append(sys.argv[i])
                i += 1
        elif sys.argv[i] == "--load-params" and i + 1 < len(sys.argv):
            params_str = sys.argv[i + 1]
            params = [float(x) for x in params_str.split(",")]
            if len(params) >= 10:
                load_params = {
                    'A': params[0], 'B': params[1], 'C': params[2], 'D': params[3],
                    'f1': params[4], 'f2': params[5], 'f3': params[6],
                    'tau': params[7], 'base': params[8], 'noise': params[9]
                }
            i += 2
        else:
            i += 1
    
    if not load_params:
        load_params = {
            'A': 5.0,
            'B': 3.0,
            'C': 2.0,
            'D': 1.5,
            'f1': 0.1,
            'f2': 0.15,
            'f3': 0.05,
            'tau': 20.0,
            'base': 2.0,
            'noise': 0.3
        }
    
    if not image_paths:
        image_paths = ["image.jpg"] if Path("image.jpg").exists() else None
    
    benchmark(host, port, duration, num_threads, image_paths, load_params)
