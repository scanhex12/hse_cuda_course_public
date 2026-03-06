#include "prefsum.cuh"

#include <catch2/catch_test_macros.hpp>

#include <chrono>
#include <climits>
#include <cstdio>
#include <iostream>
#include <random>
#include <vector>

static void select_least_loaded_gpu() {
    int ndev = 0;
    if (cudaGetDeviceCount(&ndev) != cudaSuccess || ndev <= 0) {
        return;
    }
    size_t best_free = 0;
    int best = 0;
    for (int i = 0; i < ndev; ++i) {
        if (cudaSetDevice(i) != cudaSuccess) {
            continue;
        }
        size_t free_mem = 0;
        size_t total = 0;
        if (cudaMemGetInfo(&free_mem, &total) == cudaSuccess &&
            free_mem > best_free) {
            best_free = free_mem;
            best = i;
        }
    }
    cudaSetDevice(best);
}

static double run_benchmark_cpu_prefsum(long long n) {
    if (n <= 0 || n > static_cast<long long>(INT_MAX)) {
        return -1.0;
    }
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(-10.0f, 10.0f);

    size_t n_size = n;
    std::vector<float> in(n_size);
    std::vector<float> out(n_size);
    for (size_t i = 0; i < n_size; ++i) {
        in[i] = dist(gen);
    }

    auto t0 = std::chrono::steady_clock::now();
    float acc = 0.0f;
    for (size_t i = 0; i < n_size; ++i) {
        out[i] = acc;
        acc += in[i];
    }
    auto t1 = std::chrono::steady_clock::now();
    (void)out[n_size - 1];
    return 1e-3 * (double)std::chrono::duration_cast<std::chrono::microseconds>(
                      t1 - t0)
                      .count();
}

static float run_benchmark_gpu_prefsum(long long n, bool silent = false) {
    if (n <= 0 || n > static_cast<long long>(INT_MAX)) {
        return -1.0f;
    }
    int n_int = static_cast<int>(n);

    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(-10.0f, 10.0f);

    size_t n_size = n;
    std::vector<float> h_in(n_size);
    for (size_t i = 0; i < n_size; ++i) {
        h_in[i] = dist(gen);
    }

    float *d_in = nullptr;
    float *d_out = nullptr;
    size_t bytes = n_size * sizeof(float);

    cudaError_t err = cudaMalloc(&d_in, bytes);
    if (err != cudaSuccess) {
        if (!silent) {
            std::fprintf(stderr, "GPU OOM d_in n=%lld: %s\n",
                         static_cast<long long>(n), cudaGetErrorString(err));
        }
        return -1.0f;
    }
    err = cudaMalloc(&d_out, bytes);
    if (err != cudaSuccess) {
        if (!silent) {
            std::fprintf(stderr, "GPU OOM d_out n=%lld: %s\n",
                         static_cast<long long>(n), cudaGetErrorString(err));
        }
        cudaFree(d_in);
        return -1.0f;
    }
    err = cudaMemcpy(d_in, h_in.data(), bytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        if (!silent) {
            std::fprintf(stderr, "cudaMemcpy H2D failed: %s\n",
                         cudaGetErrorString(err));
        }
        cudaFree(d_out);
        cudaFree(d_in);
        return -1.0f;
    }

    exclusive_scan_cuda(d_out, d_in, n_int);
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        if (!silent) {
            std::fprintf(stderr, "GPU scan failed: %s\n",
                         cudaGetErrorString(err));
        }
        cudaFree(d_out);
        cudaFree(d_in);
        return -1.0f;
    }

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    exclusive_scan_cuda(d_out, d_in, n_int);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);

    cudaEventDestroy(stop);
    cudaEventDestroy(start);
    cudaFree(d_out);
    cudaFree(d_in);
    return ms;
}

TEST_CASE("Benchmark: Prefsum 1M elements") {
    select_least_loaded_gpu();
    int dev = 0;
    size_t free_mem = 0;
    if (cudaGetDevice(&dev) == cudaSuccess &&
        cudaMemGetInfo(&free_mem, nullptr) == cudaSuccess) {
        std::fprintf(stderr, "Prefsum benchmark using GPU %d (%zu MiB free)\n",
                     dev, free_mem / (1024 * 1024));
    }

    const long long n = 1ll << 20;
    float gpu_ms = run_benchmark_gpu_prefsum(n);
    if (gpu_ms < 0.0f) {
        SKIP("GPU out of memory");
    }

    double cpu_ms = run_benchmark_cpu_prefsum(n);
    std::cerr << "Prefsum n=" << n << ": GPU " << gpu_ms << " ms, CPU "
              << cpu_ms << " ms"
              << (gpu_ms < (float)cpu_ms ? " [GPU faster]" : " [CPU faster]")
              << std::endl;
    REQUIRE(gpu_ms < 0.5);
}

TEST_CASE("Benchmark: Prefsum 4M elements") {
    const long long n = 4ll << 20;
    float gpu_ms = run_benchmark_gpu_prefsum(n);
    if (gpu_ms < 0.0f) {
        SKIP("GPU out of memory");
    }

    double cpu_ms = run_benchmark_cpu_prefsum(n);
    std::cerr << "Prefsum n=" << n << ": GPU " << gpu_ms << " ms, CPU "
              << cpu_ms << " ms"
              << (gpu_ms < (float)cpu_ms ? " [GPU faster]" : " [CPU faster]")
              << std::endl;
    REQUIRE(gpu_ms < 1.9f);
}

TEST_CASE("Benchmark: Prefsum 16M elements") {
    const long long n = 16ll << 20;
    float gpu_ms = run_benchmark_gpu_prefsum(n);
    if (gpu_ms < 0.0f) {
        SKIP("GPU out of memory");
    }

    double cpu_ms = run_benchmark_cpu_prefsum(n);
    std::cerr << "Prefsum n=" << n << ": GPU " << gpu_ms << " ms, CPU "
              << cpu_ms << " ms"
              << (gpu_ms < (float)cpu_ms ? " [GPU faster]" : " [CPU faster]")
              << std::endl;
    REQUIRE(gpu_ms < 4.0f);
}

TEST_CASE("Benchmark: Prefsum 64M elements") {
    const long long n = 64ll << 20;
    float gpu_ms = run_benchmark_gpu_prefsum(n);
    if (gpu_ms < 0.0f) {
        SKIP("GPU out of memory for 64M");
    }

    double cpu_ms = run_benchmark_cpu_prefsum(n);
    std::cerr << "Prefsum n=" << n << ": GPU " << gpu_ms << " ms, CPU "
              << cpu_ms << " ms"
              << (gpu_ms < (float)cpu_ms ? " [GPU faster]" : " [CPU faster]")
              << std::endl;
    REQUIRE(gpu_ms < 11.0f);
}

TEST_CASE("Benchmark: Prefsum 128M elements") {
    const long long n = 128ll << 20;
    float gpu_ms = run_benchmark_gpu_prefsum(n);
    if (gpu_ms < 0.0f) {
        SKIP("GPU out of memory for 128M (~1 GB)");
    }

    double cpu_ms = run_benchmark_cpu_prefsum(n);
    std::cerr << "Prefsum n=" << n << ": GPU " << gpu_ms << " ms, CPU "
              << cpu_ms << " ms"
              << (gpu_ms < (float)cpu_ms ? " [GPU faster]" : " [CPU faster]")
              << std::endl;
    REQUIRE(gpu_ms < 15.0f);
}

TEST_CASE("Benchmark: Prefsum 256M elements") {
    const long long n = 256ll << 20;
    float gpu_ms = run_benchmark_gpu_prefsum(n);
    if (gpu_ms < 0.0f) {
        SKIP("GPU out of memory for 256M (~2 GB)");
    }

    double cpu_ms = run_benchmark_cpu_prefsum(n);
    std::cerr << "Prefsum n=" << n << ": GPU " << gpu_ms << " ms, CPU "
              << cpu_ms << " ms"
              << (gpu_ms < (float)cpu_ms ? " [GPU faster]" : " [CPU faster]")
              << std::endl;
    REQUIRE(gpu_ms < 20.0f);
}
