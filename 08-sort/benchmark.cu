#include "sort.cuh"
#include <cuda_helpers.h>

#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <chrono>
#include <climits>
#include <iostream>
#include <random>
#include <stdexcept>
#include <vector>

inline void select_least_loaded_gpu() {
    int ndev = 0;
    if (cudaGetDeviceCount(&ndev) != cudaSuccess || ndev <= 0)
        return;
    size_t best_free = 0;
    int best = 0;
    for (int i = 0; i < ndev; ++i) {
        if (cudaSetDevice(i) != cudaSuccess)
            continue;
        size_t free_mem = 0, total = 0;
        if (cudaMemGetInfo(&free_mem, &total) == cudaSuccess &&
            free_mem > best_free) {
            best_free = free_mem;
            best = i;
        }
    }
    cudaSetDevice(best);
}

static void cpu_merge_sort(int *a, int *tmp, int n) {
    if (n <= 1)
        return;
    for (int width = 1; width < n; width *= 2) {
        for (int start = 0; start < n; start += 2 * width) {
            int mid = std::min(start + width, n);
            int end = std::min(start + 2 * width, n);
            if (mid >= end)
                continue;
            int i = start, j = mid, k = start;
            while (i < mid && j < end) {
                if (a[i] <= a[j])
                    tmp[k++] = a[i++];
                else
                    tmp[k++] = a[j++];
            }
            while (i < mid)
                tmp[k++] = a[i++];
            while (j < end)
                tmp[k++] = a[j++];
            for (int t = start; t < end; ++t)
                a[t] = tmp[t];
        }
    }
}

static double run_benchmark_std_sort(long long n) {
    std::mt19937 gen(42);
    std::uniform_int_distribution<int> dist(-10000, 10000);
    std::vector<int> v((size_t)n);
    for (long long i = 0; i < n; ++i)
        v[(size_t)i] = dist(gen);
    auto t0 = std::chrono::steady_clock::now();
    std::sort(v.begin(), v.end());
    auto t1 = std::chrono::steady_clock::now();
    return 1e-3 * (double)std::chrono::duration_cast<std::chrono::microseconds>(
                      t1 - t0)
                      .count();
}

static double run_benchmark_cpu_mergesort(long long n) {
    if (n <= 0 || n > (long long)INT_MAX)
        return -1.0;
    std::mt19937 gen(42);
    std::uniform_int_distribution<int> dist(-10000, 10000);
    std::vector<int> v((size_t)n);
    std::vector<int> tmp((size_t)n);
    for (long long i = 0; i < n; ++i)
        v[(size_t)i] = dist(gen);
    auto t0 = std::chrono::steady_clock::now();
    cpu_merge_sort(v.data(), tmp.data(), (int)n);
    auto t1 = std::chrono::steady_clock::now();
    return 1e-3 * (double)std::chrono::duration_cast<std::chrono::microseconds>(
                      t1 - t0)
                      .count();
}

static float run_benchmark_gpu(long long n, bool silent = false) {
    if (n <= 0 || n > (long long)INT_MAX)
        return -1.0f;
    int n_int = (int)n;
    std::mt19937 gen(42);
    std::uniform_int_distribution<int> dist(-10000, 10000);
    std::vector<int> h_in((size_t)n);
    for (long long i = 0; i < n; ++i)
        h_in[(size_t)i] = dist(gen);

    size_t bytes = (size_t)n * sizeof(int);
    int *d_data = nullptr;
    try {
        CheckStatus(cudaMalloc(&d_data, bytes));
        CheckStatus(cudaMemcpy(d_data, h_in.data(), bytes,
                               cudaMemcpyHostToDevice));
    } catch (const std::runtime_error &e) {
        if (!silent) {
            std::cerr << "GPU OOM n=" << n << ": " << e.what() << "\n";
        }
        return -1.0f;
    }

    device_sort_int(d_data, n_int);  // warmup
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    device_sort_int(d_data, n_int);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    CheckStatus(cudaFree(d_data));
    return ms;
}

TEST_CASE("Benchmark: Sort 1M elements") {
    select_least_loaded_gpu();
    int dev = 0;
    size_t free_mem = 0;
    if (cudaGetDevice(&dev) == cudaSuccess &&
        cudaMemGetInfo(&free_mem, nullptr) == cudaSuccess) {
        std::cerr << "Using GPU " << dev << " (" << (free_mem / (1024 * 1024))
                  << " MiB free)\n";
    }

    const long long n = 1048576ll;
    float gpu_ms = run_benchmark_gpu(n);
    REQUIRE(gpu_ms >= 0.0f);
    double cpu_merge_ms = run_benchmark_cpu_mergesort(n);
    double std_ms = run_benchmark_std_sort(n);
    std::cerr << "Benchmark n=" << n << ": GPU " << gpu_ms << " ms, CPU merge "
              << cpu_merge_ms << " ms, std::sort " << std_ms << " ms"
              << std::endl;
    REQUIRE(gpu_ms < 15.0f);
}

TEST_CASE("Benchmark: Sort 4M elements") {
    const long long n = 4194304ll;
    float gpu_ms = run_benchmark_gpu(n);
    REQUIRE(gpu_ms >= 0.0f);
    double cpu_merge_ms = run_benchmark_cpu_mergesort(n);
    double std_ms = run_benchmark_std_sort(n);
    std::cerr << "Benchmark n=" << n << ": GPU " << gpu_ms << " ms, CPU merge "
              << cpu_merge_ms << " ms, std::sort " << std_ms << " ms"
              << std::endl;
    REQUIRE(gpu_ms < 45.0f);
}