#include "lu_decomposition.cuh"

#include <cuda_runtime.h>

#include <catch2/benchmark/catch_benchmark.hpp>
#include <catch2/catch_test_macros.hpp>

#include <iostream>
#include <random>
#include <vector>

static std::vector<float> make_random_matrix(int n, float lo = -1.0f,
                                             float hi = 1.0f,
                                             uint32_t seed = 42) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(lo, hi);
    std::vector<float> v(n * n);
    for (int i = 0; i < n * n; i++)
        v[i] = dist(rng);
    for (int i = 0; i < n; i++) {
        float sum = 0.0f;
        for (int j = 0; j < n; j++)
            if (i != j)
                sum += std::abs(v[i * n + j]);
        v[i * n + i] = sum + 1.0f;
    }
    return v;
}

static float run_benchmark(int n) {
    auto h_A = make_random_matrix(n);

    float *d_A = nullptr, *d_L = nullptr, *d_U = nullptr;
    if (cudaMalloc(&d_A, n * n * sizeof(float)) != cudaSuccess ||
        cudaMalloc(&d_L, n * n * sizeof(float)) != cudaSuccess ||
        cudaMalloc(&d_U, n * n * sizeof(float)) != cudaSuccess) {
        if (d_A)
            cudaFree(d_A);
        if (d_L)
            cudaFree(d_L);
        if (d_U)
            cudaFree(d_U);
        return -1.0f;
    }

    cudaMemcpy(d_A, h_A.data(), n * n * sizeof(float), cudaMemcpyHostToDevice);

    // Warmup
    constexpr int kWarmupIters = 15;
    for (int w = 0; w < kWarmupIters; ++w) {
        lu_decomposition(d_A, d_L, d_U, n);
    }
    cudaDeviceSynchronize();
    cudaMemcpy(d_A, h_A.data(), n * n * sizeof(float), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    lu_decomposition(d_A, d_L, d_U, n);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);
    std::cerr << "LU " << n << "x" << n << ": " << ms << " ms, " << '\n';

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_A);
    cudaFree(d_L);
    cudaFree(d_U);

    return ms;
}

TEST_CASE("Benchmark: LU 64x64") {
    float ms = run_benchmark(64);
    REQUIRE(ms < 0.35f);
}

TEST_CASE("Benchmark: LU 128x128") {
    float ms = run_benchmark(128);
    REQUIRE(ms < 1.1f);
}

TEST_CASE("Benchmark: LU 768x768") {
    float ms = run_benchmark(768);
    REQUIRE(ms < 17.0f);
}

TEST_CASE("Benchmark: LU 4096x4096") {
    float ms = run_benchmark(4096);
    REQUIRE(ms < 520.0f);
}
