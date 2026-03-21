#include "attention.cuh"
#include <cuda_helpers.h>

#include <catch2/benchmark/catch_benchmark.hpp>
#include <catch2/catch_test_macros.hpp>

#include <iostream>
#include <random>
#include <vector>

static float run_benchmark(int N, int d) {
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    std::vector<float> h_Q(N * d);
    std::vector<float> h_K(d * N);
    std::vector<float> h_V(N * d);

    for (int i = 0; i < N * d; ++i) {
        h_Q[i] = dist(gen);
        h_V[i] = dist(gen);
    }
    for (int i = 0; i < d * N; ++i) {
        h_K[i] = dist(gen);
    }

    float *d_Q, *d_K, *d_V, *d_O;
    CheckStatus(cudaMalloc(&d_Q, N * d * sizeof(float)));
    CheckStatus(cudaMalloc(&d_K, d * N * sizeof(float)));
    CheckStatus(cudaMalloc(&d_V, N * d * sizeof(float)));
    CheckStatus(cudaMalloc(&d_O, N * d * sizeof(float)));
    CheckStatus(cudaMemcpy(d_Q, h_Q.data(), N * d * sizeof(float), cudaMemcpyHostToDevice));
    CheckStatus(cudaMemcpy(d_K, h_K.data(), d * N * sizeof(float), cudaMemcpyHostToDevice));
    CheckStatus(cudaMemcpy(d_V, h_V.data(), N * d * sizeof(float), cudaMemcpyHostToDevice));

    flash_attention(d_Q, d_K, d_V, d_O, N, d);
    CheckStatus(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    flash_attention(d_Q, d_K, d_V, d_O, N, d);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);

    double flops = 4.0 * N * N * d + 5.0 * N * N;
    double gflops = flops / (1e6 * ms);
    INFO("Flash attention N=" << N << " d=" << d << ": " << ms << " ms, " << gflops << " GFLOPS");

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_Q);
    cudaFree(d_K);
    cudaFree(d_V);
    cudaFree(d_O);

    return ms;
}

TEST_CASE("Benchmark: flash_attention 4096x32") {
    float ms = run_benchmark(4096, 32);
    REQUIRE(ms < 8.0f);
}

TEST_CASE("Benchmark: flash_attention 8192x32") {
    float ms = run_benchmark(8192, 32);
    REQUIRE(ms < 30.0f);
}

TEST_CASE("Benchmark: flash_attention 16384x32") {
    float ms = run_benchmark(16384, 32);
    REQUIRE(ms < 100.0f);
}
