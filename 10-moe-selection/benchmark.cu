#include "moe.cuh"
#include <cuda_helpers.h>

#include <catch2/benchmark/catch_benchmark.hpp>
#include <catch2/catch_test_macros.hpp>

#include <iostream>
#include <random>
#include <vector>

static bool run_benchmark(int M, int E, int K, float* out_ms) {
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(-2.0f, 2.0f);

    std::vector<float> h_X(M * E);
    for (int i = 0; i < M * E; ++i)
        h_X[i] = dist(gen);

    float *d_X = nullptr, *d_weights = nullptr;
    int *d_indices = nullptr;
    if (cudaMalloc(&d_X, M * E * sizeof(float)) != cudaSuccess ||
        cudaMalloc(&d_weights, M * K * sizeof(float)) != cudaSuccess ||
        cudaMalloc(&d_indices, M * K * sizeof(int)) != cudaSuccess) {
        if (d_X) cudaFree(d_X);
        if (d_weights) cudaFree(d_weights);
        if (d_indices) cudaFree(d_indices);
        return false;
    }
    cudaMemcpy(d_X, h_X.data(), M * E * sizeof(float), cudaMemcpyHostToDevice);

    choose_top_moe_experts(d_X, d_weights, d_indices, M, E, K);
    cudaDeviceSynchronize();

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    choose_top_moe_experts(d_X, d_weights, d_indices, M, E, K);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);

    INFO("MoE M=" << M << " E=" << E << " K=" << K << ": " << ms << " ms");

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_X);
    cudaFree(d_weights);
    cudaFree(d_indices);

    *out_ms = ms;
    return true;
}

TEST_CASE("Benchmark: MoE 1024x8 K=2") {
    float ms = 0.0f;
    if (run_benchmark(1024, 8, 2, &ms)) {
        REQUIRE(ms < 0.03f);
    }
}

TEST_CASE("Benchmark: MoE 2048x32 K=2") {
    float ms = 0.0f;
    if (run_benchmark(2048, 32, 2, &ms)) {
        REQUIRE(ms < 0.03f);
    }
}

TEST_CASE("Benchmark: MoE 4096x32 K=2") {
    float ms = 0.0f;
    if (run_benchmark(4096, 32, 2, &ms)) {
        REQUIRE(ms < 0.04f);
    }
}

TEST_CASE("Benchmark: MoE 1024x64 K=8") {
    float ms = 0.0f;
    if (run_benchmark(1024, 64, 8, &ms)) {
        REQUIRE(ms < 0.3f);
    }
}

TEST_CASE("Benchmark: MoE 2048x64 K=8") {
    float ms = 0.0f;
    if (run_benchmark(2048, 64, 8, &ms)) {
        REQUIRE(ms < 0.7f);
    }
}
