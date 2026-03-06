#include "gemm.cuh"
#include <cuda_helpers.h>

#include <catch2/benchmark/catch_benchmark.hpp>
#include <catch2/catch_test_macros.hpp>

#include <iostream>
#include <random>
#include <vector>

template <typename GemmFn>
static float run_benchmark(int M, int K, int N, GemmFn gemm_fn) {
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);

    size_t size_A = M * K;
    size_t size_B = K * N;
    size_t size_C = M * N;

    std::vector<float> h_A(size_A);
    std::vector<float> h_B(size_B);
    std::vector<float> h_C(size_C);
    for (size_t i = 0; i < size_A; ++i) {
        h_A[i] = dist(gen);
    }
    for (size_t i = 0; i < size_B; ++i) {
        h_B[i] = dist(gen);
    }
    for (size_t i = 0; i < size_C; ++i) {
        h_C[i] = dist(gen);
    }

    size_t bytes_A = size_A * sizeof(float);
    size_t bytes_B = size_B * sizeof(float);
    size_t bytes_C = size_C * sizeof(float);

    float *d_A = nullptr;
    float *d_B = nullptr;
    float *d_C = nullptr;
    CheckStatus(cudaMalloc(&d_A, bytes_A));
    CheckStatus(cudaMalloc(&d_B, bytes_B));
    CheckStatus(cudaMalloc(&d_C, bytes_C));
    CheckStatus(cudaMemcpy(d_A, h_A.data(), bytes_A, cudaMemcpyHostToDevice));
    CheckStatus(cudaMemcpy(d_B, h_B.data(), bytes_B, cudaMemcpyHostToDevice));
    CheckStatus(cudaMemcpy(d_C, h_C.data(), bytes_C, cudaMemcpyHostToDevice));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    gemm_fn(d_A, d_B, d_C);
    CheckStatus(cudaDeviceSynchronize());

    cudaEventRecord(start);
    gemm_fn(d_A, d_B, d_C);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);

    double gflops =
        2.0 * (double)M * (double)K * (double)N / (1e6 * (double)ms);
    INFO("GEMM " << M << "x" << K << "x" << N << ": " << ms << " ms, " << gflops
                 << " GFLOPS");

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    return ms;
}

TEST_CASE("Benchmark on 4096x4096x4096") {
    int N = 4096;
    auto fn = [N](float *d_A, float *d_B, float *d_C) {
        gemm_cuda(1.0f, d_A, d_B, 0.0f, d_C, N, N, N);
    };
    float ms = run_benchmark(N, N, N, fn);
    REQUIRE(ms < 110);
}
