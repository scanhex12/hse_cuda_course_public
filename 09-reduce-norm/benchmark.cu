#include "norm.cuh"
#include <cuda_helpers.h>

#include <catch2/benchmark/catch_benchmark.hpp>
#include <catch2/catch_test_macros.hpp>

#include <iostream>
#include <random>
#include <vector>

static float run_benchmark(int N) {
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    std::vector<float> h_input(N);
    for (int i = 0; i < N; ++i)
        h_input[i] = dist(gen);

    float *d_in, *d_out;
    if (cudaMalloc(&d_in, N * sizeof(float)) != cudaSuccess ||
        cudaMalloc(&d_out, N * sizeof(float)) != cudaSuccess) {
        if (d_in)
            cudaFree(d_in);
        if (d_out)
            cudaFree(d_out);
        return -1.0f;
    }
    cudaMemcpy(d_in, h_input.data(), N * sizeof(float), cudaMemcpyHostToDevice);

    solve(d_in, 1.0f, 0.0f, d_out, N, 1e-6f);
    cudaDeviceSynchronize();

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    solve(d_in, 1.0f, 0.0f, d_out, N, 1e-6f);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);

    double bytes = 2.0 * N * sizeof(float);  // read + write
    double gbps = bytes / (1e6 * ms);
    INFO("RMSNorm N=" << N << ": " << ms << " ms, " << gbps << " GB/s");

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_in);
    cudaFree(d_out);

    return ms;
}

TEST_CASE("Benchmark: RMSNorm N=4096") {
    float ms = run_benchmark(4096);
    std::cout << "RMSNorm N=4096: " << ms << " ms" << std::endl;
    REQUIRE(ms < 0.02f);
}

TEST_CASE("Benchmark: RMSNorm N=65536") {
    float ms = run_benchmark(65536);
    std::cout << "RMSNorm N=65536: " << ms << " ms" << std::endl;
    REQUIRE(ms < 0.04f);
}

TEST_CASE("Benchmark: RMSNorm N=1048576") {
    float ms = run_benchmark(1048576);
    std::cout << "RMSNorm N=1048576: " << ms << " ms" << std::endl;
    REQUIRE(ms < 0.3f);
}
