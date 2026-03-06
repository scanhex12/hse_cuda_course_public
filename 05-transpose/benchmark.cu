#include "transpose.cuh"

#include <catch2/benchmark/catch_benchmark.hpp>
#include <catch2/catch_test_macros.hpp>

#include <iostream>
#include <random>
#include <vector>

static float run_benchmark(int rows, int cols) {
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);

    size_t n = rows * cols;
    std::vector<float> h_in(n);
    for (size_t i = 0; i < n; ++i) {
        h_in[i] = dist(gen);
    }

    float *d_in = nullptr;
    float *d_out = nullptr;
    size_t bytes = n * sizeof(float);
    cudaMalloc(&d_in, bytes);
    cudaMalloc(&d_out, bytes);
    cudaMemcpy(d_in, h_in.data(), bytes, cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // warmup
    transpose_cuda(d_in, d_out, rows, cols);
    cudaDeviceSynchronize();

    cudaEventRecord(start);
    transpose_cuda(d_in, d_out, rows, cols);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);

    size_t bytes_rw = n * sizeof(float) * 2; // read + write
    double gb_s = (double)bytes_rw / (1e6 * (double)ms);
    INFO("Transpose " << rows << "x" << cols << ": " << ms << " ms, " << gb_s
                      << " GB/s");

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_in);
    cudaFree(d_out);
    return ms;
}

TEST_CASE("Benchmark: transpose 2048x2048") {
    float ms = run_benchmark(2048, 2048);
    std::cerr << "Benchmark: transpose 2048x2048: " << ms << " ms" << std::endl;
    REQUIRE(ms < 0.1f);
}

TEST_CASE("Benchmark: transpose 4096x4096") {
    float ms = run_benchmark(4096, 4096);
    std::cerr << "Benchmark: transpose 4096x4096: " << ms << " ms" << std::endl;
    REQUIRE(ms < 0.4f);
}

TEST_CASE("Benchmark: transpose 1024x4096 (rectangular)") {
    float ms = run_benchmark(1024, 4096);
    std::cerr << "Benchmark: transpose 1024x4096: " << ms << " ms" << std::endl;
    REQUIRE(ms < 0.1f);
}
