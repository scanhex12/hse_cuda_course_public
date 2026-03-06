#include <cmath>
#include <iostream>
#include <vector>

#include "transpose.cuh"
#include <cuda_helpers.h>

#include <catch2/catch_test_macros.hpp>

static void transpose_cpu(const float *in, float *out, int rows, int cols) {
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            out[j * rows + i] = in[i * cols + j];
        }
    }
}

static bool run_test(size_t rows, size_t cols) {
    std::vector<float> h_in(rows * cols);
    for (size_t i = 0; i < rows * cols; ++i)
        h_in[i] = (i % 100 - 50);

    std::vector<float> ref(rows * cols, 0.0f);
    transpose_cpu(h_in.data(), ref.data(), rows, cols);

    float *d_in = nullptr;
    float *d_out = nullptr;
    CheckStatus(cudaMalloc(&d_in, rows * cols * sizeof(float)));
    CheckStatus(cudaMalloc(&d_out, rows * cols * sizeof(float)));
    CheckStatus(cudaMemcpy(d_in, h_in.data(), rows * cols * sizeof(float),
                           cudaMemcpyHostToDevice));

    transpose_cuda(d_in, d_out, rows, cols);
    CheckStatus(cudaDeviceSynchronize());

    std::vector<float> gpu(rows * cols);
    CheckStatus(cudaMemcpy(gpu.data(), d_out, rows * cols * sizeof(float),
                           cudaMemcpyDeviceToHost));
    CheckStatus(cudaFree(d_in));
    CheckStatus(cudaFree(d_out));

    for (size_t i = 0; i < rows * cols; ++i) {
        if (std::fabs(gpu[i] - ref[i]) > 1e-5f) {
            std::cerr << "Mismatch at " << i << " got " << gpu[i] << " ref "
                      << ref[i] << "\n";
            return false;
        }
    }
    return true;
}

TEST_CASE("Matrix transpose") {
    REQUIRE(run_test(1, 1));
    REQUIRE(run_test(1, 100));
    REQUIRE(run_test(100, 1));
    REQUIRE(run_test(16, 16));
    REQUIRE(run_test(32, 32));
    REQUIRE(run_test(100, 50));
    REQUIRE(run_test(1024, 1024));
    REQUIRE(run_test(17, 31));
    REQUIRE(run_test(1000, 2000));
}
