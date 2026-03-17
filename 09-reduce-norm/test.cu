#include <cmath>
#include <iostream>
#include <vector>

#include "norm.cuh"

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <catch2/matchers/catch_matchers_templated.hpp>
#include <catch2/matchers/catch_matchers.hpp>

using Catch::Matchers::WithinAbs;

void rmsnorm_cpu(const float* input, float gamma, float beta, float* output,
                 int N, float eps) {
    double sum = 0.0;
    for (int i = 0; i < N; i++)
        sum += input[i] * input[i];

    float rms = 1.0f / sqrtf(sum / N + eps);

    for (int i = 0; i < N; i++)
        output[i] = gamma * input[i] * rms + beta;
}

TEST_CASE("Simple") {
    const int N = 128;
    std::vector<float> input(N), out_cpu(N), out_gpu(N);

    for (int i = 0; i < N; i++)
        input[i] = i * 0.01f;

    float gamma = 1.5f, beta = 0.3f, eps = 1e-6f;

    rmsnorm_cpu(input.data(), gamma, beta, out_cpu.data(), N, eps);

    float *d_in, *d_out;
    cudaMalloc(&d_in, N * sizeof(float));
    cudaMalloc(&d_out, N * sizeof(float));
    cudaMemcpy(d_in, input.data(), N * sizeof(float), cudaMemcpyHostToDevice);

    solve(d_in, gamma, beta, d_out, N, eps);

    cudaMemcpy(out_gpu.data(), d_out, N * sizeof(float),
               cudaMemcpyDeviceToHost);

    for (int i = 0; i < N; i++)
        REQUIRE(fabs(out_cpu[i] - out_gpu[i]) < 1e-4);

    cudaFree(d_in);
    cudaFree(d_out);
}

TEST_CASE("Large") {
    const int N = 1 << 20;
    std::vector<float> input(N), out_cpu(N), out_gpu(N);

    for (int i = 0; i < N; i++)
        input[i] = sinf(i * 0.001f);

    float gamma = 0.9f, beta = -0.2f, eps = 1e-5f;

    rmsnorm_cpu(input.data(), gamma, beta, out_cpu.data(), N, eps);

    float *d_in, *d_out;
    cudaMalloc(&d_in, N * sizeof(float));
    cudaMalloc(&d_out, N * sizeof(float));
    cudaMemcpy(d_in, input.data(), N * sizeof(float), cudaMemcpyHostToDevice);

    solve(d_in, gamma, beta, d_out, N, eps);

    cudaMemcpy(out_gpu.data(), d_out, N * sizeof(float),
               cudaMemcpyDeviceToHost);

    for (int i = 0; i < N; i++)
        REQUIRE(fabs(out_cpu[i] - out_gpu[i]) < 1e-3);

    cudaFree(d_in);
    cudaFree(d_out);
}

TEST_CASE("Zeros") {
    const int N = 256;
    std::vector<float> input(N, 0.0f), out_gpu(N);

    float *d_in, *d_out;
    cudaMalloc(&d_in, N * sizeof(float));
    cudaMalloc(&d_out, N * sizeof(float));
    cudaMemcpy(d_in, input.data(), N * sizeof(float), cudaMemcpyHostToDevice);

    solve(d_in, 1.0f, 0.0f, d_out, N, 1e-6f);

    cudaMemcpy(out_gpu.data(), d_out, N * sizeof(float),
               cudaMemcpyDeviceToHost);

    for (int i = 0; i < N; i++)
        REQUIRE(fabs(out_gpu[i]) < 1e-3);

    cudaFree(d_in);
    cudaFree(d_out);
}
