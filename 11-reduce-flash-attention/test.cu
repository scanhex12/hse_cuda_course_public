#include <cmath>
#include <vector>
#include <iostream>
#include <cstdlib>
#include <random>

#include "attention.cuh"

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

using Catch::Matchers::WithinAbs;

void flash_attention_cpu(
    const float* Q,
    const float* K,
    const float* V,
    float* O,
    int N,
    int d
) {
    std::vector<float> S(N * N);

    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < d; ++k) {
                sum += Q[i * d + k] * K[k * N + j];
            }
            S[i * N + j] = sum / std::sqrt(static_cast<float>(d));
        }
    }

    for (int i = 0; i < N; ++i) {
        float row_max = S[i * N];
        for (int j = 1; j < N; ++j) {
            row_max = std::max(row_max, S[i * N + j]);
        }
        float row_sum = 0.0f;
        for (int j = 0; j < N; ++j) {
            S[i * N + j] = std::exp(S[i * N + j] - row_max);
            row_sum += S[i * N + j];
        }
        for (int j = 0; j < N; ++j) {
            S[i * N + j] /= row_sum;
        }
    }

    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < d; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < N; ++k) {
                sum += S[i * N + k] * V[k * d + j];
            }
            O[i * d + j] = sum;
        }
    }
}

TEST_CASE("Flash attention: CPU vs GPU on random data")
{
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    const int N = 64;
    const int d = 32;

    std::vector<float> h_Q(N * d);
    std::vector<float> h_K(d * N);
    std::vector<float> h_V(N * d);

    for (int i = 0; i < N * d; ++i) {
        h_Q[i] = dist(rng);
        h_V[i] = dist(rng);
    }
    for (int i = 0; i < d * N; ++i) {
        h_K[i] = dist(rng);
    }

    std::vector<float> h_O_cpu(N * d);
    flash_attention_cpu(h_Q.data(), h_K.data(), h_V.data(), h_O_cpu.data(), N, d);

    float *d_Q, *d_K, *d_V, *d_O;
    cudaMalloc(&d_Q, N * d * sizeof(float));
    cudaMalloc(&d_K, d * N * sizeof(float));
    cudaMalloc(&d_V, N * d * sizeof(float));
    cudaMalloc(&d_O, N * d * sizeof(float));

    cudaMemcpy(d_Q, h_Q.data(), N * d * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_K, h_K.data(), d * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_V, h_V.data(), N * d * sizeof(float), cudaMemcpyHostToDevice);

    flash_attention(d_Q, d_K, d_V, d_O, N, d);
    cudaDeviceSynchronize();

    std::vector<float> h_O_gpu(N * d);
    cudaMemcpy(h_O_gpu.data(), d_O, N * d * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_Q);
    cudaFree(d_K);
    cudaFree(d_V);
    cudaFree(d_O);

    const float tol = 1e-2f;
    for (int i = 0; i < N * d; ++i) {
        REQUIRE_THAT(h_O_gpu[i], WithinAbs(h_O_cpu[i], tol));
    }
}
