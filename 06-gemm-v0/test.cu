#include <cmath>
#include <iostream>
#include <vector>

#include "gemm.cuh"
#include <cuda_helpers.h>

#include <catch2/catch_test_macros.hpp>

static void gemm_cpu(float alpha, const float *A, const float *B, float beta,
                     float *C, int M, int K, int N) {
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = alpha * sum + beta * C[i * N + j];
        }
    }
}

static bool run_test(int M, int K, int N) {
    size_t size_A = M * K;
    size_t size_B = K * N;
    size_t size_C = M * N;

    std::vector<float> h_A(size_A);
    std::vector<float> h_B(size_B);
    for (size_t i = 0; i < size_A; ++i) {
        h_A[i] = (float)((i * 7 + 1) % 11 - 5);
    }
    for (size_t i = 0; i < size_B; ++i) {
        h_B[i] = (float)((i * 13 + 2) % 7 - 3);
    }

    std::vector<float> ref(size_C, 0.0f);
    gemm_cpu(1.0f, h_A.data(), h_B.data(), 0.0f, ref.data(), M, K, N);

    float *d_A = nullptr;
    float *d_B = nullptr;
    float *d_C = nullptr;
    CheckStatus(cudaMalloc(&d_A, size_A * sizeof(float)));
    CheckStatus(cudaMalloc(&d_B, size_B * sizeof(float)));
    CheckStatus(cudaMalloc(&d_C, size_C * sizeof(float)));
    CheckStatus(cudaMemcpy(d_A, h_A.data(), size_A * sizeof(float),
                           cudaMemcpyHostToDevice));
    CheckStatus(cudaMemcpy(d_B, h_B.data(), size_B * sizeof(float),
                           cudaMemcpyHostToDevice));

    gemm_cuda(1.0f, d_A, d_B, 0.0f, d_C, M, K, N);
    CheckStatus(cudaDeviceSynchronize());

    std::vector<float> gpu(size_C);
    CheckStatus(cudaMemcpy(gpu.data(), d_C, size_C * sizeof(float),
                           cudaMemcpyDeviceToHost));
    CheckStatus(cudaFree(d_A));
    CheckStatus(cudaFree(d_B));
    CheckStatus(cudaFree(d_C));

    for (size_t i = 0; i < size_C; ++i) {
        float diff = std::fabs(gpu[i] - ref[i]);
        if (diff > 1e-2f * (1.0f + std::fabs(ref[i]))) {
            std::cerr << "Mismatch at " << i << " got " << gpu[i] << " ref "
                      << ref[i] << "\n";
            return false;
        }
    }
    return true;
}

static bool run_gemm_test(int M, int K, int N, float alpha, float beta) {
    size_t size_A = M * K;
    size_t size_B = K * N;
    size_t size_C = M * N;

    std::vector<float> h_A(size_A);
    std::vector<float> h_B(size_B);
    std::vector<float> h_C(size_C, 1.0f);
    for (size_t i = 0; i < size_A; ++i) {
        h_A[i] = (float)((i * 7 + 1) % 11 - 5);
    }
    for (size_t i = 0; i < size_B; ++i) {
        h_B[i] = (float)((i * 13 + 2) % 7 - 3);
    }

    std::vector<float> ref = h_C;
    gemm_cpu(alpha, h_A.data(), h_B.data(), beta, ref.data(), M, K, N);

    float *d_A = nullptr;
    float *d_B = nullptr;
    float *d_C = nullptr;
    CheckStatus(cudaMalloc(&d_A, size_A * sizeof(float)));
    CheckStatus(cudaMalloc(&d_B, size_B * sizeof(float)));
    CheckStatus(cudaMalloc(&d_C, size_C * sizeof(float)));
    CheckStatus(cudaMemcpy(d_A, h_A.data(), size_A * sizeof(float),
                           cudaMemcpyHostToDevice));
    CheckStatus(cudaMemcpy(d_B, h_B.data(), size_B * sizeof(float),
                           cudaMemcpyHostToDevice));
    CheckStatus(cudaMemcpy(d_C, h_C.data(), size_C * sizeof(float),
                           cudaMemcpyHostToDevice));

    gemm_cuda(alpha, d_A, d_B, beta, d_C, M, K, N);
    CheckStatus(cudaDeviceSynchronize());

    std::vector<float> gpu(size_C);
    CheckStatus(cudaMemcpy(gpu.data(), d_C, size_C * sizeof(float),
                           cudaMemcpyDeviceToHost));
    CheckStatus(cudaFree(d_A));
    CheckStatus(cudaFree(d_B));
    CheckStatus(cudaFree(d_C));

    for (size_t i = 0; i < size_C; ++i) {
        float diff = std::fabs(gpu[i] - ref[i]);
        if (diff > 1e-2f * (1.0f + std::fabs(ref[i]))) {
            std::cerr << "GEMM mismatch at " << i << " got " << gpu[i]
                      << " ref " << ref[i] << "\n";
            return false;
        }
    }
    return true;
}

TEST_CASE("GEMM (tiled, shared memory)") {
    REQUIRE(run_test(1, 1, 1));
    REQUIRE(run_test(16, 16, 16));
    REQUIRE(run_test(32, 32, 32));
    REQUIRE(run_test(17, 31, 23));
    REQUIRE(run_test(100, 50, 80));
    REQUIRE(run_test(64, 128, 256));
    REQUIRE(run_test(256, 256, 256));
    REQUIRE(run_test(512, 512, 512));
}

TEST_CASE("GEMM: alpha * A*B + beta * C") {
    REQUIRE(run_gemm_test(16, 16, 16, 2.0f, 0.0f));
    REQUIRE(run_gemm_test(32, 32, 32, 1.0f, 1.0f));
    REQUIRE(run_gemm_test(24, 48, 32, -1.0f, 0.5f));
}
