#include <cmath>
#include <vector>
#include <algorithm>
#include <numeric>
#include <random>

#include "moe.cuh"

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

using Catch::Matchers::WithinAbs;

void moe_topk_cpu(const float* X, float* top_weights, int* top_indices, int M, int E, int K) {
    std::vector<int> perm(E);
    for (int row = 0; row < M; ++row) {
        const float* row_data = X + row * E;
        std::iota(perm.begin(), perm.end(), 0);
        std::partial_sort(perm.begin(), perm.begin() + K, perm.end(),
            [row_data](int a, int b) { return row_data[a] > row_data[b]; });

        float max_val = row_data[perm[0]];
        float sum_exp = 0.0f;
        for (int k = 0; k < K; ++k) {
            float exp_val = std::exp(row_data[perm[k]] - max_val);
            sum_exp += exp_val;
        }
        for (int k = 0; k < K; ++k) {
            top_indices[row * K + k] = perm[k];
            top_weights[row * K + k] = std::exp(row_data[perm[k]] - max_val) / sum_exp;
        }
    }
}

TEST_CASE("MoE: example from README") {
    const int M = 2, E = 4, K = 2;
    std::vector<float> X = {1.0f, 2.0f, 3.0f, 4.0f,
                            4.0f, 3.0f, 2.0f, 1.0f};

    std::vector<float> h_weights(M * K);
    std::vector<int> h_indices(M * K);

    float *d_X, *d_weights;
    int *d_indices;
    cudaMalloc(&d_X, M * E * sizeof(float));
    cudaMalloc(&d_weights, M * K * sizeof(float));
    cudaMalloc(&d_indices, M * K * sizeof(int));

    cudaMemcpy(d_X, X.data(), M * E * sizeof(float), cudaMemcpyHostToDevice);

    choose_top_moe_experts(d_X, d_weights, d_indices, M, E, K);
    cudaDeviceSynchronize();

    cudaMemcpy(h_weights.data(), d_weights, M * K * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_indices.data(), d_indices, M * K * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_X);
    cudaFree(d_weights);
    cudaFree(d_indices);

    // Top-2 indices: row0 {2,3}, row1 {0,1}. Accept [2,3] or [3,2]
    bool row0_ok = (h_indices[0] == 2 && h_indices[1] == 3) || (h_indices[0] == 3 && h_indices[1] == 2);
    bool row1_ok = (h_indices[2] == 0 && h_indices[3] == 1) || (h_indices[2] == 1 && h_indices[3] == 0);
    REQUIRE(row0_ok);
    REQUIRE(row1_ok);

    const float tol = 1e-4f;
    REQUIRE_THAT(h_weights[0] + h_weights[1], WithinAbs(1.0f, tol));
    REQUIRE_THAT(h_weights[2] + h_weights[3], WithinAbs(1.0f, tol));
}

TEST_CASE("MoE: CPU vs GPU on random data (K=2)") {
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-2.0f, 2.0f);

    const int M = 256, E = 8, K = 2;

    std::vector<float> h_X(M * E);
    for (int i = 0; i < M * E; ++i)
        h_X[i] = dist(rng);

    std::vector<float> h_weights_cpu(M * K);
    std::vector<int> h_indices_cpu(M * K);
    moe_topk_cpu(h_X.data(), h_weights_cpu.data(), h_indices_cpu.data(), M, E, K);

    float *d_X, *d_weights;
    int *d_indices;
    cudaMalloc(&d_X, M * E * sizeof(float));
    cudaMalloc(&d_weights, M * K * sizeof(float));
    cudaMalloc(&d_indices, M * K * sizeof(int));

    cudaMemcpy(d_X, h_X.data(), M * E * sizeof(float), cudaMemcpyHostToDevice);

    choose_top_moe_experts(d_X, d_weights, d_indices, M, E, K);
    cudaDeviceSynchronize();

    std::vector<float> h_weights_gpu(M * K);
    std::vector<int> h_indices_gpu(M * K);
    cudaMemcpy(h_weights_gpu.data(), d_weights, M * K * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_indices_gpu.data(), d_indices, M * K * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_X);
    cudaFree(d_weights);
    cudaFree(d_indices);

    const float tol = 1e-4f;
    for (int row = 0; row < M; ++row) {
        for (int k = 0; k < K; ++k) {
            int gpu_idx = h_indices_gpu[row * K + k];
            bool found = false;
            for (int j = 0; j < K && !found; ++j)
                if (h_indices_cpu[row * K + j] == gpu_idx) found = true;
            REQUIRE(found);
        }
        float row_sum = 0.0f;
        for (int k = 0; k < K; ++k) row_sum += h_weights_gpu[row * K + k];
        REQUIRE_THAT(row_sum, WithinAbs(1.0f, tol));
    }
}

TEST_CASE("MoE: CPU vs GPU on random data (K=4)") {
    std::mt19937 rng(123);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    const int M = 128, E = 16, K = 4;

    std::vector<float> h_X(M * E);
    for (int i = 0; i < M * E; ++i)
        h_X[i] = dist(rng);

    std::vector<float> h_weights_cpu(M * K);
    std::vector<int> h_indices_cpu(M * K);
    moe_topk_cpu(h_X.data(), h_weights_cpu.data(), h_indices_cpu.data(), M, E, K);

    float *d_X, *d_weights;
    int *d_indices;
    cudaMalloc(&d_X, M * E * sizeof(float));
    cudaMalloc(&d_weights, M * K * sizeof(float));
    cudaMalloc(&d_indices, M * K * sizeof(int));

    cudaMemcpy(d_X, h_X.data(), M * E * sizeof(float), cudaMemcpyHostToDevice);

    choose_top_moe_experts(d_X, d_weights, d_indices, M, E, K);
    cudaDeviceSynchronize();

    std::vector<float> h_weights_gpu(M * K);
    std::vector<int> h_indices_gpu(M * K);
    cudaMemcpy(h_weights_gpu.data(), d_weights, M * K * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_indices_gpu.data(), d_indices, M * K * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_X);
    cudaFree(d_weights);
    cudaFree(d_indices);

    const float tol = 1e-4f;
    for (int row = 0; row < M; ++row) {
        for (int k = 0; k < K; ++k) {
            int gpu_idx = h_indices_gpu[row * K + k];
            bool found = false;
            for (int j = 0; j < K && !found; ++j)
                if (h_indices_cpu[row * K + j] == gpu_idx) found = true;
            REQUIRE(found);
        }
        float row_sum = 0.0f;
        for (int k = 0; k < K; ++k) row_sum += h_weights_gpu[row * K + k];
        REQUIRE_THAT(row_sum, WithinAbs(1.0f, tol));
    }
}

TEST_CASE("MoE: softmax sums to 1") {
    std::mt19937 rng(777);
    std::uniform_real_distribution<float> dist(-3.0f, 3.0f);

    const int M = 64, E = 32, K = 8;

    std::vector<float> h_X(M * E);
    for (int i = 0; i < M * E; ++i)
        h_X[i] = dist(rng);

    float *d_X, *d_weights;
    int *d_indices;
    cudaMalloc(&d_X, M * E * sizeof(float));
    cudaMalloc(&d_weights, M * K * sizeof(float));
    cudaMalloc(&d_indices, M * K * sizeof(int));

    cudaMemcpy(d_X, h_X.data(), M * E * sizeof(float), cudaMemcpyHostToDevice);

    choose_top_moe_experts(d_X, d_weights, d_indices, M, E, K);
    cudaDeviceSynchronize();

    std::vector<float> h_weights(M * K);
    cudaMemcpy(h_weights.data(), d_weights, M * K * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_X);
    cudaFree(d_weights);
    cudaFree(d_indices);

    for (int row = 0; row < M; ++row) {
        float row_sum = 0.0f;
        for (int k = 0; k < K; ++k)
            row_sum += h_weights[row * K + k];
        REQUIRE_THAT(row_sum, WithinAbs(1.0f, 1e-5f));
    }
}
