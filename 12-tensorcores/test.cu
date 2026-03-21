#include <cmath>
#include <cstdlib>
#include <iostream>
#include <random>
#include <vector>

#include "lu_decomposition.cuh"

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

using Catch::Matchers::WithinAbs;

static void matrix_multiply(const float *A, const float *B, float *C, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            float sum = 0.0f;
            for (int k = 0; k < n; k++) {
                sum += A[i * n + k] * B[k * n + j];
            }
            C[i * n + j] = sum;
        }
    }
}

static bool matrices_approx_equal(const float *A, const float *B, int n,
                                  float tolerance,
                                  float *out_max_diff = nullptr) {
    float max_diff = 0;
    for (int i = 0; i < n * n; i++) {
        float diff = std::abs(A[i] - B[i]);
        if (diff > max_diff)
            max_diff = diff;
    }
    if (out_max_diff)
        *out_max_diff = max_diff;
    return max_diff <= tolerance;
}

static void cpu_lu_decomposition(const float *A, float *L, float *U, int n) {
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            L[i * n + j] = (i == j) ? 1.0f : 0.0f;
            U[i * n + j] = 0.0f;
        }
    }

    for (int k = 0; k < n; ++k) {
        for (int j = k; j < n; ++j) {
            float sum = 0.0f;
            for (int t = 0; t < k; ++t) {
                sum += L[k * n + t] * U[t * n + j];
            }
            U[k * n + j] = A[k * n + j] - sum;
        }

        float piv = U[k * n + k];
        if (fabsf(piv) < 1e-20f)
            piv = (piv >= 0.0f ? 1e-20f : -1e-20f);

        for (int i = k + 1; i < n; ++i) {
            float sum = 0.0f;
            for (int t = 0; t < k; ++t) {
                sum += L[i * n + t] * U[t * n + k];
            }
            L[i * n + k] = (A[i * n + k] - sum) / piv;
        }
    }
}

template <int N>
static bool run_one_test(const std::vector<float> &h_A,
                         const std::string &name) {
    std::vector<float> h_L_ref(N * N);
    std::vector<float> h_U_ref(N * N);
    std::vector<float> h_L(N * N);
    std::vector<float> h_U(N * N);
    std::vector<float> h_reconstructed(N * N);

    cpu_lu_decomposition(h_A.data(), h_L_ref.data(), h_U_ref.data(), N);

    float *d_A = nullptr, *d_L = nullptr, *d_U = nullptr;
    cudaMalloc(&d_A, N * N * sizeof(float));
    cudaMalloc(&d_L, N * N * sizeof(float));
    cudaMalloc(&d_U, N * N * sizeof(float));

    cudaMemcpy(d_A, h_A.data(), N * N * sizeof(float), cudaMemcpyHostToDevice);

    lu_decomposition(d_A, d_L, d_U, N);

    cudaMemcpy(h_L.data(), d_L, N * N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_U.data(), d_U, N * N * sizeof(float), cudaMemcpyDeviceToHost);

    matrix_multiply(h_L.data(), h_U.data(), h_reconstructed.data(), N);

    float tolerance = 0.5f;
    float max_diff = 0;
    bool ok = matrices_approx_equal(h_A.data(), h_reconstructed.data(), N,
                                    tolerance, &max_diff);

    if (!ok) {
        std::cerr << "[FAIL] " << name << " max|A-LU|=" << max_diff
                  << " (tolerance=" << tolerance << ")\n";
    } else {
        std::cout << "[ OK ] " << name << " (n=" << N << ")\n";
    }

    cudaFree(d_A);
    cudaFree(d_L);
    cudaFree(d_U);

    return ok;
}

static std::vector<float> make_random_matrix(int n, float lo = -2.0f,
                                             float hi = 2.0f,
                                             uint32_t seed = 42) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(lo, hi);
    std::vector<float> v(n * n);
    for (int i = 0; i < n * n; i++) {
        v[i] = dist(rng);
    }
    for (int i = 0; i < n; i++) {
        float sum = 0.0f;
        for (int j = 0; j < n; j++) {
            if (i != j) {
                sum += std::abs(v[i * n + j]);
            }
        }
        v[i * n + i] = sum + 1.0f;
    }
    return v;
}

TEST_CASE("LU Decomposition - Small Matrices") {
    {
        std::vector<float> A = {4.0f, 3.0f, 6.0f, 3.0f};
        REQUIRE(run_one_test<2>(A, "2x2_simple"));
    }

    {
        std::vector<float> A = {2.0f, 1.0f, 0.0f, 1.0f, 2.0f,
                                1.0f, 0.0f, 1.0f, 2.0f};
        REQUIRE(run_one_test<3>(A, "3x3_tridiagonal"));
    }

    {
        auto A = make_random_matrix(4, -1.0f, 1.0f, 42u);
        REQUIRE(run_one_test<4>(A, "4x4_random"));
    }
}

TEST_CASE("LU Decomposition - Medium Matrices") {
    {
        auto A = make_random_matrix(8, -2.0f, 2.0f, 100u);
        REQUIRE(run_one_test<8>(A, "8x8_random"));
    }

    {
        auto A = make_random_matrix(16, -1.0f, 1.0f, 200u);
        REQUIRE(run_one_test<16>(A, "16x16_random"));
    }

    {
        auto A = make_random_matrix(32, -1.0f, 1.0f, 300u);
        REQUIRE(run_one_test<32>(A, "32x32_random"));
    }
}

TEST_CASE("LU Decomposition - Large Matrices") {
    {
        auto A = make_random_matrix(64, -1.0f, 1.0f, 400u);
        REQUIRE(run_one_test<64>(A, "64x64_random"));
    }

    {
        auto A = make_random_matrix(128, -0.5f, 0.5f, 500u);
        REQUIRE(run_one_test<128>(A, "128x128_random"));
    }
}
