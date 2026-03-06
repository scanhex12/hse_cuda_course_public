#include <cmath>
#include <cstdlib>
#include <iostream>
#include <random>
#include <vector>

#include "prefsum.cuh"

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

using Catch::Matchers::WithinAbs;

static std::vector<float> cpu_exclusive_scan_float(const std::vector<float> &in) {
    std::vector<float> out(in.size());
    float acc = 0.0f;
    for (size_t i = 0; i < in.size(); ++i) {
        out[i] = acc;
        acc += in[i];
    }
    return out;
}

static bool run_one_test(const std::vector<float> &h_in,
                         const std::string &name) {
    int n = static_cast<int>(h_in.size());
    std::vector<float> h_ref = cpu_exclusive_scan_float(h_in);
    std::vector<float> h_out(n);

    if (n <= 0) {
        std::cout << "[ OK ] " << name << " (n=0)\n";
        return true;
    }

    float *d_in = nullptr;
    float *d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_in, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out, n * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_in, h_in.data(), n * sizeof(float),
                          cudaMemcpyHostToDevice));

    exclusive_scan_cuda(d_out, d_in, n);

    CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, n * sizeof(float),
                          cudaMemcpyDeviceToHost));

    bool ok = true;
    const float tol = 5e-2f;
    for (int i = 0; i < n; ++i) {
        float diff = std::fabs(h_out[i] - h_ref[i]);
        float scale = std::max(std::fabs(h_ref[i]), 1.0f);
        if (diff > tol * scale) {
            ok = false;
            std::cerr << "[FAIL] " << name << " at i=" << i
                      << " got=" << h_out[i] << " expected=" << h_ref[i]
                      << " diff=" << diff << "\n";
            break;
        }
    }
    if (ok) {
        std::cout << "[ OK ] " << name << " (n=" << n << ")\n";
    }

    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));
    return ok;
}

static std::vector<float> make_random_floats(int n, float lo = -5.0f,
                                             float hi = 10.0f,
                                             uint32_t seed = 12345) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(lo, hi);
    std::vector<float> v(n);
    for (int i = 0; i < n; ++i) {
        v[i] = dist(rng);
    }
    return v;
}

TEST_CASE("Prefsum") {
    {
        std::vector<float> v0;
        REQUIRE(run_one_test(v0, "empty"));

        std::vector<float> v1 = {7.0f};
        REQUIRE(run_one_test(v1, "single"));

        std::vector<float> v2 = {1.0f, 2.0f, 3.0f, 4.0f};
        // exclusive: [0, 1, 3, 6]
        REQUIRE(run_one_test(v2, "small_known"));

        std::vector<float> v3 = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
        REQUIRE(run_one_test(v3, "all_zeros"));

        std::vector<float> v4 = {-1.0f, -2.0f, -3.0f, 4.0f, 5.0f};
        REQUIRE(run_one_test(v4, "negatives"));
    }

    {
        std::vector<int> sizes = {2,    3,    31,   32,   33,   255,
                                  256,  257,  511,  512,  513,  1000,
                                  1023, 1024, 1025, 4095, 4096, 4097};
        for (int n : sizes) {
            auto v = make_random_floats(n, -3.0f, 7.0f,
                                        777u + static_cast<uint32_t>(n));
            REQUIRE(run_one_test(v, "boundary_n=" + std::to_string(n)));
        }
    }

    {
        for (int t = 0; t < 5; ++t) {
            int n = 1'000'000 + t * 12345;
            auto v = make_random_floats(n, -2.0f, 5.0f,
                                        999u + static_cast<uint32_t>(t));
            REQUIRE(run_one_test(v, "large_random_" + std::to_string(t)));
        }
    }

    {
        std::mt19937 rng(2025);
        std::uniform_int_distribution<int> ndist(1, 200000);
        for (int t = 0; t < 20; ++t) {
            int n = ndist(rng);
            auto v = make_random_floats(n, -10.0f, 10.0f,
                                        4242u + static_cast<uint32_t>(t));
            REQUIRE(run_one_test(v, "fuzz_" + std::to_string(t) +
                                        "_n=" + std::to_string(n)));
        }
    }
}
