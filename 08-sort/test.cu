// test.cu
#include <algorithm>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "sort.cuh"
#include <cuda_helpers.h>

#include <catch2/catch_test_macros.hpp>

inline void select_least_loaded_gpu() {
    int ndev = 0;
    if (cudaGetDeviceCount(&ndev) != cudaSuccess || ndev <= 0)
        return;
    size_t best_free = 0;
    int best = 0;
    for (int i = 0; i < ndev; ++i) {
        if (cudaSetDevice(i) != cudaSuccess)
            continue;
        size_t free_mem = 0, total = 0;
        if (cudaMemGetInfo(&free_mem, &total) == cudaSuccess &&
            free_mem > best_free) {
            best_free = free_mem;
            best = i;
        }
    }
    cudaSetDevice(best);
}

template <typename T>
static std::vector<T> make_random_ints(int n, int lo, int hi, uint32_t seed) {
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> dist(lo, hi);
    std::vector<T> v(n);
    for (int i = 0; i < n; ++i)
        v[i] = (T)dist(rng);
    return v;
}

static bool run_one_test(const std::vector<int> &h_in,
                         const std::string &name) {
    int n = (int)h_in.size();
    std::vector<int> ref = h_in;
    std::sort(ref.begin(), ref.end());

    std::vector<int> out(n);

    if (n == 0) {
        device_sort_int(nullptr, n);
        std::cout << "[ OK ] " << name << " (n=0)\n";
        return true;
    }

    int *d_data = nullptr;
    size_t bytes = (size_t)n * sizeof(int);

    CheckStatus(cudaMalloc(&d_data, bytes));
    CheckStatus(cudaMemcpy(d_data, h_in.data(), bytes, cudaMemcpyHostToDevice));

    device_sort_int(d_data, n);
    CheckStatus(cudaDeviceSynchronize());

    CheckStatus(cudaMemcpy(out.data(), d_data, bytes, cudaMemcpyDeviceToHost));
    CheckStatus(cudaFree(d_data));

    bool ok = true;
    for (int i = 0; i < n; ++i) {
        if (out[i] != ref[i]) {
            ok = false;
            std::cerr << "[FAIL] " << name << " at i=" << i << " got=" << out[i]
                      << " expected=" << ref[i] << "\n";
            break;
        }
    }
    if (ok)
        std::cout << "[ OK ] " << name << " (n=" << n << ")\n";
    return ok;
}

TEST_CASE("Sort") {
    select_least_loaded_gpu();
    int dev = 0;
    size_t free_mem = 0, total = 0;
    if (cudaGetDevice(&dev) == cudaSuccess &&
        cudaMemGetInfo(&free_mem, &total) == cudaSuccess) {
        std::cerr << "Sort tests using GPU " << dev << " ("
                  << (free_mem / (1024 * 1024)) << " MiB free / "
                  << (total / (1024 * 1024)) << " MiB total)\n";
    }

    {
        std::vector<int> v0;
        REQUIRE(run_one_test(v0, "empty"));

        std::vector<int> v1 = {7};
        REQUIRE(run_one_test(v1, "single"));

        std::vector<int> v2 = {5, 1, 9, 3, 7, 2, 8, 6, 4};
        REQUIRE(run_one_test(v2, "small_known"));

        std::vector<int> v3 = {0, 0, 0, 0, 0};
        REQUIRE(run_one_test(v3, "all_zeros"));

        std::vector<int> v4 = {-1, -2, -3, 4, 5};
        REQUIRE(run_one_test(v4, "negatives"));

        std::vector<int> v5 = {2, 2, 2, 1, 1, 3, 3, 0, 0};
        REQUIRE(run_one_test(v5, "duplicates"));
    }

    {
        std::vector<int> sizes = {2,   4,    8,    16,   32,   64,    128,  256,
                                  512, 1024, 2048, 4096, 8192, 16384, 65536};
        for (int n : sizes) {
            auto v = make_random_ints<int>(n, -50, 50, 777u + (uint32_t)n);
            REQUIRE(run_one_test(v, "boundary_n=" + std::to_string(n)));
        }
    }

    {
        for (int exp = 20; exp <= 24; ++exp) {
            int n = 1 << exp;
            auto v = make_random_ints<int>(n, -100, 100, 999u + (uint32_t)exp);
            REQUIRE(run_one_test(v, "large_n=" + std::to_string(n)));
        }
    }
}