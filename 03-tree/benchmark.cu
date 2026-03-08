#include "tree.cuh"

#include <cuda_helpers.h>

#include <catch2/benchmark/catch_benchmark.hpp>
#include <catch2/catch_test_macros.hpp>

#include <random>
#include <vector>

static void select_least_loaded_gpu() {
    int ndev = 0;
    if (cudaGetDeviceCount(&ndev) != cudaSuccess || ndev <= 0) {
        return;
    }
    size_t best_free = 0;
    int best = 0;
    for (int i = 0; i < ndev; ++i) {
        if (cudaSetDevice(i) != cudaSuccess) {
            continue;
        }
        size_t free_mem = 0;
        size_t total = 0;
        if (cudaMemGetInfo(&free_mem, &total) == cudaSuccess &&
            free_mem > best_free) {
            best_free = free_mem;
            best = i;
        }
    }
    cudaSetDevice(best);
}

TEST_CASE("Benchmark: GPU performance") {
    select_least_loaded_gpu();
    const size_t N = 10000000;

    std::mt19937 gen(42);
    std::uniform_int_distribution<int> dist(0, N - 1);

    std::vector<int> parents(N);
    parents[0] = -1;
    for (size_t i = 1; i < N; ++i) {
        parents[i] = dist(gen) % i;
    }

    BENCHMARK("GPU findHeights (10M vertices)") {
        auto result = findHeights(parents.data(), N);
        delete[] result;
        return N;
    };

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    {
        auto warmup = findHeights(parents.data(), N);
        delete[] warmup;
    }

    cudaEventRecord(start);
    auto result = findHeights(parents.data(), N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float gpu_time_ms = 0.0f;
    cudaEventElapsedTime(&gpu_time_ms, start, stop);

    INFO("GPU time: " << gpu_time_ms << " ms");

    const float MAX_GPU_TIME_MS = 60.0f;
    REQUIRE(gpu_time_ms < MAX_GPU_TIME_MS);

    delete[] result;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}
