#include "binary_image.cuh"

#include <cuda_helpers.h>

#include <catch2/benchmark/catch_benchmark.hpp>
#include <catch2/catch_test_macros.hpp>

#include <random>

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
    const size_t N = 1000;
    const size_t M = 1000;

    std::mt19937 gen(42);
    std::uniform_int_distribution<int> dist(0, 9);

    auto src_matrix = allocMatrixHost(N, M);
    for (size_t i = 0; i < N * M; ++i) {
        src_matrix.data[i] = (dist(gen) < 1) ? 1 : 0;
    }

    BENCHMARK("GPU solve (1000x1000)") {
        auto result = solve(src_matrix);
        freeMatrixHost(result);
        return result.N * result.M;
    };

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    {
        auto warmup = solve(src_matrix);
        freeMatrixHost(warmup);
    }

    cudaEventRecord(start);
    auto gpu_result = solve(src_matrix);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float gpu_time_ms = 0.0f;
    cudaEventElapsedTime(&gpu_time_ms, start, stop);

    INFO("GPU time: " << gpu_time_ms << " ms");

    const float MAX_GPU_TIME_MS = 25.0f;
    REQUIRE(gpu_time_ms < MAX_GPU_TIME_MS);

    freeMatrixHost(src_matrix);
    freeMatrixHost(gpu_result);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}
