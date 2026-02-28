#include "binary_image.cuh"

#include <cuda_helpers.h>

#include <catch2/catch_test_macros.hpp>
#include <catch2/benchmark/catch_benchmark.hpp>

#include <random>

TEST_CASE("Benchmark: GPU performance")
{
    const size_t N = 1000;
    const size_t M = 1000;
    
    std::mt19937 gen(42);
    std::uniform_int_distribution<int> dist(0, 9);
    
    auto src_matrix = allocMatrixHost(N, M);
    for (size_t i = 0; i < N * M; ++i)
    {
        src_matrix.data[i] = (dist(gen) < 1) ? 1 : 0;
    }
    
    BENCHMARK("GPU solve (1000x1000)")
    {
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
    
    const float MAX_GPU_TIME_MS = 20.0f;
    REQUIRE(gpu_time_ms < MAX_GPU_TIME_MS);
    
    freeMatrixHost(src_matrix);
    freeMatrixHost(gpu_result);
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}
