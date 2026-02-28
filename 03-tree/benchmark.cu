#include "tree.cuh"

#include <cuda_helpers.h>

#include <catch2/catch_test_macros.hpp>
#include <catch2/benchmark/catch_benchmark.hpp>

#include <random>
#include <vector>

TEST_CASE("Benchmark: GPU performance")
{
    const size_t N = 10000000;
    
    std::mt19937 gen(42);
    std::uniform_int_distribution<int> dist(0, N - 1);
    
    std::vector<int> parents(N);
    parents[0] = -1;
    for (size_t i = 1; i < N; ++i)
    {
        parents[i] = dist(gen) % i;
    }
    
    BENCHMARK("GPU findHeights (10M vertices)")
    {
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
