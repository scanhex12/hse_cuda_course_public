#include "vector.cuh"

#include <cuda_helpers.h>

#include <catch2/benchmark/catch_benchmark.hpp>
#include <catch2/catch_test_macros.hpp>

#include <random>
#include <vector>

TEST_CASE("Benchmark: GPU performance") {
    const size_t N = 10000000;

    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(0.0f, 100.0f);

    std::vector<float> host_a(N);
    std::vector<float> host_b(N);
    for (size_t i = 0; i < N; ++i) {
        host_a[i] = dist(gen);
        host_b[i] = dist(gen);
    }

    CudaVector<float> a, b;
    a.copy_from_host(host_a.data(), N);
    b.copy_from_host(host_b.data(), N);

    BENCHMARK("GPU vector add (10M elements)") {
        auto result = a + b;
        return result.size();
    };

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    { auto warmup = a + b; }

    cudaEventRecord(start);
    auto result = a + b;
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float gpu_time_ms = 0.0f;
    cudaEventElapsedTime(&gpu_time_ms, start, stop);

    INFO("GPU time: " << gpu_time_ms << " ms");

    const float MAX_GPU_TIME_MS = 10.0f;
    REQUIRE(gpu_time_ms < MAX_GPU_TIME_MS);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}
