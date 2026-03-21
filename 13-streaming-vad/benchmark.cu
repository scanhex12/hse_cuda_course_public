#include "streaming_vad.cuh"

#include <catch2/catch_test_macros.hpp>

#include <cuda_runtime.h>

#include <cmath>
#include <iomanip>
#include <iostream>
#include <vector>

static float time_vad(const std::vector<float> &in,
                      std::vector<unsigned char> &vad, int num_chunks,
                      int samples_per_chunk, float threshold, int num_streams) {
    constexpr int kWarmupIters = 3;
    for (int w = 0; w < kWarmupIters; ++w) {
        run_streaming_vad(in.data(), vad.data(), num_chunks, samples_per_chunk,
                          threshold, num_streams);
    }
    cudaDeviceSynchronize();

    cudaEvent_t start{}, stop{};
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaDeviceSynchronize();
    cudaEventRecord(start);
    run_streaming_vad(in.data(), vad.data(), num_chunks, samples_per_chunk,
                      threshold, num_streams);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return ms;
}

TEST_CASE("Benchmark: streaming VAD many chunks") {
    const int num_chunks = 4096;
    const int samples_per_chunk = 512;
    const float thr = 0.05f;

    std::vector<float> in(static_cast<size_t>(num_chunks) * samples_per_chunk);
    for (size_t i = 0; i < in.size(); ++i) {
        in[i] = 0.04f * std::sin(0.01f * static_cast<float>(i));
    }
    for (int c = 0; c < num_chunks; c += 17) {
        in[static_cast<size_t>(c) * samples_per_chunk + 3] = 0.2f;
    }

    std::vector<unsigned char> vad(static_cast<size_t>(num_chunks));

    float ms1 = time_vad(in, vad, num_chunks, samples_per_chunk, thr, 1);
    float ms8 = time_vad(in, vad, num_chunks, samples_per_chunk, thr, 8);

    std::cerr << std::fixed << std::setprecision(3) << "VAD " << num_chunks
              << " chunks x " << samples_per_chunk << " samples: 1 stream "
              << ms1 << " ms, 8 streams " << ms8 << " ms\n";

    REQUIRE(ms1 < 100.0f);
    REQUIRE(ms8 < 35.0f);
}
