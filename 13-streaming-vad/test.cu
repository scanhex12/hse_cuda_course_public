#include "streaming_vad.cuh"

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include <cmath>
#include <vector>

static void cpu_streaming_vad(const float *host_input, unsigned char *host_vad,
                              int num_chunks, int samples_per_chunk,
                              float threshold) {
    for (int i = 0; i < num_chunks; ++i) {
        float m = 0.0f;
        const float *p =
            host_input + static_cast<size_t>(i) * samples_per_chunk;
        for (int j = 0; j < samples_per_chunk; ++j) {
            m = fmaxf(m, fabsf(p[j]));
        }
        host_vad[i] = (m >= threshold) ? 1 : 0;
    }
}

TEST_CASE("VAD: silence vs speech") {
    const int nch = 4;
    const int spc = 64;
    std::vector<float> in(static_cast<size_t>(nch) * spc, 0.0f);
    for (int j = 0; j < spc; ++j) {
        in[static_cast<size_t>(1) * spc + j] = 0.9f;
    }

    std::vector<unsigned char> gpu(nch, 0);
    std::vector<unsigned char> cpu(nch, 0);

    run_streaming_vad(in.data(), gpu.data(), nch, spc, 0.5f, 2);
    cpu_streaming_vad(in.data(), cpu.data(), nch, spc, 0.5f);

    REQUIRE(gpu == cpu);
    REQUIRE(gpu[0] == 0);
    REQUIRE(gpu[1] == 1);
}

TEST_CASE("VAD: matches CPU reference for random-ish data") {
    const int nch = 32;
    const int spc = 128;
    std::vector<float> in(static_cast<size_t>(nch) * spc);
    for (size_t k = 0; k < in.size(); ++k) {
        in[k] = std::sin(0.03f * static_cast<float>(k)) * 0.4f;
    }
    in[100] = 0.95f;

    const float thr = 0.5f;
    const int nstreams = GENERATE(1, 2, 4);

    std::vector<unsigned char> gpu(static_cast<size_t>(nch));
    std::vector<unsigned char> cpu(static_cast<size_t>(nch));

    run_streaming_vad(in.data(), gpu.data(), nch, spc, thr, nstreams);
    cpu_streaming_vad(in.data(), cpu.data(), nch, spc, thr);

    REQUIRE(gpu == cpu);
}

TEST_CASE("VAD: single chunk threshold boundary") {
    const int spc = 32;
    std::vector<float> in(static_cast<size_t>(spc), 0.2f);

    std::vector<unsigned char> a(1), b(1);
    run_streaming_vad(in.data(), a.data(), 1, spc, 0.25f, 1);
    run_streaming_vad(in.data(), b.data(), 1, spc, 0.15f, 3);
    REQUIRE(a[0] == 0);
    REQUIRE(b[0] == 1);
}
