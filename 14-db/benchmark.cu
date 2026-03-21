#include "db.cuh"

#include <catch2/catch_test_macros.hpp>

#include <cuda_runtime.h>

#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

static void fill_table(std::vector<std::vector<int>> &cols, int n,
                       unsigned seed) {
    std::mt19937 gen(seed);
    std::uniform_int_distribution<int> dist(-1000, 1000);
    for (auto &col : cols) {
        col.resize(static_cast<size_t>(n));
        for (int i = 0; i < n; ++i) {
            col[static_cast<size_t>(i)] = dist(gen);
        }
    }
}

static float time_execute(Database &db, const std::string &query,
                          int warmup_iters, int timed_iters) {
    for (int w = 0; w < warmup_iters; ++w) {
        (void)db.Execute(query);
    }
    cudaDeviceSynchronize();

    cudaEvent_t start{};
    cudaEvent_t stop{};
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaDeviceSynchronize();
    cudaEventRecord(start);
    for (int i = 0; i < timed_iters; ++i) {
        (void)db.Execute(query);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return ms;
}

TEST_CASE("Benchmark: GPU database expression graph") {
    const int n = 1 << 18;
    std::vector<std::vector<int>> data(3);
    fill_table(data, n, 12345u);

    Database db;
    db.AddTable("big", {"a", "b", "c"}, data);

    const std::string query = "(big.a + big.b) * big.c - big.a";
    constexpr int kWarmup = 3;
    constexpr int kTimed = 12;

    float ms = time_execute(db, query, kWarmup, kTimed);
    float ms_per = ms / static_cast<float>(kTimed);

    std::cerr << std::fixed << std::setprecision(3) << "DB graph Execute x"
              << kTimed << " (n=" << n << "): total " << ms << " ms, per call "
              << ms_per << " ms\n";

    REQUIRE(ms_per < 2000.0f);
}
