#include "delta_stepping.cuh"

#include <catch2/catch_test_macros.hpp>

#include <cstdint>
#include <random>
#include <tuple>
#include <utility>
#include <vector>

#include <cuda_helpers.h>

using namespace delta_stepping;

namespace {

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
        if (cudaMemGetInfo(&free_mem, &total) == cudaSuccess && free_mem > best_free) {
            best_free = free_mem;
            best = i;
        }
    }
    cudaSetDevice(best);
}

void build_csr(int n, const std::vector<std::tuple<int, int, int>>& edges,
               std::vector<int>& csr_row_ptr, std::vector<int>& csr_col_idx,
               std::vector<int>& edge_weight) {
    csr_row_ptr.assign((size_t)n + 1, 0);
    for (const auto& e : edges) {
        int u = std::get<0>(e);
        if (u >= 0 && u < n)
            csr_row_ptr[(size_t)u + 1]++;
    }
    for (int i = 0; i < n; ++i)
        csr_row_ptr[(size_t)i + 1] += csr_row_ptr[(size_t)i];

    std::vector<int> next = csr_row_ptr;
    csr_col_idx.resize(edges.size());
    edge_weight.resize(edges.size());
    for (const auto& e : edges) {
        int u = std::get<0>(e), v = std::get<1>(e), w = std::get<2>(e);
        if (u >= 0 && u < n) {
            size_t pos = (size_t)next[u]++;
            csr_col_idx[pos] = v;
            edge_weight[pos] = w;
        }
    }
}

std::vector<std::tuple<int, int, int>> make_grid_edges(int side) {
    std::vector<std::tuple<int, int, int>> edges;
    const int n = side * side;
    edges.reserve((size_t)2 * n);
    auto id = [side](int r, int c) { return r * side + c; };
    for (int r = 0; r < side; ++r) {
        for (int c = 0; c < side; ++c) {
            int u = id(r, c);
            if (c + 1 < side)
                edges.emplace_back(u, id(r, c + 1), 1);
            if (r + 1 < side)
                edges.emplace_back(u, id(r + 1, c), 1);
        }
    }
    return edges;
}

std::vector<std::tuple<int, int, int>> make_wide_tree_edges(int n, int fanout) {
    std::vector<std::tuple<int, int, int>> edges;
    if (n <= 1 || fanout < 1)
        return edges;
    edges.reserve((size_t)n - 1);
    for (int v = 1; v < n; ++v) {
        int p = (v - 1) / fanout;
        edges.emplace_back(p, v, 1);
    }
    return edges;
}

std::vector<std::tuple<int, int, int>> make_random_edges(int n, int out_deg, uint32_t seed) {
    std::vector<std::tuple<int, int, int>> edges;
    edges.reserve((size_t)n * (size_t)out_deg);
    std::mt19937 gen(seed);
    std::uniform_int_distribution<int> v_dist(0, n - 1);
    std::uniform_int_distribution<int> w_dist(1, 32);
    for (int u = 0; u < n; ++u)
        for (int k = 0; k < out_deg; ++k)
            edges.emplace_back(u, v_dist(gen), w_dist(gen));
    return edges;
}

float benchmark_avg_ms(int n, const int* csr_row_ptr, const int* csr_col_idx,
                       const int* edge_weight, std::vector<int>& dist_buf, int source, int delta,
                       int warmup, int reps) {
    for (int i = 0; i < warmup; ++i)
        shortest_paths(n, csr_row_ptr, csr_col_idx, edge_weight, source, delta, dist_buf.data());

    cudaEvent_t start{};
    cudaEvent_t stop{};
    CheckStatus(cudaEventCreate(&start));
    CheckStatus(cudaEventCreate(&stop));

    float total_ms = 0.f;
    for (int i = 0; i < reps; ++i) {
        CheckStatus(cudaEventRecord(start));
        shortest_paths(n, csr_row_ptr, csr_col_idx, edge_weight, source, delta, dist_buf.data());
        CheckStatus(cudaEventRecord(stop));
        CheckStatus(cudaEventSynchronize(stop));
        float ms = 0.f;
        CheckStatus(cudaEventElapsedTime(&ms, start, stop));
        total_ms += ms;
    }

    CheckStatus(cudaEventDestroy(stop));
    CheckStatus(cudaEventDestroy(start));
    return total_ms / (float)reps;
}

}

TEST_CASE("Benchmark: shortest paths grid") {
    select_least_loaded_gpu();

    const int side = 256;
    const int n = side * side;
    auto edges = make_grid_edges(side);

    std::vector<int> csr_row_ptr, csr_col_idx, edge_weight;
    build_csr(n, edges, csr_row_ptr, csr_col_idx, edge_weight);

    std::vector<int> dist_buf((size_t)n);

    const int delta = 16;
    float avg_ms =
        benchmark_avg_ms(n, csr_row_ptr.data(), csr_col_idx.data(), edge_weight.data(), dist_buf,
                         0, delta, 1, 5);

    REQUIRE(avg_ms < 2800.f);
}

TEST_CASE("Benchmark: shortest paths wide shallow tree") {
    select_least_loaded_gpu();

    const int n = 65536;
    const int fanout = 64;
    auto edges = make_wide_tree_edges(n, fanout);

    std::vector<int> csr_row_ptr, csr_col_idx, edge_weight;
    build_csr(n, edges, csr_row_ptr, csr_col_idx, edge_weight);

    std::vector<int> dist_buf((size_t)n);

    const int delta = 16;
    float avg_ms =
        benchmark_avg_ms(n, csr_row_ptr.data(), csr_col_idx.data(), edge_weight.data(), dist_buf,
                         0, delta, 1, 8);

    REQUIRE(avg_ms < 35.f);
}

TEST_CASE("Benchmark: shortest paths random sparse") {
    select_least_loaded_gpu();

    const int n = 65536;
    const int out_deg = 8;
    auto edges = make_random_edges(n, out_deg, 12345);

    std::vector<int> csr_row_ptr, csr_col_idx, edge_weight;
    build_csr(n, edges, csr_row_ptr, csr_col_idx, edge_weight);

    std::vector<int> dist_buf((size_t)n);

    const int delta = 16;
    float avg_ms =
        benchmark_avg_ms(n, csr_row_ptr.data(), csr_col_idx.data(), edge_weight.data(), dist_buf,
                         0, delta, 1, 6);

    REQUIRE(avg_ms > 0.f);
    REQUIRE(avg_ms < 240.f);
}
