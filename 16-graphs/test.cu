#include <algorithm>
#include <queue>
#include <tuple>
#include <utility>
#include <vector>

#include "delta_stepping.cuh"

#include <catch2/catch_test_macros.hpp>

using namespace delta_stepping;

namespace {
constexpr int REF_INF = delta_stepping::INF;
}

static void build_csr(int n, const std::vector<std::tuple<int, int, int>>& edges,
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

static void dijkstra_cpu(int n, const std::vector<int>& csr_row_ptr,
                         const std::vector<int>& csr_col_idx,
                         const std::vector<int>& edge_weight, int source,
                         std::vector<int>& dist) {
    dist.assign((size_t)n, REF_INF);
    dist[(size_t)source] = 0;
    std::priority_queue<std::pair<int, int>, std::vector<std::pair<int, int>>, std::greater<>>
        pq;
    pq.push({0, source});

    while (!pq.empty()) {
        int d = pq.top().first;
        int u = pq.top().second;
        pq.pop();
        if (d != dist[(size_t)u])
            continue;
        for (int e = csr_row_ptr[(size_t)u]; e < csr_row_ptr[(size_t)u + 1]; ++e) {
            int v = csr_col_idx[(size_t)e];
            int w = edge_weight[(size_t)e];
            int new_d = d + w;
            if (new_d < dist[(size_t)v]) {
                dist[(size_t)v] = new_d;
                pq.push({new_d, v});
            }
        }
    }
}

static bool run_one_test(int n, const std::vector<std::tuple<int, int, int>>& edges, int source,
                         int delta) {
    std::vector<int> csr_row_ptr, csr_col_idx, edge_weight;
    build_csr(n, edges, csr_row_ptr, csr_col_idx, edge_weight);

    std::vector<int> ref_dist((size_t)n);
    dijkstra_cpu(n, csr_row_ptr, csr_col_idx, edge_weight, source, ref_dist);

    std::vector<int> gpu_dist((size_t)n);
    shortest_paths(n, csr_row_ptr.data(), csr_col_idx.data(), edge_weight.data(), source, delta,
                   gpu_dist.data());

    for (int i = 0; i < n; ++i) {
        if (gpu_dist[(size_t)i] != ref_dist[(size_t)i])
            return false;
    }
    return true;
}

TEST_CASE("Shortest paths (delta-stepping)") {
    {
        int n = 1;
        std::vector<std::tuple<int, int, int>> edges;
        REQUIRE(run_one_test(n, edges, 0, 1));
    }
    {
        int n = 3;
        std::vector<std::tuple<int, int, int>> edges = {
            {0, 1, 1}, {1, 0, 1}, {1, 2, 1}, {2, 1, 1}};
        REQUIRE(run_one_test(n, edges, 0, 1));
        REQUIRE(run_one_test(n, edges, 0, 2));
    }
    {
        int n = 3;
        std::vector<std::tuple<int, int, int>> edges = {
            {0, 1, 2}, {0, 2, 1}, {2, 1, 1}};
        REQUIRE(run_one_test(n, edges, 0, 1));
    }
    {
        int n = 4;
        std::vector<std::tuple<int, int, int>> edges = {
            {0, 1, 1}, {0, 2, 4}, {1, 2, 2}, {1, 3, 6}, {2, 3, 3}};
        REQUIRE(run_one_test(n, edges, 0, 1));
        REQUIRE(run_one_test(n, edges, 0, 2));
    }
    {
        int n = 4;
        std::vector<std::tuple<int, int, int>> edges = {
            {0, 1, 1}, {1, 0, 1}, {2, 3, 1}, {3, 2, 1}};
        REQUIRE(run_one_test(n, edges, 0, 1));
    }
    {
        int n = 5;
        std::vector<std::tuple<int, int, int>> edges = {
            {0, 1, 10}, {0, 2, 3}, {1, 2, 1}, {1, 3, 2}, {2, 1, 4}, {2, 3, 8}, {2, 4, 2}, {3, 4, 7}, {4, 3, 9}};
        REQUIRE(run_one_test(n, edges, 0, 2));
    }
    {
        int n = 100;
        std::vector<std::tuple<int, int, int>> edges;
        for (int i = 0; i < n - 1; ++i) {
            edges.push_back({i, i + 1, 1});
            edges.push_back({i + 1, i, 1});
        }
        REQUIRE(run_one_test(n, edges, 0, 1));
        REQUIRE(run_one_test(n, edges, 0, 16));
    }
}

TEST_CASE("Shortest paths: light/heavy edges vs Dijkstra") {
    int n = 4;
    std::vector<std::tuple<int, int, int>> edges = {
        {0, 1, 1}, {0, 2, 10}, {1, 2, 1}, {1, 3, 10}, {2, 3, 1}};
    std::vector<int> csr_row_ptr, csr_col_idx, edge_weight;
    build_csr(n, edges, csr_row_ptr, csr_col_idx, edge_weight);

    std::vector<int> ref_dist((size_t)n);
    dijkstra_cpu(n, csr_row_ptr, csr_col_idx, edge_weight, 0, ref_dist);

    std::vector<int> dist((size_t)n);
    shortest_paths(n, csr_row_ptr.data(), csr_col_idx.data(), edge_weight.data(), 0, 2,
                   dist.data());

    for (int i = 0; i < n; ++i)
        REQUIRE(dist[(size_t)i] == ref_dist[(size_t)i]);
}
