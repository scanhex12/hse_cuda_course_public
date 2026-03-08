#include "tree.cuh"
#include <cmath>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/matchers/catch_matchers.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <catch2/matchers/catch_matchers_templated.hpp>

using Catch::Matchers::WithinAbs;

namespace {

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

void testCase(const std::vector<int> &parents,
              const std::vector<int> &result_heights) {
    select_least_loaded_gpu();
    auto *result = findHeights(parents.data(), parents.size());
    for (size_t i = 0; i < result_heights.size(); ++i) {
        REQUIRE(result_heights[i] == result[i]);
    }
    delete[] result;
}

} // namespace

TEST_CASE("Tree") {
    testCase({-1, 0, 1, 2, 3}, {0, 1, 2, 3, 4});
    testCase({-1, 0, 0, 0, 0}, {0, 1, 1, 1, 1});
    testCase({-1, 0, 1, 0, 3}, {0, 1, 2, 1, 2});
    testCase({-1, 0}, {0, 1});
    testCase({-1}, {0});
    testCase({-1, 0, 1, 2, 3, 4, 5, 6, 7, 8}, {0, 1, 2, 3, 4, 5, 6, 7, 8});
    testCase({-1, 0, 0, 1, 1, 2, 2}, {0, 1, 1, 2, 2, 2, 2});
    testCase({-1, 0, 0, 1, 1, 2, 2, 3}, {0, 1, 1, 2, 2, 2, 2, 3});
    testCase({-1, 0, 0, 0, 0, 0, 0}, {0, 1, 1, 1, 1, 1, 1});
    testCase({-1, 0, 0, 1, 1, 1, 2}, {0, 1, 1, 2, 2, 2, 2});
}
