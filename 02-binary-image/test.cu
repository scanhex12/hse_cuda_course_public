#include "binary_image.cuh"

#include <cstdio>
#include <cstdlib>
#include <exception>
#include <string>
#include <unistd.h>

#include <cuda_helpers.h>

#include <filesystem>
#include <string>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wimplicit-fallthrough"
#pragma GCC diagnostic pop

namespace {

Matrix createMatrixFromVector(const std::vector<std::vector<int>> &src) {
    auto matrix = allocMatrixHost(src.size(), src.empty() ? 0 : src[0].size());
    for (size_t i = 0; i < src.size(); ++i) {
        for (size_t j = 0; j < src[i].size(); ++j) {
            matrix.data[i * src[i].size() + j] = src[i][j];
        }
    }
    return matrix;
}

void testCase(const std::vector<std::vector<int>> &src,
              const std::vector<std::vector<int>> &expected_result) {
    auto matrix = createMatrixFromVector(src);
    auto result = solve(matrix);
    for (size_t i = 0; i < expected_result.size(); ++i) {
        for (size_t j = 0; j < expected_result[i].size(); ++j) {
            REQUIRE(expected_result[i][j] ==
                    result.data[i * expected_result[i].size() + j]);
        }
    }
    freeMatrixHost(matrix);
}

} // namespace

TEST_CASE("BinaryImage1") {
    std::vector<std::vector<int>> src = {{0, 1}, {1, 0}};

    std::vector<std::vector<int>> dst = {{1, 0}, {0, 1}};

    testCase(src, dst);
}

TEST_CASE("BinaryImage2") {
    std::vector<std::vector<int>> src = {{0, 0}, {1, 0}};

    std::vector<std::vector<int>> dst = {{1, -1}, {0, 1}};

    testCase(src, dst);
}

TEST_CASE("BinaryImage3") {
    std::vector<std::vector<int>> src = {
        {0, 0, 1}, {1, 0, 0}, {0, 0, 0}, {1, 0, 0}};

    std::vector<std::vector<int>> dst = {
        {1, 1, 0}, {0, 1, 1}, {1, -1, 2}, {0, 1, 2}};

    testCase(src, dst);
}
