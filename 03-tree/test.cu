#include <cmath>
#include <vector>
#include "tree.cuh"

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <catch2/matchers/catch_matchers_templated.hpp>
#include <catch2/matchers/catch_matchers.hpp>

using Catch::Matchers::WithinAbs;

namespace
{

void testCase(const std::vector<int> & parents, const std::vector<int> & result_heights)
{
    auto * result = findHeights(parents.data(), parents.size());
    for (size_t i = 0; i < result_heights.size(); ++i)
    {
        REQUIRE(result_heights[i] == result[i]);
    }
    delete[] result;
}

}

TEST_CASE("Tree")
{
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
