#include <cmath>
#include <vector>
#include "vector.cuh"

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <catch2/matchers/catch_matchers_templated.hpp>
#include <catch2/matchers/catch_matchers.hpp>

using Catch::Matchers::WithinAbs;

TEST_CASE("InitialState") {
    CudaVector<float> v;
    REQUIRE(v.size() == 0);
    REQUIRE(v.capacity() == 0);
}

TEST_CASE("ReserveIncreasesCapacity") {
    CudaVector<int> v;
    v.reserve(10);
    REQUIRE(v.capacity() == 10);
    REQUIRE(v.size() == 0);
}

TEST_CASE("PushBackGrowsSize") {
    CudaVector<float> v;
    v.push_back(1.5f);
    v.push_back(2.5f);
    REQUIRE(v.size() == 2);
    REQUIRE(v.capacity() == 2);
    REQUIRE(std::fabs(v.get(0) - 1.5f) < 1e-5f);
    REQUIRE(std::fabs(v.get(1) - 2.5f) < 1e-5f);
}

TEST_CASE("PushBackTriggersRealloc") {
    CudaVector<int> v;
    for (int i = 0; i < 16; ++i) {
        v.push_back(i);
    }
    REQUIRE(v.size() == 16);
    REQUIRE(v.capacity() == 16);
    for (int i = 0; i < 16; ++i) {
        REQUIRE(v.get(i) == i);
    }
}

TEST_CASE("SetAndGet") {
    CudaVector<double> v(5);
    for (size_t i = 0; i < 5; ++i) {
        v.set(i, i * 10.0);
    }
    for (size_t i = 0; i < 5; ++i) {
        REQUIRE_THAT(v.get(i), WithinAbs(i * 10.0, 1e-9));
    }
}

TEST_CASE("ProxyAssignmentAndRead") {
    CudaVector<float> v;
    v.push_back(10.0f);
    v.push_back(20.0f);

    v[1] = 99.5f;
    REQUIRE(std::fabs(v[1] - 99.5f) < 1e-5f);
}

TEST_CASE("ProxyChainedOps") {
    CudaVector<int> v;
    for (int i = 0; i < 4; ++i) v.push_back(i);

    int x = v[2];
    REQUIRE(x == 2);

    v[2] = 777;
    REQUIRE(v[2] == 777);
}

TEST_CASE("OutOfRangeGetThrows") {
    CudaVector<int> v;
    v.reserve(2);
    v.push_back(1);

    REQUIRE_THROWS_AS(v.get(5), std::out_of_range);
}

TEST_CASE("OutOfRangeSetThrows") {
    CudaVector<int> v(3);
    REQUIRE_THROWS_AS(v.set(10, 42), std::out_of_range);
}

TEST_CASE("BracketOutOfRangeThrows") {
    CudaVector<float> v;
    v.push_back(1.0f);
    REQUIRE_THROWS_AS(v[3] = 2.0f, std::out_of_range);
}

TEST_CASE("ElementwiseAdd") {
    CudaVector<int> a, b;
    for (int i = 0; i < 5; ++i) {
        a.push_back(i);
        b.push_back(i * 2);
    }
    auto c = a + b;
    REQUIRE(c.size() == 5);
    for (size_t i = 0; i < 5; ++i) {
        REQUIRE(c.get(i) == a.get(i) + b.get(i));
    }
}

TEST_CASE("ElementwiseSub") {
    CudaVector<int> a, b;
    for (int i = 0; i < 5; ++i) {
        a.push_back(i * 3);
        b.push_back(i * 5);
    }
    auto c = b - a;
    REQUIRE(c.size() == 5);
    for (size_t i = 0; i < 5; ++i) {
        REQUIRE(c.get(i) == b.get(i) - a.get(i));
    }
}

TEST_CASE("ElementwiseMul") {
    CudaVector<float> a, b;
    for (int i = 1; i <= 6; ++i) {
        a.push_back(float(i));
        b.push_back(float(i * i));
    }
    auto c = a * b;
    REQUIRE(c.size() == 6);
    for (size_t i = 0; i < 6; ++i) {
        REQUIRE_THAT(c.get(i), WithinAbs(a.get(i) * b.get(i), 1e-6f));
    }
}

TEST_CASE("ElementwiseDiv") {
    CudaVector<float> a, b;
    for (int i = 1; i <= 4; ++i) {
        a.push_back(float(i * 2));
        b.push_back(float(i));
    }
    auto c = a / a;
    REQUIRE(c.size() == 4);
    for (size_t i = 0; i < 4; ++i) {
        REQUIRE_THAT(c.get(i), WithinAbs(1.0f, 1e-6f));
    }
}

TEST_CASE("CopyFromHost") {
    std::vector<int> host{1, 2, 3, 4, 5};
    CudaVector<int> v;
    v.copy_from_host(host.data(), host.size());
    REQUIRE(v.size() == host.size());
    REQUIRE(v.capacity() >= host.size());
    for (size_t i = 0; i < host.size(); ++i) {
        REQUIRE(v.get(i) == host[i]);
    }
}

TEST_CASE("CopyToHost") {
    CudaVector<float> v;
    for (int i = 0; i < 7; ++i) v.push_back(float(i + 1));

    std::vector<float> host(7);
    v.copy_to_host(host.data());
    for (size_t i = 0; i < 7; ++i) {
        REQUIRE_THAT(host[i], WithinAbs(v.get(i), 1e-6f));
    }
}


TEST_CASE("MoveConstructor") {
    CudaVector<int> a;
    for (int i = 0; i < 5; ++i) a.push_back(i);

    int* original_ptr = a.data();
    size_t original_cap = a.capacity();

    CudaVector<int> b(std::move(a));

    REQUIRE(b.size() == 5);
    REQUIRE(b.data() == original_ptr);
    REQUIRE(b.capacity() == original_cap);

    REQUIRE(a.size() == 0);
    REQUIRE(a.capacity() == 0);
    REQUIRE(a.data() == nullptr);
}

TEST_CASE("MoveAssignment") {
    CudaVector<int> a;
    for (int i = 0; i < 9; ++i) a.push_back(i * 10);

    int* ptr = a.data();
    size_t size = a.size();
    size_t cap = a.capacity();

    CudaVector<int> b;
    b = std::move(a);

    REQUIRE(b.data() == ptr);
    REQUIRE(b.size() == size);
    REQUIRE(b.capacity() == cap);

    REQUIRE(a.data() == nullptr);
    REQUIRE(a.size() == 0);
    REQUIRE(a.capacity() == 0);
}

TEST_CASE("ElementwiseSizeMismatchThrows") {
    CudaVector<int> a, b;
    a.push_back(1);
    a.push_back(2);
    b.push_back(10);

    REQUIRE_THROWS_AS(a + b, std::runtime_error);
    REQUIRE_THROWS_AS(a - b, std::runtime_error);
    REQUIRE_THROWS_AS(a * b, std::runtime_error);
    REQUIRE_THROWS_AS(a / b, std::runtime_error);
}
