#include "db.cuh"

#include <catch2/catch_test_macros.hpp>

#include <vector>

TEST_CASE("Database: add two columns") {
    Database db;
    db.AddTable("t", {"a", "b"},
                {
                    {1, 2, 3},
                    {4, 5, 6},
                });

    std::vector<int> r = db.Execute("t.a + t.b");
    REQUIRE(r.size() == 3);
    REQUIRE(r[0] == 5);
    REQUIRE(r[1] == 7);
    REQUIRE(r[2] == 9);
}

TEST_CASE("Database: compound expression") {
    Database db;
    db.AddTable("t", {"a", "b", "c"},
                {
                    {1, 2},
                    {3, 4},
                    {5, 6},
                });

    std::vector<int> r = db.Execute("(t.a + t.b) * t.c");
    REQUIRE(r.size() == 2);
    REQUIRE(r[0] == (1 + 3) * 5);
    REQUIRE(r[1] == (2 + 4) * 6);
}

TEST_CASE("Database: subtract and divide") {
    Database db;
    db.AddTable("t", {"x", "y"},
                {
                    {10, 20},
                    {3, 5},
                });

    std::vector<int> sub = db.Execute("t.x - t.y");
    REQUIRE(sub[0] == 7);
    REQUIRE(sub[1] == 15);

    std::vector<int> div = db.Execute("t.x / t.y");
    REQUIRE(div[0] == 3);
    REQUIRE(div[1] == 4);
}

TEST_CASE("Database: repeated Execute") {
    Database db;
    db.AddTable("t", {"a", "b"}, {{1, 2}, {3, 4}});

    for (int k = 0; k < 8; ++k) {
        std::vector<int> r = db.Execute("t.a * t.b");
        REQUIRE(r[0] == 3);
        REQUIRE(r[1] == 8);
    }
}
