#include "fibonacci.cuh"

#include <cstdio>
#include <unistd.h>

#include <algorithm>
#include <sstream>
#include <string>
#include <vector>

#include <catch2/catch_test_macros.hpp>

namespace {

std::vector<int> reference_fibonacci(int count) {
  std::vector<int> out((size_t)count);
  for (int i = 0; i < count; ++i)
    out[(size_t)i] = fib(i);
  return out;
}

std::string capture_print_fibonacci(int count, int grid_dim, int block_dim) {
  char buffer[8192] = "";
  int backup_stdout = dup(fileno(stdout));
  REQUIRE(backup_stdout > 0);

  REQUIRE(freopen("/dev/null", "w", stdout) != nullptr);
  setbuf(stdout, buffer);
  fflush(stdout);

  run_print_fibonacci(count, grid_dim, block_dim);

  setbuf(stdout, nullptr);
  fclose(stdout);
  FILE* fp = fdopen(backup_stdout, "w");
  fclose(stdout);
  *stdout = *fp;

  return std::string(buffer);
}

std::vector<int> parse_fibonacci_output(const std::string& output) {
  std::vector<int> nums;
  std::istringstream iss(output);
  int v;
  while (iss >> v)
    nums.push_back(v);
  std::sort(nums.begin(), nums.end());
  return nums;
}

void check_print_fibonacci_with_config(int count, int grid_dim, int block_dim) {
  auto ref = reference_fibonacci(count);
  std::sort(ref.begin(), ref.end());

  std::string output = capture_print_fibonacci(count, grid_dim, block_dim);
  std::vector<int> got = parse_fibonacci_output(output);

  REQUIRE(got.size() == static_cast<size_t>(count));
  for (int i = 0; i < count; ++i)
    REQUIRE(got[(size_t)i] == ref[(size_t)i]);
}

TEST_CASE("FibHostValues") {
  REQUIRE(fib(0) == 0);
  REQUIRE(fib(1) == 1);
  REQUIRE(fib(2) == 1);
  REQUIRE(fib(3) == 2);
  REQUIRE(fib(10) == 55);
  REQUIRE(fib(15) == 610);
}

TEST_CASE("PrintFibonacciSingleThread") {
  std::string output = capture_print_fibonacci(8, 1, 1);
  REQUIRE(output == "0\n1\n1\n2\n3\n5\n8\n13\n");
}

TEST_CASE("PrintFibonacciOneBlock") {
  check_print_fibonacci_with_config(16, 1, 4);
  check_print_fibonacci_with_config(32, 1, 8);
}

TEST_CASE("PrintFibonacciMultipleBlocks") {
  check_print_fibonacci_with_config(64, 4, 16);
  check_print_fibonacci_with_config(50, 5, 10);
}

TEST_CASE("PrintFibonacciVariousGridsAndBlocks") {
  const int count = 20;
  check_print_fibonacci_with_config(count, 1, 1);
  check_print_fibonacci_with_config(count, 1, 8);
  check_print_fibonacci_with_config(count, 2, 4);
  check_print_fibonacci_with_config(count, 5, 4);
}

TEST_CASE("PrintFibonacciSmallN") {
  std::string out1 = capture_print_fibonacci(1, 1, 1);
  REQUIRE(parse_fibonacci_output(out1) == std::vector<int>{0});

  std::string out2 = capture_print_fibonacci(2, 1, 1);
  REQUIRE(parse_fibonacci_output(out2) == std::vector<int>({0, 1}));
}

}  // namespace
