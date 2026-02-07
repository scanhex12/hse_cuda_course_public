#include "hello_world.cuh"

#include <cstdio>
#include <cstdlib>
#include <unistd.h>  // dup

#include <catch2/catch_test_macros.hpp>

namespace
{

TEST_CASE("HelloWorldOnGPU")
{
    // Nasty hack to capture CUDA's printf output
    char buffer[1024] = "";
    int backup_stdout = dup(fileno(stdout));
    REQUIRE(backup_stdout > 0);

    REQUIRE(freopen("/dev/null", "w", stdout) != nullptr);
    setbuf(stdout, buffer);
    fflush(stdout);

    CallHelloWorld();

    setbuf(stdout, nullptr);
    fclose(stdout);
    FILE *fp = fdopen(backup_stdout, "w");

    fclose(stdout);
    *stdout = *fp;

    REQUIRE(buffer == std::string{"Hello, world!"});
}

}
