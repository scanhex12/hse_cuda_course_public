add_catch(test_graphs test.cu)
add_catch(benchmark_graphs benchmark.cu)

target_link_libraries(benchmark_graphs PRIVATE hse_cuda::util)
