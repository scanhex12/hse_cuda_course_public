add_catch(test_graphs test.cu db.cu)
target_link_libraries(test_graphs PRIVATE hse_cuda::util)

add_catch(benchmark_graphs benchmark.cu db.cu)
target_link_libraries(benchmark_graphs PRIVATE hse_cuda::util)
