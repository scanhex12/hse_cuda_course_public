add_catch(test_reduce_norm test.cu norm.cu)
target_link_libraries(test_reduce_norm PRIVATE hse_cuda::util)

add_catch(benchmark_reduce_norm benchmark.cu norm.cu)
target_link_libraries(benchmark_reduce_norm PRIVATE hse_cuda::util)