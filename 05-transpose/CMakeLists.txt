add_catch(test_transpose test.cu transpose.cu)
target_link_libraries(test_transpose PRIVATE hse_cuda::util)

add_catch(benchmark_transpose benchmark.cu transpose.cu)
target_link_libraries(benchmark_transpose PRIVATE hse_cuda::util)
