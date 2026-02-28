add_catch(test_vector test.cu)
target_link_libraries(test_vector PRIVATE hse_cuda::util)

add_catch(benchmark_vector benchmark.cu)
target_link_libraries(benchmark_vector PRIVATE hse_cuda::util)