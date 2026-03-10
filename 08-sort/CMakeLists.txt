add_catch(test_sort test.cu sort.cu sort.cuh)
target_link_libraries(test_sort PRIVATE hse_cuda::util)

add_catch(benchmark_sort benchmark.cu sort.cu sort.cuh)
target_link_libraries(benchmark_sort PRIVATE hse_cuda::util)
