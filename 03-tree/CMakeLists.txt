add_catch(test_tree test.cu tree.cu)
target_link_libraries(test_tree PRIVATE hse_cuda::util)

add_catch(benchmark_tree benchmark.cu tree.cu)
target_link_libraries(benchmark_tree PRIVATE hse_cuda::util)