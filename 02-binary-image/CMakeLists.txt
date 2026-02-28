add_catch(test_binary_image test.cu binary_image.cu)
target_link_libraries(test_binary_image PRIVATE hse_cuda::util)

add_catch(benchmark_binary_image benchmark.cu binary_image.cu)
target_link_libraries(benchmark_binary_image PRIVATE hse_cuda::util)