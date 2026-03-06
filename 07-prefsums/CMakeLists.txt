add_catch(test_prefsums test.cu prefsum.cuh)
target_link_libraries(test_prefsums PRIVATE hse_cuda::util)

add_catch(benchmark_prefsums benchmark.cu prefsum.cuh)
target_link_libraries(benchmark_prefsums PRIVATE hse_cuda::util)