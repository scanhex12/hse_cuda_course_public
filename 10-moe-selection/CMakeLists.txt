add_catch(test_moe_selection test.cu moe.cu)
target_link_libraries(test_moe_selection PRIVATE hse_cuda::util)

add_catch(benchmark_moe_selection benchmark.cu moe.cu)
target_link_libraries(benchmark_moe_selection PRIVATE hse_cuda::util)
