add_catch(test_db test.cu db.cu)
target_link_libraries(test_db PRIVATE hse_cuda::util)

add_catch(benchmark_db benchmark.cu db.cu)
target_link_libraries(benchmark_db PRIVATE hse_cuda::util)
