add_catch(test_gemm_v0 test.cu gemm.cu)
target_link_libraries(test_gemm_v0 PRIVATE hse_cuda::util)

add_catch(benchmark_gemm_v0 benchmark.cu gemm.cu)
target_link_libraries(benchmark_gemm_v0 PRIVATE hse_cuda::util)
