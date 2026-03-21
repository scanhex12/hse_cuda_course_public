add_catch(test_lu_decomposition test.cu lu_decomposition.cuh lu_decomposition.cu)
target_link_libraries(test_lu_decomposition PRIVATE hse_cuda::util)

add_catch(benchmark_lu_decomposition benchmark.cu lu_decomposition.cuh lu_decomposition.cu)
target_link_libraries(benchmark_lu_decomposition PRIVATE hse_cuda::util)

set_property(TARGET test_lu_decomposition PROPERTY CUDA_ARCHITECTURES "75-real;80-real")
set_property(TARGET benchmark_lu_decomposition PROPERTY CUDA_ARCHITECTURES "75-real;80-real")
