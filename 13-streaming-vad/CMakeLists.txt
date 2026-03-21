add_catch(test_streaming_vad test.cu streaming_vad.cuh streaming_vad.cu)
target_link_libraries(test_streaming_vad PRIVATE hse_cuda::util)

add_catch(benchmark_streaming_vad benchmark.cu streaming_vad.cuh streaming_vad.cu)
target_link_libraries(benchmark_streaming_vad PRIVATE hse_cuda::util)
