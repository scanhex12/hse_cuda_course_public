add_catch(test_fibonacci test.cu fibonacci.cuh)
target_link_libraries(test_fibonacci PRIVATE hse_cuda::util)
