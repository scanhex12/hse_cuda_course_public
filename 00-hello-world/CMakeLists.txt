add_catch(test_hello_world test.cu)
target_link_libraries(test_hello_world PRIVATE hse_cuda::util)