add_shad_library(util include/cuda_helpers.h src/cuda_helpers.cpp)
add_library(hse_cuda::util ALIAS util)
target_link_libraries(util PUBLIC CUDA::cudart) 
target_include_directories(
    util 
    PUBLIC
    $<INSTALL_INTERFACE:include>
	$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)
