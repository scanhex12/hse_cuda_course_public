add_library(stb_image INTERFACE include/stb_image.h)
add_library(stb::stb_image ALIAS stb_image)
target_include_directories(
    stb_image 
    INTERFACE
    $<INSTALL_INTERFACE:include>
	$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)
