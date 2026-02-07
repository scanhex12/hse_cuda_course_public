cmake_minimum_required(VERSION 3.22)
project(hse-cuda LANGUAGES C CXX CUDA)

set(CMAKE_CXX_STANDARD             20)
set(CMAKE_MODULE_PATH              "${CMAKE_SOURCE_DIR}/tools/cmake")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
set(CMAKE_EXPORT_COMPILE_COMMANDS  ON)

find_package(Catch REQUIRED)
find_package(CUDAToolkit REQUIRED)

set(Clang_DIR "/usr/lib/llvm-16/lib/cmake/clang")
set(LLVM_DIR "/usr/lib/llvm-16/lib/cmake/llvm")
find_package(Clang CONFIG)

include(tools/cmake/TestSolution.cmake)
include(tools/cmake/BuildFlags.cmake)
add_subdirectory(tools)

function(add_if_exists name)
  if (EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${name})
    add_subdirectory(${name})
  endif()
endfunction()

add_if_exists(00-hello-world)

if (EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/private)
  add_subdirectory(private)
endif()
