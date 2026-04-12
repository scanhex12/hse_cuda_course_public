if(NOT DEFINED TENSORRT_ROOT AND DEFINED ENV{TENSORRT_ROOT})
  set(TENSORRT_ROOT "$ENV{TENSORRT_ROOT}")
endif()
if(NOT TENSORRT_ROOT OR NOT EXISTS "${TENSORRT_ROOT}/include/NvInfer.h")
  message(STATUS "15-inference: TENSORRT_ROOT not set or invalid; skipping (export TENSORRT_ROOT=/path/to/TensorRT-*)")
  return()
endif()

set(TENSORRT_INCLUDE_DIR "${TENSORRT_ROOT}/include")
set(TENSORRT_LIBRARY_DIR "${TENSORRT_ROOT}/lib")

find_library(NVINFER_LIB NAMES nvinfer_10 nvinfer PATHS ${TENSORRT_LIBRARY_DIR} NO_DEFAULT_PATH)
find_library(NVINFER_PLUGIN_LIB NAMES nvinfer_plugin_10 nvinfer_plugin PATHS ${TENSORRT_LIBRARY_DIR} NO_DEFAULT_PATH)
find_library(NVONNXPARSER_LIB NAMES nvonnxparser_10 nvonnxparser PATHS ${TENSORRT_LIBRARY_DIR} NO_DEFAULT_PATH)
if(NOT NVINFER_LIB OR NOT NVINFER_PLUGIN_LIB OR NOT NVONNXPARSER_LIB)
  message(STATUS "15-inference: TensorRT libraries not found under ${TENSORRT_LIBRARY_DIR}; skipping")
  return()
endif()

message(STATUS "15-inference: TensorRT at ${TENSORRT_ROOT}")

find_package(Protobuf QUIET)
find_package(gRPC CONFIG QUIET)
find_program(GRPC_CPP_PLUGIN_EXECUTABLE NAMES grpc_cpp_plugin grpc_cpp_plugin.exe)

set(_PROTOC_CMD "")
if(TARGET protobuf::protoc)
  set(_PROTOC_CMD protobuf::protoc)
elseif(Protobuf_PROTOC_EXECUTABLE)
  set(_PROTOC_CMD "${Protobuf_PROTOC_EXECUTABLE}")
endif()

if(Protobuf_FOUND AND gRPC_FOUND AND GRPC_CPP_PLUGIN_EXECUTABLE AND _PROTOC_CMD)
  set(_PROTO "${CMAKE_CURRENT_SOURCE_DIR}/inference.proto")
  set(_OUT "${CMAKE_CURRENT_BINARY_DIR}")
  add_custom_command(
    OUTPUT ${_OUT}/inference.pb.cc ${_OUT}/inference.pb.h
           ${_OUT}/inference.grpc.pb.cc ${_OUT}/inference.grpc.pb.h
    COMMAND ${_PROTOC_CMD}
    ARGS --cpp_out=${_OUT}
         --grpc_out=${_OUT}
         -I ${CMAKE_CURRENT_SOURCE_DIR}
         --plugin=protoc-gen-grpc=${GRPC_CPP_PLUGIN_EXECUTABLE}
         ${_PROTO}
    DEPENDS ${_PROTO}
    VERBATIM
    COMMENT "Generating C++ protobuf and gRPC from inference.proto"
  )

  add_shad_executable(trt_server server.cu ${_OUT}/inference.pb.cc ${_OUT}/inference.grpc.pb.cc)
  target_link_libraries(trt_server PRIVATE
    CUDA::cudart
    ${NVINFER_LIB}
    ${NVINFER_PLUGIN_LIB}
    ${NVONNXPARSER_LIB}
    hse_cuda::util
    gRPC::grpc++
  )
  if(TARGET protobuf::libprotobuf)
    target_link_libraries(trt_server PRIVATE protobuf::libprotobuf)
  else()
    target_link_libraries(trt_server PRIVATE ${Protobuf_LIBRARIES})
    target_include_directories(trt_server PRIVATE ${Protobuf_INCLUDE_DIRS})
  endif()
  target_include_directories(trt_server PRIVATE ${TENSORRT_INCLUDE_DIR} ${_OUT})
  target_link_directories(trt_server PRIVATE ${TENSORRT_LIBRARY_DIR})
  set_property(TARGET trt_server PROPERTY CUDA_ARCHITECTURES "70-real;75-real;80-real")

  set_target_properties(trt_server PROPERTIES
    BUILD_RPATH "${TENSORRT_LIBRARY_DIR}"
    INSTALL_RPATH "${TENSORRT_LIBRARY_DIR}"
    BUILD_RPATH_USE_ORIGIN TRUE
  )
else()
  if(NOT Protobuf_FOUND)
    message(STATUS "15-inference: Protobuf not found; trt_server skipped (apt: libprotobuf-dev protobuf-compiler)")
  elseif(NOT gRPC_FOUND)
    message(STATUS "15-inference: gRPC CMake package not found; trt_server skipped (apt: libgrpc++-dev)")
  elseif(NOT GRPC_CPP_PLUGIN_EXECUTABLE)
    message(STATUS "15-inference: grpc_cpp_plugin not in PATH; trt_server skipped (apt: protobuf-compiler-grpc)")
  elseif(NOT _PROTOC_CMD)
    message(STATUS "15-inference: protoc not found; trt_server skipped")
  endif()
endif()
