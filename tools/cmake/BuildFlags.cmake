set(CMAKE_CXX_FLAGS_ASAN "-g -fsanitize=address,undefined -fno-sanitize-recover=all"
  CACHE STRING "Compiler flags in asan build"
  FORCE)

set(CMAKE_CXX_FLAGS_TSAN "-g -fsanitize=thread -fno-sanitize-recover=all"
  CACHE STRING "Compiler flags in tsan build"
  FORCE)
