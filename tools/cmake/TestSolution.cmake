option(GRADER "Building on CI" OFF)

function(add_shad_executable NAME)
  add_executable(${NAME} ${ARGN})
  target_compile_options(${NAME} PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:--compiler-options -Werror,-Wall,-Wextra>"
                                         "$<$<COMPILE_LANGUAGE:CXX>:-Werror;-Wall;-Wextra>")
endfunction()

function(add_shad_library NAME)
  add_library(${NAME} ${ARGN})
  target_compile_options(${NAME} PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:--compiler-options -Werror,-Wall,-Wextra>"
                                         "$<$<COMPILE_LANGUAGE:CXX>:-Werror;-Wall;-Wextra>")
endfunction()

function(add_shad_shared_library NAME)
  add_library(${NAME} SHARED ${ARGN})
  target_compile_options(${NAME} PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:--compiler-options -Werror,-Wall,-Wextra>"
                                         "$<$<COMPILE_LANGUAGE:CXX>:-Werror;-Wall;-Wextra>")
endfunction()

function(add_catch TARGET)
  add_shad_executable(${TARGET} ${ARGN})
  target_link_libraries(${TARGET} PRIVATE Catch2::Catch2WithMain)
endfunction()
