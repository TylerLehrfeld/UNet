set(CMAKE_CUDA_ARCHITECTURES 89) 

add_library(cuda_lib STATIC
  add.cu
)

set_target_properties(cuda_lib PROPERTIES
  CUDA_SEPERABLE_COMPILATION ON
)

target_include_directories(cuda_lib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})



