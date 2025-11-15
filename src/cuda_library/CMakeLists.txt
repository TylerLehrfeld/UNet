set(CMAKE_CUDA_ARCHITECTURES 89) 

add_library(cuda_lib SHARED
  cuda_ops.cu
)

set_target_properties(cuda_lib PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
)
target_link_libraries(cuda_lib PUBLIC CUDA::cudart)
target_include_directories(cuda_lib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})



