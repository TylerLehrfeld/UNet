set(CMAKE_CUDA_ARCHITECTURES 89) 

add_library(cuda_lib SHARED
  cuda_ops.cu
  ./max_pool/max_pool.cu
  ./2Dconvolution/convolution.cu
  ./upsample/upsample.cu
  ./batch_norm/batch_norm.cu
  ./attention/attention.cu
  cuda_lib.h
)


set_target_properties(cuda_lib PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
)
target_link_libraries(cuda_lib PUBLIC CUDA::cudart)
target_include_directories(cuda_lib PUBLIC 
  ${CMAKE_CURRENT_SOURCE_DIR}
  ./test/
  ./max_pool/
  ./2Dconvolution/
  ./upsample/
  ./batch_norm/
  ./attention/
)

add_library(cuda_manager_lib SHARED
  cuda_manager_ops.cu
)

set_target_properties(cuda_manager_lib PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
)

target_link_libraries(cuda_manager_lib PUBLIC CUDA::cudart)

target_include_directories(cuda_manager_lib PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}
)
