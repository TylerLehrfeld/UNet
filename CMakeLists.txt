# define the project parameters
cmake_minimum_required(VERSION 3.4...3.18)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_CUDA_ARCHITECTURES 89) 
set(CMAKE_BUILD_TYPE Debug)
project(neural_net LANGUAGES C CXX CUDA)
find_package(CUDAToolkit REQUIRED)
add_subdirectory(extern/pybind11)
add_subdirectory(src/cuda_library)
add_subdirectory(src/neural_net_implementation)
include_directories(
    ${CMAKE_SOURCE_DIR}/src/neural_net_implementation
    ${CMAKE_SOURCE_DIR}/src/cuda_library
    ${CMAKE_SOURCE_DIR}/src/cuda_library/test
)

# define NN implementation module
pybind11_add_module(neural_net src/bindings/net.cpp)
target_link_libraries(neural_net PRIVATE 
  cuda_lib
)
target_compile_definitions(neural_net
                           PRIVATE VERSION_INFO=${EXAMPLE_VERSION_INFO})



execute_process(COMMAND ${CMAKE_COMMAND} -E create_symlink
    ${CMAKE_BINARY_DIR}/compile_commands.json
    ${CMAKE_SOURCE_DIR}/compile_commands.json
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
)
