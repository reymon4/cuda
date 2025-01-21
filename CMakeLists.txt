cmake_minimum_required(VERSION 3.29)
project(cuda CUDA)

set(CMAKE_CUDA_STANDARD 17)
enable_language(CXX CUDA)
add_executable(cuda kernel.cu
        main.cpp)

target_include_directories(cuda PUBLIC ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
set_target_properties(cuda PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

