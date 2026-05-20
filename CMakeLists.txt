cmake_minimum_required(VERSION 3.18)

set(CMAKE_CUDA_ARCHITECTURES 86 CACHE STRING "CUDA architecture(s)")

project(cuda_demo LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

add_subdirectory(leetCuda)

add_subdirectory(exercise/ch3)

add_subdirectory(key_kernel/convolution)
add_subdirectory(key_kernel/stencil)
add_subdirectory(key_kernel/histogram)
add_subdirectory(key_kernel/reduction)
