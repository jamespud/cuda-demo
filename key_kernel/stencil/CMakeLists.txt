cmake_minimum_required(VERSION 3.18)
project(stencil LANGUAGES CXX CUDA)

function(add_cuda_executable target_name source_file)
    add_executable(${target_name} ${source_file} ${ARGN})
    set_target_properties(${target_name} PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
    )
endfunction()

include_directories(./include)

add_cuda_executable(stencil
    ./src/main.cu
    ./src/cpu_stencil.cpp
    ./src/naive_stencil.cu
    ./src/shared_stencil.cu
    ./src/coarsened_stencil.cu
)
