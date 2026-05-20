function(add_cuda_executable target_name source_file)
    add_executable(${target_name} ${source_file} ${ARGN})
    set_target_properties(${target_name} PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
    )
endfunction()

include_directories(./include)

add_cuda_executable(reduction
    ./src/main.cu
    ./src/cpu_reduction.cu
    ./src/native_reduction.cu
    ./src/shared_reduction.cu
    ./src/coarsened_reduction.cu
)
