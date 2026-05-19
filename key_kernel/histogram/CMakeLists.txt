function(add_cuda_executable target_name source_file)
    add_executable(${target_name} ${source_file} ${ARGN})
    set_target_properties(${target_name} PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
    )
endfunction()

include_directories(./include)

add_cuda_executable(histogram
    ./src/main.cu
    ./src/cpu_histogram.cu
    ./src/native_histogram.cu
    ./src/shared_histogram.cu
    ./src/coarsened_histogram.cu
    ./src/registered_histogram.cu
)
