function(add_cuda_executable target_name source_file)
    add_executable(${target_name} ${source_file} ${ARGN})
    set_target_properties(${target_name} PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
    )
endfunction()

include_directories(./include)

add_cuda_executable(main
    ./src/main.cu
    ./src/conv_cpu.cu
    ./src/conv_native.cu
    ./src/conv_constant.cu
    ./src/conv_tiled.cu
    ./src/utils.cu
)
