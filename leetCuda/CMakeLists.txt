function(add_cuda_executable target_name source_file)
    add_executable(${target_name} ${source_file})
    set_target_properties(${target_name} PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
    )
endfunction()

add_cuda_executable(matrix_multiply_compare matrix_multiply_compare.cu)