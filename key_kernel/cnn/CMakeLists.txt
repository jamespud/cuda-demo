function(add_cuda_executable target_name source_file)
    add_executable(${target_name} ${source_file} ${ARGN})
    set_target_properties(${target_name} PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
    )
endfunction()

include_directories(./include)

find_path(CUDNN_INCLUDE_DIR
    NAMES cudnn.h
    HINTS
        /usr/include
        /usr/include/x86_64-linux-gnu
        /usr/local/cuda/include
    PATH_SUFFIXES
        x86_64-linux-gnu
)

find_library(CUDNN_LIBRARY
    NAMES cudnn libcudnn
    HINTS
        /usr/lib/x86_64-linux-gnu
        /usr/local/cuda/lib64
        /usr/local/cuda/targets/x86_64-linux/lib
)

if(NOT CUDNN_INCLUDE_DIR OR NOT CUDNN_LIBRARY)
    message(FATAL_ERROR
        "cuDNN not found. Install libcudnn9-dev-cuda-13 or set the include/library paths manually.")
endif()

add_cuda_executable(cnn
    ./src/main.cu
    ./src/cpu_cnn.cu
    ./src/cnn_native.cu
    ./src/cnn_unroll.cu
    ./src/cudnn_cnn.cu
)

target_include_directories(cnn PRIVATE ${CUDNN_INCLUDE_DIR})
target_link_libraries(cnn PRIVATE ${CUDNN_LIBRARY})
