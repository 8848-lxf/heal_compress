cmake_minimum_required(VERSION 3.18)

project(pointpillar_scatter_trt LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT DEFINED TRT_ROOT)
    if(DEFINED ENV{TRT_ROOT})
        set(TRT_ROOT $ENV{TRT_ROOT})
    else()
        message(FATAL_ERROR "TRT_ROOT is required; pass -DTRT_ROOT=<TensorRT root> or set the TRT_ROOT environment variable")
    endif()
endif()

if(NOT CMAKE_CUDA_ARCHITECTURES)
    if(DEFINED ENV{CMAKE_CUDA_ARCHITECTURES})
        set(CMAKE_CUDA_ARCHITECTURES $ENV{CMAKE_CUDA_ARCHITECTURES})
    else()
        set(CMAKE_CUDA_ARCHITECTURES 89)
    endif()
endif()

find_package(CUDAToolkit REQUIRED)

find_path(TENSORRT_INCLUDE_DIR NvInfer.h HINTS ${TRT_ROOT}/include REQUIRED)
find_library(TENSORRT_LIB nvinfer HINTS
    ${TRT_ROOT}/lib
    ${TRT_ROOT}/lib64
    ${TRT_ROOT}/targets/x86_64-linux-gnu/lib
    REQUIRED)
find_library(TENSORRT_PLUGIN_LIB nvinfer_plugin HINTS
    ${TRT_ROOT}/lib
    ${TRT_ROOT}/lib64
    ${TRT_ROOT}/targets/x86_64-linux-gnu/lib
    REQUIRED)

add_library(pointpillar_scatter_trt SHARED
    pointpillar_scatter_plugin.cpp
    pointpillar_scatter_kernel.cu
    plugin_register.cpp)

target_include_directories(pointpillar_scatter_trt PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${TENSORRT_INCLUDE_DIR}
    ${CUDAToolkit_INCLUDE_DIRS})

target_link_libraries(pointpillar_scatter_trt PRIVATE
    ${TENSORRT_LIB}
    ${TENSORRT_PLUGIN_LIB}
    CUDA::cudart)

target_compile_options(pointpillar_scatter_trt PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -Wno-deprecated-declarations>
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr --use_fast_math>)

set_target_properties(pointpillar_scatter_trt PROPERTIES
    OUTPUT_NAME "pointpillar_scatter_trt"
    CUDA_SEPARABLE_COMPILATION ON)
