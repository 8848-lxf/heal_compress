cmake_minimum_required(VERSION 3.18)
project(cobevt_qk_mixed_accum LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT DEFINED TRT_ROOT)
    message(FATAL_ERROR "TRT_ROOT must identify the audited TensorRT installation")
endif()
if(NOT CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 89)
endif()

find_package(CUDAToolkit REQUIRED)
find_path(TENSORRT_INCLUDE_DIR NvInfer.h HINTS ${TRT_ROOT}/include REQUIRED)
find_library(TENSORRT_LIB nvinfer HINTS ${TRT_ROOT}/lib ${TRT_ROOT}/targets/x86_64-linux-gnu/lib REQUIRED)
find_library(TENSORRT_PLUGIN_LIB nvinfer_plugin HINTS ${TRT_ROOT}/lib ${TRT_ROOT}/targets/x86_64-linux-gnu/lib REQUIRED)

add_library(cobevt_qk_mixed_accum SHARED
    qk_mixed_accum_plugin.cpp
    qk_mixed_accum_kernel.cu
    plugin_register.cpp)
target_include_directories(cobevt_qk_mixed_accum PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR} ${TENSORRT_INCLUDE_DIR} ${CUDAToolkit_INCLUDE_DIRS})
target_link_libraries(cobevt_qk_mixed_accum PRIVATE
    ${TENSORRT_LIB} ${TENSORRT_PLUGIN_LIB} CUDA::cudart CUDA::cublasLt)
target_compile_options(cobevt_qk_mixed_accum PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -Wno-deprecated-declarations>
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
set_target_properties(cobevt_qk_mixed_accum PROPERTIES
    OUTPUT_NAME "cobevt_qk_mixed_accum" CUDA_SEPARABLE_COMPILATION ON)
