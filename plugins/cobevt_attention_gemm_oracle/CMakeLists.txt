cmake_minimum_required(VERSION 3.18)
project(cobevt_attention_gemm_oracle LANGUAGES CXX CUDA)

find_package(CUDAToolkit REQUIRED)
add_executable(cobevt_attention_gemm_oracle oracle.cu)
target_compile_features(cobevt_attention_gemm_oracle PRIVATE cxx_std_17)
target_link_libraries(cobevt_attention_gemm_oracle PRIVATE CUDA::cudart CUDA::cublasLt)
set_target_properties(
  cobevt_attention_gemm_oracle
  PROPERTIES CUDA_ARCHITECTURES 89 CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED YES
)
