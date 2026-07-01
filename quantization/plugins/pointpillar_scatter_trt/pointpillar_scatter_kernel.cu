#include "plugin_common.h"

#include <cuda_fp16.h>

namespace pointpillar_scatter_trt
{
namespace
{

template <typename T>
__global__ void zeroKernel(T* output, int64_t n)
{
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < n)
    {
        output[idx] = T(0);
    }
}

template <>
__global__ void zeroKernel<half>(half* output, int64_t n)
{
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < n)
    {
        output[idx] = __float2half(0.0F);
    }
}

__device__ bool maskValid(void const* mask, int32_t idx, nvinfer1::DataType type)
{
    if (type == nvinfer1::DataType::kINT32)
    {
        return static_cast<int32_t const*>(mask)[idx] != 0;
    }
    if (type == nvinfer1::DataType::kBOOL)
    {
        return static_cast<bool const*>(mask)[idx];
    }
    if (type == nvinfer1::DataType::kHALF)
    {
        return __half2float(static_cast<half const*>(mask)[idx]) > 0.5F;
    }
    return static_cast<float const*>(mask)[idx] > 0.5F;
}

template <typename T>
__global__ void scatterKernel(
    T const* pillarFeatures,
    int32_t const* coords,
    void const* validMask,
    T* output,
    int32_t K,
    int32_t C,
    int32_t B,
    int32_t H,
    int32_t W,
    nvinfer1::DataType maskType)
{
    int64_t linear = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t total = static_cast<int64_t>(K) * C;
    if (linear >= total)
    {
        return;
    }
    int32_t c = static_cast<int32_t>(linear % C);
    int32_t k = static_cast<int32_t>(linear / C);
    if (!maskValid(validMask, k, maskType))
    {
        return;
    }
    int32_t b = coords[k * 4 + 0];
    int32_t z = coords[k * 4 + 1];
    int32_t y = coords[k * 4 + 2];
    int32_t x = coords[k * 4 + 3];
    if (z != 0 || b < 0 || b >= B || y < 0 || y >= H || x < 0 || x >= W)
    {
        return;
    }
    int64_t outIndex = (((static_cast<int64_t>(b) * C + c) * H + y) * W + x);
    output[outIndex] = pillarFeatures[static_cast<int64_t>(k) * C + c];
}

template <typename T>
void launchTyped(
    void const* pillarFeatures,
    void const* voxelCoords,
    void const* validVoxelMask,
    void* output,
    int32_t K,
    int32_t C,
    int32_t B,
    int32_t H,
    int32_t W,
    nvinfer1::DataType maskType,
    cudaStream_t stream)
{
    int threads = 256;
    int64_t outElems = static_cast<int64_t>(B) * C * H * W;
    int blocks = static_cast<int>((outElems + threads - 1) / threads);
    zeroKernel<T><<<blocks, threads, 0, stream>>>(static_cast<T*>(output), outElems);

    int64_t scatterElems = static_cast<int64_t>(K) * C;
    blocks = static_cast<int>((scatterElems + threads - 1) / threads);
    scatterKernel<T><<<blocks, threads, 0, stream>>>(
        static_cast<T const*>(pillarFeatures),
        static_cast<int32_t const*>(voxelCoords),
        validVoxelMask,
        static_cast<T*>(output),
        K,
        C,
        B,
        H,
        W,
        maskType);
}

} // namespace

void launchPointPillarScatter(
    void const* pillarFeatures,
    void const* voxelCoords,
    void const* validVoxelMask,
    void* output,
    int32_t K,
    int32_t C,
    int32_t B,
    int32_t H,
    int32_t W,
    nvinfer1::DataType featureType,
    nvinfer1::DataType,
    nvinfer1::DataType maskType,
    cudaStream_t stream)
{
    if (featureType == nvinfer1::DataType::kHALF)
    {
        launchTyped<half>(pillarFeatures, voxelCoords, validVoxelMask, output, K, C, B, H, W, maskType, stream);
    }
    else
    {
        launchTyped<float>(pillarFeatures, voxelCoords, validVoxelMask, output, K, C, B, H, W, maskType, stream);
    }
}

} // namespace pointpillar_scatter_trt
