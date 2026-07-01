#pragma once

#include <NvInfer.h>
#include <cuda_runtime_api.h>

#include <cassert>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

namespace pointpillar_scatter_trt
{

constexpr char const* kPluginNamespace = "";
constexpr char const* kPluginVersion = "1";

#define PPS_CUDA_CHECK(expr)                                                                                          \
    do                                                                                                                \
    {                                                                                                                 \
        cudaError_t _err = (expr);                                                                                    \
        if (_err != cudaSuccess)                                                                                      \
        {                                                                                                             \
            std::cerr << "[PointPillarScatterTRT][CUDA] " << cudaGetErrorString(_err) << " at " << __FILE__ << ":" \
                      << __LINE__ << std::endl;                                                                       \
            return static_cast<int>(_err);                                                                            \
        }                                                                                                             \
    } while (0)

template <typename T>
inline void writeToBuffer(char*& buffer, T const& value)
{
    std::memcpy(buffer, &value, sizeof(T));
    buffer += sizeof(T);
}

template <typename T>
inline T readFromBuffer(char const*& buffer)
{
    T value{};
    std::memcpy(&value, buffer, sizeof(T));
    buffer += sizeof(T);
    return value;
}

inline bool isFp(nvinfer1::DataType type)
{
    return type == nvinfer1::DataType::kFLOAT || type == nvinfer1::DataType::kHALF;
}

struct PluginFieldReader
{
    nvinfer1::PluginFieldCollection const* fc{nullptr};

    explicit PluginFieldReader(nvinfer1::PluginFieldCollection const* fields)
        : fc(fields)
    {
    }

    template <typename T>
    T get(char const* name, T defaultValue) const
    {
        if (fc == nullptr)
        {
            return defaultValue;
        }
        for (int32_t i = 0; i < fc->nbFields; ++i)
        {
            auto const& f = fc->fields[i];
            if (std::strcmp(f.name, name) == 0 && f.data != nullptr)
            {
                return *static_cast<T const*>(f.data);
            }
        }
        return defaultValue;
    }
};

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
    nvinfer1::DataType coordType,
    nvinfer1::DataType maskType,
    cudaStream_t stream);

} // namespace pointpillar_scatter_trt
