#pragma once

#include <NvInfer.h>
#include <NvInferPlugin.h>
#include <cublasLt.h>
#include <cuda_runtime.h>

#include <cassert>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

namespace cobevt_mixed_accum
{

constexpr char const* kPluginVersion = "1";
constexpr char const* kPluginNamespace = "";
constexpr size_t kWorkspaceBytes = 64ULL * 1024ULL * 1024ULL;

template <typename T>
void writeToBuffer(char*& buffer, T const& value)
{
    std::memcpy(buffer, &value, sizeof(T));
    buffer += sizeof(T);
}

template <typename T>
T readFromBuffer(char const*& buffer)
{
    T value{};
    std::memcpy(&value, buffer, sizeof(T));
    buffer += sizeof(T);
    return value;
}

class PluginFieldReader
{
public:
    explicit PluginFieldReader(nvinfer1::PluginFieldCollection const* fields)
        : mFields(fields)
    {
    }

    template <typename T>
    T get(char const* name, T fallback) const
    {
        if (mFields == nullptr)
        {
            return fallback;
        }
        for (int32_t index = 0; index < mFields->nbFields; ++index)
        {
            auto const& field = mFields->fields[index];
            if (std::strcmp(field.name, name) == 0 && field.data != nullptr)
            {
                return *static_cast<T const*>(field.data);
            }
        }
        return fallback;
    }

private:
    nvinfer1::PluginFieldCollection const* mFields{};
};

int32_t launchMixedAccum(
    cublasLtHandle_t handle,
    void const* left,
    void const* right,
    void* output,
    void* workspace,
    size_t workspaceBytes,
    int32_t batch,
    int32_t m,
    int32_t n,
    int32_t k,
    bool transposeRight,
    float scale,
    nvinfer1::DataType outputType,
    cudaStream_t stream);

} // namespace cobevt_mixed_accum
