#pragma once

#include "plugin_common.h"

namespace cobevt_mixed_accum
{

struct MixedAccumParams
{
    int32_t family{0}; // 0 = QK, 1 = AV
    int32_t outputType{0}; // 0 = FP32, 1 = FP16
    float scale{1.0F};
};

class QKMixedAccumPlugin final : public nvinfer1::IPluginV2DynamicExt
{
public:
    explicit QKMixedAccumPlugin(MixedAccumParams params);
    QKMixedAccumPlugin(void const* data, size_t length);

    nvinfer1::IPluginV2DynamicExt* clone() const noexcept override;
    int32_t getNbOutputs() const noexcept override;
    nvinfer1::DimsExprs getOutputDimensions(int32_t outputIndex, nvinfer1::DimsExprs const* inputs,
        int32_t nbInputs, nvinfer1::IExprBuilder& exprBuilder) noexcept override;
    bool supportsFormatCombination(int32_t pos, nvinfer1::PluginTensorDesc const* inOut,
        int32_t nbInputs, int32_t nbOutputs) noexcept override;
    void configurePlugin(nvinfer1::DynamicPluginTensorDesc const* in, int32_t nbInputs,
        nvinfer1::DynamicPluginTensorDesc const* out, int32_t nbOutputs) noexcept override;
    size_t getWorkspaceSize(nvinfer1::PluginTensorDesc const* inputs, int32_t nbInputs,
        nvinfer1::PluginTensorDesc const* outputs, int32_t nbOutputs) const noexcept override;
    int32_t enqueue(nvinfer1::PluginTensorDesc const* inputDesc,
        nvinfer1::PluginTensorDesc const* outputDesc, void const* const* inputs,
        void* const* outputs, void* workspace, cudaStream_t stream) noexcept override;
    size_t getSerializationSize() const noexcept override;
    void serialize(void* buffer) const noexcept override;
    void destroy() noexcept override;
    void setPluginNamespace(char const* pluginNamespace) noexcept override;
    char const* getPluginNamespace() const noexcept override;
    char const* getPluginType() const noexcept override;
    char const* getPluginVersion() const noexcept override;
    nvinfer1::DataType getOutputDataType(int32_t index,
        nvinfer1::DataType const* inputTypes, int32_t nbInputs) const noexcept override;
    void attachToContext(cudnnContext* cudnnContext, cublasContext* cublasContext,
        nvinfer1::IGpuAllocator* gpuAllocator) noexcept override;
    void detachFromContext() noexcept override;
    int32_t initialize() noexcept override;
    void terminate() noexcept override;

private:
    bool ensureLtHandle() noexcept;
    void releaseLtHandle() noexcept;

    MixedAccumParams mParams{};
    std::string mNamespace{kPluginNamespace};
    cublasLtHandle_t mLtHandle{};
};

class QKMixedAccumPluginCreator final : public nvinfer1::IPluginCreator
{
public:
    QKMixedAccumPluginCreator();
    char const* getPluginName() const noexcept override;
    char const* getPluginVersion() const noexcept override;
    nvinfer1::PluginFieldCollection const* getFieldNames() noexcept override;
    nvinfer1::IPluginV2* createPlugin(char const* name,
        nvinfer1::PluginFieldCollection const* fc) noexcept override;
    nvinfer1::IPluginV2* deserializePlugin(char const* name, void const* serialData,
        size_t serialLength) noexcept override;
    void setPluginNamespace(char const* pluginNamespace) noexcept override;
    char const* getPluginNamespace() const noexcept override;

private:
    std::string mNamespace{kPluginNamespace};
    std::vector<nvinfer1::PluginField> mFields;
    nvinfer1::PluginFieldCollection mFC{};
};

} // namespace cobevt_mixed_accum
