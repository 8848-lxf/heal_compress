#include "qk_mixed_accum_plugin.h"

#include <iostream>

using namespace nvinfer1;

namespace cobevt_mixed_accum
{
namespace
{
char const* kName = "QKMixedAccumPlugin";
}

QKMixedAccumPlugin::QKMixedAccumPlugin(MixedAccumParams params)
    : mParams(params)
{
}

QKMixedAccumPlugin::QKMixedAccumPlugin(void const* data, size_t length)
{
    assert(length == sizeof(MixedAccumParams));
    char const* cursor = static_cast<char const*>(data);
    mParams = readFromBuffer<MixedAccumParams>(cursor);
}

IPluginV2DynamicExt* QKMixedAccumPlugin::clone() const noexcept
{
    auto* plugin = new QKMixedAccumPlugin(mParams);
    plugin->setPluginNamespace(mNamespace.c_str());
    return plugin;
}

int32_t QKMixedAccumPlugin::getNbOutputs() const noexcept { return 1; }

DimsExprs QKMixedAccumPlugin::getOutputDimensions(
    int32_t, DimsExprs const* inputs, int32_t nbInputs, IExprBuilder&) noexcept
{
    assert(nbInputs == 2);
    DimsExprs out{};
    out.nbDims = 4;
    out.d[0] = inputs[0].d[0];
    out.d[1] = inputs[0].d[1];
    out.d[2] = inputs[0].d[2];
    out.d[3] = (mParams.family == 0) ? inputs[1].d[2] : inputs[1].d[3];
    return out;
}

bool QKMixedAccumPlugin::supportsFormatCombination(
    int32_t pos, PluginTensorDesc const* inOut, int32_t nbInputs, int32_t nbOutputs) noexcept
{
    assert(nbInputs == 2 && nbOutputs == 1);
    if (pos < 2)
    {
        return inOut[pos].type == DataType::kHALF && inOut[pos].format == TensorFormat::kLINEAR;
    }
    auto outputType = mParams.outputType == 0 ? DataType::kFLOAT : DataType::kHALF;
    return inOut[pos].type == outputType && inOut[pos].format == TensorFormat::kLINEAR;
}

void QKMixedAccumPlugin::configurePlugin(DynamicPluginTensorDesc const* in, int32_t nbInputs,
    DynamicPluginTensorDesc const*, int32_t) noexcept
{
    if (nbInputs != 2 || in[0].desc.dims.nbDims != 4 || in[1].desc.dims.nbDims != 4)
    {
        std::cerr << "[QKMixedAccumPlugin] expected two rank-4 tensors" << std::endl;
    }
}

size_t QKMixedAccumPlugin::getWorkspaceSize(
    PluginTensorDesc const*, int32_t, PluginTensorDesc const*, int32_t) const noexcept
{
    return kWorkspaceBytes;
}

int32_t QKMixedAccumPlugin::enqueue(PluginTensorDesc const* inputDesc,
    PluginTensorDesc const*, void const* const* inputs, void* const* outputs,
    void* workspace, cudaStream_t stream) noexcept
{
    auto const& left = inputDesc[0].dims;
    auto const& right = inputDesc[1].dims;
    if (left.nbDims != 4 || right.nbDims != 4 || mLtHandle == nullptr)
    {
        std::cerr << "[QKMixedAccumPlugin] enqueue precondition failed left_rank=" << left.nbDims
                  << " right_rank=" << right.nbDims << " handle=" << mLtHandle << std::endl;
        return 1;
    }
    int32_t const batch = left.d[0] * left.d[1];
    int32_t const m = left.d[2];
    int32_t const k = left.d[3];
    int32_t const n = mParams.family == 0 ? right.d[2] : right.d[3];
    DataType const outputType = mParams.outputType == 0 ? DataType::kFLOAT : DataType::kHALF;
    return launchMixedAccum(mLtHandle, inputs[0], inputs[1], outputs[0], workspace, kWorkspaceBytes,
        batch, m, n, k, mParams.family == 0, mParams.scale, outputType, stream);
}

size_t QKMixedAccumPlugin::getSerializationSize() const noexcept { return sizeof(MixedAccumParams); }

void QKMixedAccumPlugin::serialize(void* buffer) const noexcept
{
    char* cursor = static_cast<char*>(buffer);
    writeToBuffer<MixedAccumParams>(cursor, mParams);
}

void QKMixedAccumPlugin::destroy() noexcept { delete this; }
void QKMixedAccumPlugin::setPluginNamespace(char const* value) noexcept
{
    mNamespace = value ? value : "";
}
char const* QKMixedAccumPlugin::getPluginNamespace() const noexcept { return mNamespace.c_str(); }
char const* QKMixedAccumPlugin::getPluginType() const noexcept { return kName; }
char const* QKMixedAccumPlugin::getPluginVersion() const noexcept { return kPluginVersion; }
DataType QKMixedAccumPlugin::getOutputDataType(int32_t, DataType const*, int32_t) const noexcept
{
    return mParams.outputType == 0 ? DataType::kFLOAT : DataType::kHALF;
}
bool QKMixedAccumPlugin::ensureLtHandle() noexcept
{
    return mLtHandle != nullptr
        || cublasLtCreate(&mLtHandle) == CUBLAS_STATUS_SUCCESS;
}

void QKMixedAccumPlugin::releaseLtHandle() noexcept
{
    if (mLtHandle != nullptr)
    {
        cublasLtDestroy(mLtHandle);
        mLtHandle = nullptr;
    }
}

void QKMixedAccumPlugin::attachToContext(cudnnContext*, cublasContext*, IGpuAllocator*) noexcept
{
    ensureLtHandle();
}
void QKMixedAccumPlugin::detachFromContext() noexcept
{
    releaseLtHandle();
}
int32_t QKMixedAccumPlugin::initialize() noexcept
{
    return ensureLtHandle() ? 0 : 1;
}
void QKMixedAccumPlugin::terminate() noexcept
{
    releaseLtHandle();
}

QKMixedAccumPluginCreator::QKMixedAccumPluginCreator()
{
    mFields.emplace_back(PluginField{"family", nullptr, PluginFieldType::kINT32, 1});
    mFields.emplace_back(PluginField{"output_type", nullptr, PluginFieldType::kINT32, 1});
    mFields.emplace_back(PluginField{"scale", nullptr, PluginFieldType::kFLOAT32, 1});
    mFC.nbFields = static_cast<int32_t>(mFields.size());
    mFC.fields = mFields.data();
}
char const* QKMixedAccumPluginCreator::getPluginName() const noexcept { return kName; }
char const* QKMixedAccumPluginCreator::getPluginVersion() const noexcept { return kPluginVersion; }
PluginFieldCollection const* QKMixedAccumPluginCreator::getFieldNames() noexcept { return &mFC; }
IPluginV2* QKMixedAccumPluginCreator::createPlugin(char const*, PluginFieldCollection const* fields) noexcept
{
    PluginFieldReader reader(fields);
    MixedAccumParams params{};
    params.family = reader.get<int32_t>("family", 0);
    params.outputType = reader.get<int32_t>("output_type", 0);
    params.scale = reader.get<float>("scale", 1.0F);
    auto* plugin = new QKMixedAccumPlugin(params);
    plugin->setPluginNamespace(mNamespace.c_str());
    return plugin;
}
IPluginV2* QKMixedAccumPluginCreator::deserializePlugin(
    char const*, void const* data, size_t length) noexcept
{
    auto* plugin = new QKMixedAccumPlugin(data, length);
    plugin->setPluginNamespace(mNamespace.c_str());
    return plugin;
}
void QKMixedAccumPluginCreator::setPluginNamespace(char const* value) noexcept
{
    mNamespace = value ? value : "";
}
char const* QKMixedAccumPluginCreator::getPluginNamespace() const noexcept { return mNamespace.c_str(); }

} // namespace cobevt_mixed_accum
