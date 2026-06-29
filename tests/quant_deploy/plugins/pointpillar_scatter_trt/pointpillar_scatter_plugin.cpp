#include "pointpillar_scatter_plugin.h"

#include <NvInferPlugin.h>

using namespace nvinfer1;

namespace pointpillar_scatter_trt
{
namespace
{
char const* kName = "PointPillarScatterTRT";
} // namespace

PointPillarScatterPlugin::PointPillarScatterPlugin(PointPillarScatterParams params)
    : mParams(params)
{
}

PointPillarScatterPlugin::PointPillarScatterPlugin(void const* data, size_t length)
{
    assert(length == getSerializationSize());
    char const* d = static_cast<char const*>(data);
    mParams = readFromBuffer<PointPillarScatterParams>(d);
}

IPluginV2DynamicExt* PointPillarScatterPlugin::clone() const noexcept
{
    auto* plugin = new PointPillarScatterPlugin(mParams);
    plugin->setPluginNamespace(mNamespace.c_str());
    return plugin;
}

int32_t PointPillarScatterPlugin::getNbOutputs() const noexcept { return 1; }

DimsExprs PointPillarScatterPlugin::getOutputDimensions(
    int32_t,
    DimsExprs const* inputs,
    int32_t nbInputs,
    IExprBuilder& exprBuilder) noexcept
{
    assert(nbInputs == 3 || nbInputs == 4);
    DimsExprs out{};
    out.nbDims = 4;
    // New single-engine path passes pairwise_t_matrix as input 3 and uses its
    // runtime agent dimension. The legacy 3-input path keeps serialized N.
    out.d[0] = (nbInputs == 4) ? inputs[3].d[1] : exprBuilder.constant(mParams.numAgents);
    out.d[1] = inputs[0].d[1];
    out.d[2] = exprBuilder.constant(mParams.height);
    out.d[3] = exprBuilder.constant(mParams.width);
    return out;
}

bool PointPillarScatterPlugin::supportsFormatCombination(
    int32_t pos,
    PluginTensorDesc const* inOut,
    int32_t nbInputs,
    int32_t nbOutputs) noexcept
{
    assert((nbInputs == 3 || nbInputs == 4) && nbOutputs == 1);
    if (pos == 0)
    {
        return isFp(inOut[pos].type) && inOut[pos].format == TensorFormat::kLINEAR;
    }
    if (pos == 1)
    {
        return inOut[pos].type == DataType::kINT32 && inOut[pos].format == TensorFormat::kLINEAR;
    }
    if (pos == 2)
    {
        return (inOut[pos].type == DataType::kFLOAT || inOut[pos].type == DataType::kHALF
                   || inOut[pos].type == DataType::kINT32 || inOut[pos].type == DataType::kBOOL)
            && inOut[pos].format == TensorFormat::kLINEAR;
    }
    if (nbInputs == 4 && pos == 3)
    {
        return isFp(inOut[pos].type) && inOut[pos].format == TensorFormat::kLINEAR;
    }
    return inOut[pos].type == inOut[0].type && inOut[pos].format == TensorFormat::kLINEAR;
}

void PointPillarScatterPlugin::configurePlugin(
    DynamicPluginTensorDesc const* in,
    int32_t nbInputs,
    DynamicPluginTensorDesc const*,
    int32_t) noexcept
{
    if (nbInputs != 3 && nbInputs != 4)
    {
        std::cerr << "[PointPillarScatterTRT] expected pillar_features, voxel_coords, valid_voxel_mask"
                     " and optional pairwise_t_matrix shape reference."
                  << std::endl;
        return;
    }
    auto const& pf = in[0].desc.dims;
    auto const& coords = in[1].desc.dims;
    auto const& mask = in[2].desc.dims;
    if (pf.nbDims != 2 || coords.nbDims != 2 || coords.d[1] != 4 || mask.nbDims != 1)
    {
        std::cerr << "[PointPillarScatterTRT] expected [K,C], [K,4], [K]." << std::endl;
    }
    if (nbInputs == 4)
    {
        auto const& shapeRef = in[3].desc.dims;
        if (shapeRef.nbDims != 5)
        {
            std::cerr << "[PointPillarScatterTRT] dynamic-N mode expects pairwise_t_matrix [1,N,N,4,4]." << std::endl;
        }
    }
}

size_t PointPillarScatterPlugin::getWorkspaceSize(PluginTensorDesc const*, int32_t, PluginTensorDesc const*, int32_t)
    const noexcept
{
    return 0;
}

int32_t PointPillarScatterPlugin::enqueue(
    PluginTensorDesc const* inputDesc,
    PluginTensorDesc const* outputDesc,
    void const* const* inputs,
    void* const* outputs,
    void*,
    cudaStream_t stream) noexcept
{
    auto const& d = inputDesc[0].dims;
    int32_t K = d.d[0];
    int32_t C = d.d[1];
    int32_t runtimeAgents = outputDesc[0].dims.d[0];
    if (runtimeAgents <= 0)
    {
        runtimeAgents = mParams.numAgents;
    }
    launchPointPillarScatter(
        inputs[0],
        inputs[1],
        inputs[2],
        outputs[0],
        K,
        C,
        runtimeAgents,
        mParams.height,
        mParams.width,
        inputDesc[0].type,
        inputDesc[1].type,
        inputDesc[2].type,
        stream);
    return 0;
}

size_t PointPillarScatterPlugin::getSerializationSize() const noexcept { return sizeof(PointPillarScatterParams); }

void PointPillarScatterPlugin::serialize(void* buffer) const noexcept
{
    char* d = static_cast<char*>(buffer);
    writeToBuffer<PointPillarScatterParams>(d, mParams);
}

void PointPillarScatterPlugin::destroy() noexcept { delete this; }
void PointPillarScatterPlugin::setPluginNamespace(char const* pluginNamespace) noexcept
{
    mNamespace = pluginNamespace ? pluginNamespace : "";
}
char const* PointPillarScatterPlugin::getPluginNamespace() const noexcept { return mNamespace.c_str(); }
char const* PointPillarScatterPlugin::getPluginType() const noexcept { return kName; }
char const* PointPillarScatterPlugin::getPluginVersion() const noexcept { return kPluginVersion; }
DataType PointPillarScatterPlugin::getOutputDataType(int32_t, DataType const* inputTypes, int32_t) const noexcept
{
    return inputTypes[0];
}
void PointPillarScatterPlugin::attachToContext(cudnnContext*, cublasContext*, IGpuAllocator*) noexcept {}
void PointPillarScatterPlugin::detachFromContext() noexcept {}
int32_t PointPillarScatterPlugin::initialize() noexcept { return 0; }
void PointPillarScatterPlugin::terminate() noexcept {}

PointPillarScatterPluginCreator::PointPillarScatterPluginCreator()
{
    mFields.emplace_back(PluginField{"num_agents", nullptr, PluginFieldType::kINT32, 1});
    mFields.emplace_back(PluginField{"height", nullptr, PluginFieldType::kINT32, 1});
    mFields.emplace_back(PluginField{"width", nullptr, PluginFieldType::kINT32, 1});
    mFC.nbFields = static_cast<int32_t>(mFields.size());
    mFC.fields = mFields.data();
}

char const* PointPillarScatterPluginCreator::getPluginName() const noexcept { return kName; }
char const* PointPillarScatterPluginCreator::getPluginVersion() const noexcept { return kPluginVersion; }
PluginFieldCollection const* PointPillarScatterPluginCreator::getFieldNames() noexcept { return &mFC; }

IPluginV2* PointPillarScatterPluginCreator::createPlugin(char const*, PluginFieldCollection const* fc) noexcept
{
    PluginFieldReader reader(fc);
    PointPillarScatterParams p{};
    p.numAgents = reader.get<int32_t>("num_agents", 2);
    p.height = reader.get<int32_t>("height", 200);
    p.width = reader.get<int32_t>("width", 704);
    auto* plugin = new PointPillarScatterPlugin(p);
    plugin->setPluginNamespace(mNamespace.c_str());
    return plugin;
}

IPluginV2* PointPillarScatterPluginCreator::deserializePlugin(
    char const*,
    void const* serialData,
    size_t serialLength) noexcept
{
    auto* plugin = new PointPillarScatterPlugin(serialData, serialLength);
    plugin->setPluginNamespace(mNamespace.c_str());
    return plugin;
}

void PointPillarScatterPluginCreator::setPluginNamespace(char const* pluginNamespace) noexcept
{
    mNamespace = pluginNamespace ? pluginNamespace : "";
}
char const* PointPillarScatterPluginCreator::getPluginNamespace() const noexcept { return mNamespace.c_str(); }

} // namespace pointpillar_scatter_trt
