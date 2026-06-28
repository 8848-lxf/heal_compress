#include "pointpillar_scatter_plugin.h"

#include <NvInferPlugin.h>

#include <array>
#include <cstdint>

namespace
{
static nvinfer1::PluginRegistrar<pointpillar_scatter_trt::PointPillarScatterPluginCreator>
    gPointPillarScatterPluginRegistrar{};
static pointpillar_scatter_trt::PointPillarScatterPluginCreator gPointPillarScatterCreator{};
static std::array<nvinfer1::IPluginCreatorInterface*, 1> gPluginCreators{&gPointPillarScatterCreator};
} // namespace

extern "C" void setLoggerFinder(nvinfer1::ILoggerFinder*) {}

extern "C" nvinfer1::IPluginCreatorInterface* const* getCreators(int32_t& nbCreators)
{
    nbCreators = static_cast<int32_t>(gPluginCreators.size());
    return gPluginCreators.data();
}
