#include "qk_mixed_accum_plugin.h"

#include <array>

namespace
{
static nvinfer1::PluginRegistrar<cobevt_mixed_accum::QKMixedAccumPluginCreator> gRegistrar{};
static cobevt_mixed_accum::QKMixedAccumPluginCreator gCreator{};
static std::array<nvinfer1::IPluginCreatorInterface*, 1> gCreators{&gCreator};
}

extern "C" void setLoggerFinder(nvinfer1::ILoggerFinder*) {}
extern "C" nvinfer1::IPluginCreatorInterface* const* getCreators(int32_t& count)
{
    count = static_cast<int32_t>(gCreators.size());
    return gCreators.data();
}
