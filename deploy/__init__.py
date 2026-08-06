"""TensorRT engine building and deployment utilities."""

from .post_scatter import (
    POST_SCATTER_CONTRACT,
    audit_post_scatter_onnx,
    export_post_scatter_onnx,
    filter_post_scatter_module_paths,
    filter_post_scatter_pruning_units,
    filter_post_scatter_quantization_groups,
    post_scatter_shape_profiles,
    prepare_post_scatter_inputs,
)

__all__ = [
    "TRTBuilder",
    "POST_SCATTER_CONTRACT",
    "audit_post_scatter_onnx",
    "export_post_scatter_onnx",
    "filter_post_scatter_module_paths",
    "filter_post_scatter_pruning_units",
    "filter_post_scatter_quantization_groups",
    "post_scatter_shape_profiles",
    "prepare_post_scatter_inputs",
]


def __getattr__(name: str):
    if name == "TRTBuilder":
        from .trt_builder import TRTBuilder

        return TRTBuilder
    raise AttributeError(name)
