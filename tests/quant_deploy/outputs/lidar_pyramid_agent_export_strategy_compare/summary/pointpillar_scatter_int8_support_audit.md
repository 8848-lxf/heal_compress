# PointPillarScatterTRT INT8 Support Audit

- plugin_supports_int8_io: False
- plugin_supports_fp16_io: True
- plugin_supports_fp32_io: True
- expected_plugin_precision_in_int8_engine: fp16
- int8_engine_requires_reformat_around_plugin: True
- risk_level: medium

## Dtype Findings

- valid_voxel_mask dtype: {'onnx_expected': 'float32 in the dynamic fixed-K ONNX inputs', 'plugin_accepts': ['fp32', 'fp16', 'int32', 'bool']}
- voxel_coords dtype: int32
- feature/output INT8: not supported unless both supportsFormatCombination and CUDA enqueue add kINT8 handling.

## Recommendation

Use native TensorRT INT8 calibration first and allow this plugin to run as an FP16/FP32 precision island. Audit the INT8 engine layer info for Reformat/Quantize/Dequantize around PointPillarScatterTRT. If AP drops, protect heads and the plugin in FP16 before moving to Q/DQ or ModelOpt.
