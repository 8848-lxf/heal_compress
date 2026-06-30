# LiDAR Pyramid TensorRT Deployment Summary

- model: None
- checkpoint: None
- hypes_yaml: None
- output_root: None
- onnx_path: None

## Environment

- python: None
- conda_prefix: None
- TRT_ROOT: None
- trtexec_found: None
- trtexec_path: None
- tensorrt_available: None
- tensorrt_version: None
- modelopt_available: None
- modelopt_version: None
- cuda_available: None
- gpu_names: None

## ONNX Export

- success: None
- export_boundary: None
- opset: None
- error: None

## Deployment Equivalence

- export_forward_mode: None
- is_export_specialized_wrapper: None
- is_original_forward: None
- num_pyramid_scales: None
- sequence_ops_removed: None
- wrapper_equivalence_num_frames: None
- wrapper_equivalence_max_abs_error: null
- wrapper_equivalence_mean_abs_error: null
- trt_fp32_vs_pytorch_error: None
- trt_fp16_vs_pytorch_error: None

## Precision Results

precision | implemented | build | benchmark | engine size MB | p50 ms | p90 ms | p95 ms | FPS | speedup vs FP32 | error
--- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ---

## Five-Way AP Comparison

backend | AP@0.30 | AP@0.50 | AP@0.70 | mAP | mAP drop vs PyTorch | actual frames
--- | --- | --- | --- | --- | --- | ---
PyTorch original | 0.8721 | 0.8534 | 0.7462 | 0.8239 | 0.0000 | 50
PyTorch export wrapper | 0.8721 | 0.8534 | 0.7462 | 0.8239 | 0.0000 | 50
ONNXRuntime FP32 | 0.8720 | 0.8533 | 0.7462 | 0.8238 | 0.0001 | 50
TensorRT FP32 | 0.8721 | 0.8534 | 0.7462 | 0.8239 | 0.0000 | 50
TensorRT FP16 | 0.8735 | 0.8535 | 0.7334 | 0.8201 | 0.0038 | 50

- onnxruntime_fp32_close_to_pytorch: True
- tensorrt_fp32_close_to_pytorch: True
- suspected_area: AP is aligned across PyTorch, ONNXRuntime, and TensorRT FP32

## Special Ops

- GridSample: 0
- AffineGrid: 0
- Scatter: 0
- Gather: 0
- NonZero: 0
- Inverse: 0
- unsupported_ops: 0

## Evaluation

- evaluation_status: not_run
- reason: current stage only benchmarks TensorRT engine forward latency

## Output Directories

