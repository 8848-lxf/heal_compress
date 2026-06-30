# LiDAR Pyramid TensorRT Deployment Summary

- model: lidar_pyramid
- checkpoint: /home/lixingfeng/UniAD_examine/Auto_Search/original_models/dairv2s/LiDAROnly/lidar_pyramid/net_epoch_bestval_at17.pth
- hypes_yaml: /home/lixingfeng/UniAD_examine/Auto_Search/original_models/dairv2s/LiDAROnly/lidar_pyramid/config.yaml
- output_root: /home/lixingfeng/UniAD_examine/heal_compress/tests/quant_deploy/outputs/lidar_pyramid_agent_export_strategy_compare
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
PyTorch original | 0.8721 | 0.8534 | 0.7464 | 0.8240 | 0.0000 | 50
PyTorch export wrapper | 0.8721 | 0.8534 | 0.7464 | 0.8240 | 0.0000 | 50
ONNXRuntime FP32 | 0.8721 | 0.8534 | 0.7475 | 0.8243 | -0.0003 | 50
TensorRT FP32 | 0.8721 | 0.8534 | 0.7464 | 0.8240 | 0.0000 | 50
TensorRT FP16 | 0.8722 | 0.8523 | 0.7310 | 0.8185 | 0.0055 | 50

- onnxruntime_fp32_close_to_pytorch: None
- tensorrt_fp32_close_to_pytorch: None
- suspected_area: None

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

- configs: /home/lixingfeng/UniAD_examine/heal_compress/tests/quant_deploy/outputs/lidar_pyramid_agent_export_strategy_compare/configs
- onnx: /home/lixingfeng/UniAD_examine/heal_compress/tests/quant_deploy/outputs/lidar_pyramid_agent_export_strategy_compare/artifacts/onnx
- engines: /home/lixingfeng/UniAD_examine/heal_compress/tests/quant_deploy/outputs/lidar_pyramid_agent_export_strategy_compare/artifacts/engines
- calibration: /home/lixingfeng/UniAD_examine/heal_compress/tests/quant_deploy/outputs/lidar_pyramid_agent_export_strategy_compare/calibration
- benchmark: /home/lixingfeng/UniAD_examine/heal_compress/tests/quant_deploy/outputs/lidar_pyramid_agent_export_strategy_compare/benchmark
- evaluation: /home/lixingfeng/UniAD_examine/heal_compress/tests/quant_deploy/outputs/lidar_pyramid_agent_export_strategy_compare/evaluation
- logs: /home/lixingfeng/UniAD_examine/heal_compress/tests/quant_deploy/outputs/lidar_pyramid_agent_export_strategy_compare/logs
- summary: /home/lixingfeng/UniAD_examine/heal_compress/tests/quant_deploy/outputs/lidar_pyramid_agent_export_strategy_compare/summary
- debug: /home/lixingfeng/UniAD_examine/heal_compress/tests/quant_deploy/outputs/lidar_pyramid_agent_export_strategy_compare/debug
