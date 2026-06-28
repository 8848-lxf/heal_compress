# LiDAR Pyramid TensorRT Deployment Summary

- model: lidar_pyramid
- checkpoint: /home/lixingfeng/UniAD_examine/Auto_Search/original_models/dairv2s/LiDAROnly/lidar_pyramid/net_epoch_bestval_at17.pth
- hypes_yaml: /home/lixingfeng/UniAD_examine/Auto_Search/original_models/dairv2s/LiDAROnly/lidar_pyramid/config.yaml
- output_root: tests/quant_deploy/outputs/lidar_pyramid_fp32_fp16_real_engine
- onnx_path: /home/lixingfeng/UniAD_examine/heal_compress/tests/quant_deploy/outputs/lidar_pyramid_fp32_fp16_real_engine/artifacts/onnx/fp32/lidar_pyramid_fp32_dynamic.onnx

## Environment

- python: /home/lixingfeng/anaconda3/envs/modelopt/bin/python
- conda_prefix: /home/lixingfeng/anaconda3/envs/modelopt
- TRT_ROOT: /home/lixingfeng/UniAD_examine/HEAL/prune_model/TensorRT-10.9_x86_cu118
- trtexec_found: True
- trtexec_path: /home/lixingfeng/UniAD_examine/HEAL/prune_model/TensorRT-10.9_x86_cu118/bin/trtexec
- tensorrt_available: True
- tensorrt_version: 10.9.0.34
- modelopt_available: False
- modelopt_version: None
- cuda_available: True
- gpu_names: ['NVIDIA GeForce RTX 4090', 'NVIDIA GeForce RTX 4090', 'NVIDIA GeForce RTX 4090', 'NVIDIA GeForce RTX 4090', 'NVIDIA GeForce RTX 4090', 'NVIDIA GeForce RTX 4090', 'NVIDIA GeForce RTX 4090', 'NVIDIA GeForce RTX 4090']

## ONNX Export

- success: True
- export_boundary: full_model
- opset: 17
- error: None

## Deployment Equivalence

- export_forward_mode: fixed_static
- is_export_specialized_wrapper: True
- is_original_forward: False
- num_pyramid_scales: 3
- sequence_ops_removed: True
- wrapper_equivalence_num_frames: 50
- wrapper_equivalence_max_abs_error: 0.0000
- wrapper_equivalence_mean_abs_error: 0.0000
- trt_fp32_vs_pytorch_error: {'success': True, 'max_abs_error': 0.011945724487304688, 'mean_abs_error': 0.0007648059947200636, 'relative_error': 0.0008250186801715633}
- trt_fp16_vs_pytorch_error: {'success': True, 'max_abs_error': 0.10426950454711914, 'mean_abs_error': 0.007969814216181703, 'relative_error': 0.009384643321528111}

## Precision Results

precision | implemented | build | benchmark | engine size MB | p50 ms | p90 ms | p95 ms | FPS | speedup vs FP32 | error
--- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ---
fp32 | yes | yes | yes | 62.8378 | 4.9574 | 7.1858 | 8.1238 | 201.7186 | 1.0000 | None
fp16 | yes | yes | yes | 30.5934 | 1.9224 | 3.0033 | 3.6306 | 520.1939 | 2.5788 | None
int8 | yes | yes | yes | 42.7562 | 1.2492 | 3.4602 | 7.0616 | 800.5059 | 2.8276 | not reevaluated in this run; previous AP=0 invalidated by dtype binding bug

## Fixed TRT Evaluation

- previous_trt_ap_zero_invalidated: True
- invalid_reason: pairwise_t_matrix dtype binding mismatch
- trt_dtype_binding_mismatch_found: True
- trt_dtype_binding_mismatched_inputs: ['pairwise_t_matrix']
- plugin_needed: False

engine | AP@0.30 | AP@0.50 | AP@0.70 | mAP | forward p50 ms | speedup vs PyTorch | mean AP drop
--- | --- | --- | --- | --- | --- | --- | ---
PyTorch | 0.8722 | 0.8535 | 0.7477 | 0.8245 | 16.5950 | 1.0000 | 0.0000
TensorRT FP32 | 0.5294 | 0.5058 | 0.4492 | 0.4948 | 4.9574 | 3.3475 | 0.3297
TensorRT FP16 | 0.5270 | 0.5034 | 0.4465 | 0.4923 | 1.9224 | 8.6326 | 0.3322

## Five-Way AP Comparison

backend | AP@0.30 | AP@0.50 | AP@0.70 | mAP | mAP drop vs PyTorch | actual frames
--- | --- | --- | --- | --- | --- | ---
PyTorch original | 0.8722 | 0.8535 | 0.7477 | 0.8245 | 0.0000 | 50
PyTorch fixed wrapper | 0.8722 | 0.8535 | 0.7477 | 0.8245 | 0.0000 | 50
ONNXRuntime FP32 | 0.5295 | 0.5058 | 0.4493 | 0.4949 | 0.3296 | 50
TensorRT FP32 | 0.5295 | 0.5058 | 0.4492 | 0.4948 | 0.3297 | 50
TensorRT FP16 | 0.5271 | 0.5035 | 0.4466 | 0.4924 | 0.3321 | 50

- onnxruntime_fp32_close_to_pytorch: False
- tensorrt_fp32_close_to_pytorch: False
- suspected_area: ONNX export/fixed_static wrapper/ORT-TRT shared evaluator

## Special Ops

- GridSample: 6
- AffineGrid: 0
- Scatter: 4
- Gather: 37
- NonZero: 2
- Inverse: 0
- unsupported_ops: 0

## Evaluation

- evaluation_status: rerun_after_dtype_fix
- latency_scope: engine_forward_only for TensorRT p50/p90/p95/FPS

## Output Directories

- configs: /home/lixingfeng/UniAD_examine/heal_compress/tests/quant_deploy/outputs/lidar_pyramid_fp32_fp16_real_engine/configs
- onnx: /home/lixingfeng/UniAD_examine/heal_compress/tests/quant_deploy/outputs/lidar_pyramid_fp32_fp16_real_engine/artifacts/onnx
- engines: /home/lixingfeng/UniAD_examine/heal_compress/tests/quant_deploy/outputs/lidar_pyramid_fp32_fp16_real_engine/artifacts/engines
- calibration: /home/lixingfeng/UniAD_examine/heal_compress/tests/quant_deploy/outputs/lidar_pyramid_fp32_fp16_real_engine/calibration
- benchmark: /home/lixingfeng/UniAD_examine/heal_compress/tests/quant_deploy/outputs/lidar_pyramid_fp32_fp16_real_engine/benchmark
- evaluation: /home/lixingfeng/UniAD_examine/heal_compress/tests/quant_deploy/outputs/lidar_pyramid_fp32_fp16_real_engine/evaluation
- logs: /home/lixingfeng/UniAD_examine/heal_compress/tests/quant_deploy/outputs/lidar_pyramid_fp32_fp16_real_engine/logs
- summary: /home/lixingfeng/UniAD_examine/heal_compress/tests/quant_deploy/outputs/lidar_pyramid_fp32_fp16_real_engine/summary
- debug: /home/lixingfeng/UniAD_examine/heal_compress/tests/quant_deploy/outputs/lidar_pyramid_fp32_fp16_real_engine/debug
