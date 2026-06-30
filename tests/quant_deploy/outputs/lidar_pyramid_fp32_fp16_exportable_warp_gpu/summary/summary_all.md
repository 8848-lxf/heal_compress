# LiDAR Pyramid TensorRT Deployment Summary

- model: lidar_pyramid
- checkpoint: /home/lixingfeng/UniAD_examine/Auto_Search/original_models/dairv2s/LiDAROnly/lidar_pyramid/net_epoch_bestval_at17.pth
- hypes_yaml: /home/lixingfeng/UniAD_examine/Auto_Search/original_models/dairv2s/LiDAROnly/lidar_pyramid/config.yaml
- output_root: /home/lixingfeng/UniAD_examine/heal_compress/tests/quant_deploy/outputs/lidar_pyramid_fp32_fp16_exportable_warp_gpu
- onnx_path: /home/lixingfeng/UniAD_examine/heal_compress/tests/quant_deploy/outputs/lidar_pyramid_fp32_fp16_exportable_warp_gpu/artifacts/onnx/fp32/lidar_pyramid_fp32_dynamic.onnx

## Environment

- python: /home/lixingfeng/miniconda3/envs/modelopt/bin/python
- conda_prefix: /home/lixingfeng/miniconda3/envs/modelopt
- TRT_ROOT: /home/lixingfeng/UniAD_examine/TensorRT-10.9_x86_cu118
- trtexec_found: True
- trtexec_path: /home/lixingfeng/UniAD_examine/TensorRT-10.9_x86_cu118/bin/trtexec
- tensorrt_available: True
- tensorrt_version: 10.9.0.34
- modelopt_available: False
- modelopt_version: None
- cuda_available: True
- gpu_names: ['NVIDIA H800']

## ONNX Export

- success: True
- export_boundary: full_model
- opset: 17
- error: None

## Precision Results

precision | implemented | build | benchmark | engine size MB | p50 ms | p90 ms | p95 ms | FPS | speedup vs FP32 | error
--- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ---
fp32 | yes | no | no | null | null | null | null | null | null | [06/25/2026-08:21:47] [E] [TRT] ModelImporter.cpp:967: ERROR: onnxOpImporters.cpp:5889 In function importSqueeze:
fp16 | yes | no | no | null | null | null | null | null | null | [06/25/2026-08:21:55] [E] [TRT] ModelImporter.cpp:967: ERROR: onnxOpImporters.cpp:5889 In function importSqueeze:

## Special Ops

- GridSample: 6
- AffineGrid: 0
- Scatter: 8
- Gather: 95
- NonZero: 2
- Inverse: 0
- unsupported_ops: 0

## Evaluation

- evaluation_status: not_run
- reason: current stage only benchmarks TensorRT engine forward latency

## Output Directories

- configs: /home/lixingfeng/UniAD_examine/heal_compress/tests/quant_deploy/outputs/lidar_pyramid_fp32_fp16_exportable_warp_gpu/configs
- onnx: /home/lixingfeng/UniAD_examine/heal_compress/tests/quant_deploy/outputs/lidar_pyramid_fp32_fp16_exportable_warp_gpu/artifacts/onnx
- engines: /home/lixingfeng/UniAD_examine/heal_compress/tests/quant_deploy/outputs/lidar_pyramid_fp32_fp16_exportable_warp_gpu/artifacts/engines
- calibration: /home/lixingfeng/UniAD_examine/heal_compress/tests/quant_deploy/outputs/lidar_pyramid_fp32_fp16_exportable_warp_gpu/calibration
- benchmark: /home/lixingfeng/UniAD_examine/heal_compress/tests/quant_deploy/outputs/lidar_pyramid_fp32_fp16_exportable_warp_gpu/benchmark
- evaluation: /home/lixingfeng/UniAD_examine/heal_compress/tests/quant_deploy/outputs/lidar_pyramid_fp32_fp16_exportable_warp_gpu/evaluation
- logs: /home/lixingfeng/UniAD_examine/heal_compress/tests/quant_deploy/outputs/lidar_pyramid_fp32_fp16_exportable_warp_gpu/logs
- summary: /home/lixingfeng/UniAD_examine/heal_compress/tests/quant_deploy/outputs/lidar_pyramid_fp32_fp16_exportable_warp_gpu/summary
- debug: /home/lixingfeng/UniAD_examine/heal_compress/tests/quant_deploy/outputs/lidar_pyramid_fp32_fp16_exportable_warp_gpu/debug
