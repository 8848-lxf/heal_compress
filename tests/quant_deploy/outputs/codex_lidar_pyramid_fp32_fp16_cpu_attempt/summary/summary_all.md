# LiDAR Pyramid TensorRT Deployment Summary

- model: lidar_pyramid
- checkpoint: /home/lixingfeng/UniAD_examine/Auto_Search/original_models/dairv2s/LiDAROnly/lidar_pyramid/net_epoch_bestval_at17.pth
- hypes_yaml: /home/lixingfeng/UniAD_examine/Auto_Search/original_models/dairv2s/LiDAROnly/lidar_pyramid/config.yaml
- output_root: /home/lixingfeng/UniAD_examine/heal_compress/tests/quant_deploy/outputs/codex_lidar_pyramid_fp32_fp16_cpu_attempt
- onnx_path: None

## ONNX Export

- success: False
- export_boundary: full_model
- opset: 17
- error: Exporting the operator 'aten::affine_grid_generator' to ONNX opset version 17 is not supported. Please feel free to request support or submit a pull request on PyTorch GitHub: https://github.com/pytorch/pytorch/issues.

## Precision Results

precision | implemented | build | benchmark | engine size MB | p50 ms | p90 ms | p95 ms | FPS | speedup vs FP32 | error
--- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ---
fp32 | yes | no | no | null | null | null | null | null | null | skipped because ONNX export failed: Exporting the operator 'aten::affine_grid_generator' to ONNX opset version 17 is not supported. Please feel free to request support or submit a pull request on PyTorch GitHub: https://github.com/pytorch/pytorch/issues.
fp16 | yes | no | no | null | null | null | null | null | null | skipped because ONNX export failed: Exporting the operator 'aten::affine_grid_generator' to ONNX opset version 17 is not supported. Please feel free to request support or submit a pull request on PyTorch GitHub: https://github.com/pytorch/pytorch/issues.

## Special Ops

- GridSample: 0
- AffineGrid: 1
- Scatter: 0
- Gather: 0
- NonZero: 0
- Inverse: 0
- unsupported_ops: 1

## Evaluation

- evaluation_status: not_run
- reason: current stage only benchmarks TensorRT engine forward latency

## Output Directories

- configs: /home/lixingfeng/UniAD_examine/heal_compress/tests/quant_deploy/outputs/codex_lidar_pyramid_fp32_fp16_cpu_attempt/configs
- onnx: /home/lixingfeng/UniAD_examine/heal_compress/tests/quant_deploy/outputs/codex_lidar_pyramid_fp32_fp16_cpu_attempt/artifacts/onnx
- engines: /home/lixingfeng/UniAD_examine/heal_compress/tests/quant_deploy/outputs/codex_lidar_pyramid_fp32_fp16_cpu_attempt/artifacts/engines
- calibration: /home/lixingfeng/UniAD_examine/heal_compress/tests/quant_deploy/outputs/codex_lidar_pyramid_fp32_fp16_cpu_attempt/calibration
- benchmark: /home/lixingfeng/UniAD_examine/heal_compress/tests/quant_deploy/outputs/codex_lidar_pyramid_fp32_fp16_cpu_attempt/benchmark
- evaluation: /home/lixingfeng/UniAD_examine/heal_compress/tests/quant_deploy/outputs/codex_lidar_pyramid_fp32_fp16_cpu_attempt/evaluation
- logs: /home/lixingfeng/UniAD_examine/heal_compress/tests/quant_deploy/outputs/codex_lidar_pyramid_fp32_fp16_cpu_attempt/logs
- summary: /home/lixingfeng/UniAD_examine/heal_compress/tests/quant_deploy/outputs/codex_lidar_pyramid_fp32_fp16_cpu_attempt/summary
- debug: /home/lixingfeng/UniAD_examine/heal_compress/tests/quant_deploy/outputs/codex_lidar_pyramid_fp32_fp16_cpu_attempt/debug
