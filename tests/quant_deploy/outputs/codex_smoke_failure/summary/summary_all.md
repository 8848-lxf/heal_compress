# LiDAR Pyramid TensorRT Deployment Summary

- model: lidar_pyramid
- checkpoint: /tmp/missing_lidar_pyramid.pth
- hypes_yaml: /tmp/missing_lidar_pyramid.yaml
- output_root: /home/lixingfeng/UniAD_examine/heal_compress/tests/quant_deploy/outputs/codex_smoke_failure
- onnx_path: None

## ONNX Export

- success: False
- export_boundary: full_model
- opset: 17
- error: hypes_yaml is required and does not exist: /tmp/missing_lidar_pyramid.yaml

## Precision Results

precision | implemented | build | benchmark | engine size MB | p50 ms | p90 ms | p95 ms | FPS | speedup vs FP32 | error
--- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ---
fp32 | yes | no | no | null | null | null | null | null | null | skipped because ONNX export failed: hypes_yaml is required and does not exist: /tmp/missing_lidar_pyramid.yaml
fp16 | yes | no | no | null | null | null | null | null | null | skipped because ONNX export failed: hypes_yaml is required and does not exist: /tmp/missing_lidar_pyramid.yaml
int8 | no | no | no | null | null | null | null | null | null | INT8 deployment is not implemented yet. The current implementation supports fp32/fp16 engine build and reserves the INT8 Q/DQ interface for future extension.

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

- configs: /home/lixingfeng/UniAD_examine/heal_compress/tests/quant_deploy/outputs/codex_smoke_failure/configs
- onnx: /home/lixingfeng/UniAD_examine/heal_compress/tests/quant_deploy/outputs/codex_smoke_failure/artifacts/onnx
- engines: /home/lixingfeng/UniAD_examine/heal_compress/tests/quant_deploy/outputs/codex_smoke_failure/artifacts/engines
- calibration: /home/lixingfeng/UniAD_examine/heal_compress/tests/quant_deploy/outputs/codex_smoke_failure/calibration
- benchmark: /home/lixingfeng/UniAD_examine/heal_compress/tests/quant_deploy/outputs/codex_smoke_failure/benchmark
- evaluation: /home/lixingfeng/UniAD_examine/heal_compress/tests/quant_deploy/outputs/codex_smoke_failure/evaluation
- logs: /home/lixingfeng/UniAD_examine/heal_compress/tests/quant_deploy/outputs/codex_smoke_failure/logs
- summary: /home/lixingfeng/UniAD_examine/heal_compress/tests/quant_deploy/outputs/codex_smoke_failure/summary
- debug: /home/lixingfeng/UniAD_examine/heal_compress/tests/quant_deploy/outputs/codex_smoke_failure/debug
