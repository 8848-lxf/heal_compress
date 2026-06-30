# Rebuild fixedK Full Cover Engines Report

- old_fixed_K: 24064
- new_fixed_K: 29696
- covers_full_val: true
- calibration_split: train
- plugin_path: tests/quant_deploy/outputs/lidar_pyramid_agent_export_strategy_compare/artifacts/plugins/pointpillar_scatter_trt_build/libpointpillar_scatter_trt.so

strategy | count
--- | ---
padded_fp32 | 4
padded_fp16 | 4
dynamic_fp32 | 8
dynamic_fp16 | 8
dynamic_int8_train | 10
single_fp32 | 1
single_fp16 | 1
single_int8_train | 2
