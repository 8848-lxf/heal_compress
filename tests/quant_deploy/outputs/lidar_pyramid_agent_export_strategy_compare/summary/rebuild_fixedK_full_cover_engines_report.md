# Rebuild fixedK Full Cover Engines Report

- server_hostname: zs-nj-tap-gpu18
- selected_gpu_for_build: 3
- old_fixed_K: 24064
- new_fixed_K: 29696
- covers_full_val: true
- calibration_split: train
- plugin_path: tests/quant_deploy/outputs/lidar_pyramid_agent_export_strategy_compare/artifacts/plugins/pointpillar_scatter_trt_build/libpointpillar_scatter_trt.so

strategy | count
--- | ---
padded_fp32 | 4
padded_fp16 | 4
padded_int8_train_calib200 | 4
dynamic_fp32 | 8
dynamic_fp16 | 8
dynamic_int8_train | 5
single_fp32 | 1
single_fp16 | 1
single_int8_train | 1
