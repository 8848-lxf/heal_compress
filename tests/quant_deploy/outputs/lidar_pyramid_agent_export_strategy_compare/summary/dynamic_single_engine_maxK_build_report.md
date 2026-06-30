# Dynamic Single Engine maxK Build Report

- onnx_path: /home/lixingfeng/UniAD_examine/heal_compress/tests/quant_deploy/outputs/lidar_pyramid_agent_export_strategy_compare/artifacts/onnx/fixedK29696/dynamic_agent_single_engine_maxK/lidar_pyramid_dynamic_agent_single_engine_maxK.onnx
- fixed_K: 29696
- no_bucket_router: True
- no_N_engine_router: True
- true_dynamic_N: True
- engine_count_fp32: 1
- engine_count_fp16: 1
- engine_count_int8_train_calib50: 1
- engine_count_int8_train_calib200: 1

precision | calibration | success | engine
--- | --- | --- | ---
fp32 | None | True | /home/lixingfeng/UniAD_examine/heal_compress/tests/quant_deploy/outputs/lidar_pyramid_agent_export_strategy_compare/artifacts/engines/fixedK29696/dynamic_agent_single_engine_maxK/fp32/lidar_pyramid_dynamic_agent_single_engine_maxK_fp32.engine
fp16 | None | True | /home/lixingfeng/UniAD_examine/heal_compress/tests/quant_deploy/outputs/lidar_pyramid_agent_export_strategy_compare/artifacts/engines/fixedK29696/dynamic_agent_single_engine_maxK/fp16/lidar_pyramid_dynamic_agent_single_engine_maxK_fp16.engine
int8 | 50 | True | /home/lixingfeng/UniAD_examine/heal_compress/tests/quant_deploy/outputs/lidar_pyramid_agent_export_strategy_compare/artifacts/engines/fixedK29696/dynamic_agent_single_engine_maxK/int8_train_calib50/lidar_pyramid_dynamic_agent_single_engine_maxK_int8_train_calib50.engine
int8 | 200 | True | /home/lixingfeng/UniAD_examine/heal_compress/tests/quant_deploy/outputs/lidar_pyramid_agent_export_strategy_compare/artifacts/engines/fixedK29696/dynamic_agent_single_engine_maxK/int8_train_calib200/lidar_pyramid_dynamic_agent_single_engine_maxK_int8_train_calib200.engine
