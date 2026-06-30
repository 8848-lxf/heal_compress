# Padded Agent Static INT8 Train-Calib200 Build Report

- strategy: padded_agent_static
- fixed_K: 29696
- max_cav: 2
- calibration_split: train
- calibration_frames: 200
- calibration_npz_dir: tests/quant_deploy/outputs/lidar_pyramid_agent_export_strategy_compare/artifacts/calibration/train_calib_padded_agent_static_fixedK29696_200
- engine_count: 4
- build_success: True

bucket | samples | success | PointPillarScatterTRT | valid_agent_mask | valid_voxel_mask | int8_layers | fp16_fallback_layers | engine
--- | --- | --- | --- | --- | --- | --- | --- | ---
0 | 3 | True | True | True | True | None | None | /home/lixingfeng/UniAD_examine/heal_compress/tests/quant_deploy/outputs/lidar_pyramid_agent_export_strategy_compare/artifacts/engines/fixedK29696/padded_agent_static/int8_train_calib200/lidar_pyramid_padded_agent_static_fixedK29696_int8_train_calib200_bucket0.engine
1 | 144 | True | True | True | True | None | None | /home/lixingfeng/UniAD_examine/heal_compress/tests/quant_deploy/outputs/lidar_pyramid_agent_export_strategy_compare/artifacts/engines/fixedK29696/padded_agent_static/int8_train_calib200/lidar_pyramid_padded_agent_static_fixedK29696_int8_train_calib200_bucket1.engine
2 | 18 | True | True | True | True | None | None | /home/lixingfeng/UniAD_examine/heal_compress/tests/quant_deploy/outputs/lidar_pyramid_agent_export_strategy_compare/artifacts/engines/fixedK29696/padded_agent_static/int8_train_calib200/lidar_pyramid_padded_agent_static_fixedK29696_int8_train_calib200_bucket2.engine
3 | 35 | True | True | True | True | None | None | /home/lixingfeng/UniAD_examine/heal_compress/tests/quant_deploy/outputs/lidar_pyramid_agent_export_strategy_compare/artifacts/engines/fixedK29696/padded_agent_static/int8_train_calib200/lidar_pyramid_padded_agent_static_fixedK29696_int8_train_calib200_bucket3.engine
