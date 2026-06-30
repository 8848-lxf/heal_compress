# Dynamic Fixed-K INT8 Engine Build Report

N | bucket | success | PointPillarScatterTRT | valid_voxel_mask | engine
--- | --- | --- | --- | --- | ---
1 | 0 | True | True | True | /home/lixingfeng/UniAD_examine/heal_compress/tests/quant_deploy/outputs/lidar_pyramid_agent_export_strategy_compare/artifacts/engines/dynamic_agent_dim_fixed_k_scatter_plugin/N1/int8_calib200_mixed_heads_fp16/lidar_pyramid_dynamic_agent_dim_N1_fixed_k_scatter_plugin_bucket0_int8_calib200_mixed_heads_fp16.engine
1 | 1 | True | True | True | /home/lixingfeng/UniAD_examine/heal_compress/tests/quant_deploy/outputs/lidar_pyramid_agent_export_strategy_compare/artifacts/engines/dynamic_agent_dim_fixed_k_scatter_plugin/N1/int8_calib200_mixed_heads_fp16/lidar_pyramid_dynamic_agent_dim_N1_fixed_k_scatter_plugin_bucket1_int8_calib200_mixed_heads_fp16.engine
1 | 2 | False | False | False | /home/lixingfeng/UniAD_examine/heal_compress/tests/quant_deploy/outputs/lidar_pyramid_agent_export_strategy_compare/artifacts/engines/dynamic_agent_dim_fixed_k_scatter_plugin/N1/int8_calib200_mixed_heads_fp16/lidar_pyramid_dynamic_agent_dim_N1_fixed_k_scatter_plugin_bucket2_int8_calib200_mixed_heads_fp16.engine
1 | 3 | False | False | False | /home/lixingfeng/UniAD_examine/heal_compress/tests/quant_deploy/outputs/lidar_pyramid_agent_export_strategy_compare/artifacts/engines/dynamic_agent_dim_fixed_k_scatter_plugin/N1/int8_calib200_mixed_heads_fp16/lidar_pyramid_dynamic_agent_dim_N1_fixed_k_scatter_plugin_bucket3_int8_calib200_mixed_heads_fp16.engine
2 | 0 | False | False | False | /home/lixingfeng/UniAD_examine/heal_compress/tests/quant_deploy/outputs/lidar_pyramid_agent_export_strategy_compare/artifacts/engines/dynamic_agent_dim_fixed_k_scatter_plugin/N2/int8_calib200_mixed_heads_fp16/lidar_pyramid_dynamic_agent_dim_N2_fixed_k_scatter_plugin_bucket0_int8_calib200_mixed_heads_fp16.engine
2 | 1 | True | True | True | /home/lixingfeng/UniAD_examine/heal_compress/tests/quant_deploy/outputs/lidar_pyramid_agent_export_strategy_compare/artifacts/engines/dynamic_agent_dim_fixed_k_scatter_plugin/N2/int8_calib200_mixed_heads_fp16/lidar_pyramid_dynamic_agent_dim_N2_fixed_k_scatter_plugin_bucket1_int8_calib200_mixed_heads_fp16.engine
2 | 2 | True | True | True | /home/lixingfeng/UniAD_examine/heal_compress/tests/quant_deploy/outputs/lidar_pyramid_agent_export_strategy_compare/artifacts/engines/dynamic_agent_dim_fixed_k_scatter_plugin/N2/int8_calib200_mixed_heads_fp16/lidar_pyramid_dynamic_agent_dim_N2_fixed_k_scatter_plugin_bucket2_int8_calib200_mixed_heads_fp16.engine
2 | 3 | True | True | True | /home/lixingfeng/UniAD_examine/heal_compress/tests/quant_deploy/outputs/lidar_pyramid_agent_export_strategy_compare/artifacts/engines/dynamic_agent_dim_fixed_k_scatter_plugin/N2/int8_calib200_mixed_heads_fp16/lidar_pyramid_dynamic_agent_dim_N2_fixed_k_scatter_plugin_bucket3_int8_calib200_mixed_heads_fp16.engine

- any_build_success: True
- all_build_success: False

- strategy: INT8 backbone with detection heads requested FP16
- forced_fp16_layer_patterns: cls, reg, dir, head
