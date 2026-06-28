# PointPillarScatterTRT Engine Audit

field | value
--- | ---
plugin_compiled | True
plugin_shared_library_path | tests/quant_deploy/outputs/lidar_pyramid_agent_export_strategy_compare/artifacts/plugins/pointpillar_scatter_trt_build/libpointpillar_scatter_trt.so
plugin_loaded_during_engine_build | True
plugin_loaded_during_engine_deserialize | True
onnx_contains_pointpillar_scatter_plugin_node | True
engine_contains_pointpillar_scatter_plugin_layer | True
valid_voxel_mask_is_engine_input | True
valid_voxel_mask_consumed_by_plugin | True
current_fixed_k_results_are_plugin_results | True
plugin_equivalence_fp32_max_abs_error | 0.0
plugin_equivalence_fp16_max_abs_error | 0.007808685302734375
invalid_padded_voxel_affects_spatial_features | False
