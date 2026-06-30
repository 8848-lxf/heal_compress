# Final fixedK Full-Cover TrainCalib Deployment Report

- old_fixed_K: 24064
- new_fixed_K: 29696
- calibration_split: train for formal INT8
- evaluation_split: val
- calibration_eval_overlap: false
- old fixed_K=24064 full-val filtered results are superseded.

scheme | engine_strategy | fixed_K | precision | calibration_split | calibration_frames | evaluation_split | total_val | evaluated | skipped | engine_count | total_engine_size_MB | AP@0.30 | AP@0.50 | AP@0.70 | mAP | mAP_drop_vs_FP16 | execute_p50 | forward_p50 | FPS | reliable_latency | notes
--- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ---
padded_agent_static | padded_agent_static_fixed_k_plugin | 24064 | fp32 | None | None | val | 1789 | 1165 | 624 | 4 | None | 0.8361 | 0.7904 | 0.6091 | 0.7452 | None | 5.7730560302734375 | 6.814312189817429 | 146.74995394168937 | True | superseded_by_fixedK_full_cover=true
padded_agent_static | padded_agent_static_fixed_k_plugin | 24064 | fp16 | None | None | val | 1789 | 1165 | 624 | 4 | None | 0.8358 | 0.7908 | 0.6074 | 0.7447 | None | 1.9134080410003662 | 2.9458515346050262 | 339.46042027337876 | True | superseded_by_fixedK_full_cover=true
padded_agent_static | padded_agent_static_fixed_k_plugin | 24064 | int8 | None | 200 | None | None | None | None | None | None | None | None | None | None | None | None | None | None | None | superseded_by_fixedK_full_cover=true
dynamic_agent_dim | dynamic_agent_dim_bucket_fixed_k_plugin | 24064 | fp32 | None | None | val | 1789 | 1165 | 624 | 5 | None | 0.8362 | 0.7905 | 0.6092 | 0.7453 | None | 5.7325119972229 | 6.703704595565796 | 149.17125087245876 | True | superseded_by_fixedK_full_cover=true
dynamic_agent_dim | dynamic_agent_dim_bucket_fixed_k_plugin | 24064 | fp16 | None | None | val | 1789 | 1165 | 624 | 5 | None | 0.8358 | 0.7906 | 0.6077 | 0.7447 | None | 1.8693439960479736 | 2.8651803731918335 | 349.0181663104135 | True | superseded_by_fixedK_full_cover=true
dynamic_agent_dim | dynamic_agent_dim_bucket_int8 | 24064 | int8 | None | 50 | val | 1789 | 1165 | 624 | 5 | None | 0.7141 | 0.6742 | 0.4912 | 0.6265 | None | 1.5631359815597534 | 2.566862851381302 | 389.58061178137024 | True | superseded_by_fixedK_full_cover=true
dynamic_agent_dim | dynamic_agent_dim_bucket_int8 | 24064 | int8 | None | 200 | val | 1789 | 1165 | 624 | 5 | None | 0.7446 | 0.7019 | 0.5065 | 0.651 | None | 1.5567359924316406 | 2.558410167694092 | 390.8677399063439 | True | superseded_by_fixedK_full_cover=true
dynamic_agent_dim | dynamic_agent_dim_bucket_int8 | 24064 | int8 | None | None | val | 1789 | 1165 | 624 | 5 | None | 0.7451 | 0.7037 | 0.5064 | 0.6517 | None | 1.5872960090637207 | 2.593815326690674 | 385.5324585794057 | True | superseded_by_fixedK_full_cover=true
dynamic_agent_single_engine_maxK | single TensorRT engine | 24064 | fp32 | None | None | val | 1789 | 1165 | 624 | 1 | None | 0.8361 | 0.7904 | 0.6093 | 0.7453 | None | 5.910048007965088 | 6.852108985185623 | 145.94046915511956 | True | superseded_by_fixedK_full_cover=true
dynamic_agent_single_engine_maxK | single TensorRT engine | 24064 | fp16 | None | None | val | 1789 | 1165 | 624 | 1 | None | 0.8358 | 0.7907 | 0.6076 | 0.7447 | None | 1.9547200202941895 | 2.944324165582657 | 339.63651546571754 | True | superseded_by_fixedK_full_cover=true
dynamic_agent_single_engine_maxK | single TensorRT engine | 24064 | int8 | train | 50 | val | 1789 | 1165 | 624 | 1 | None | 0.6481 | 0.6135 | 0.4314 | 0.5643 | None | 1.6291520595550537 | 2.5889016687870026 | 386.2641876501001 | True | superseded_by_fixedK_full_cover=true
dynamic_agent_single_engine_maxK | single TensorRT engine | 24064 | int8 | train | 200 | val | 1789 | 1165 | 624 | 1 | None | 0.7015 | 0.6616 | 0.4696 | 0.6109 | None | 1.6639039516448975 | 2.6270970702171326 | 380.64828716715397 | True | superseded_by_fixedK_full_cover=true
padded_agent_static | padded_agent_static_fixed_k_plugin | 29696 | fp32 | None | None | val | 1789 | 1789 | 0 | 4 | 120.3061 | 0.8263 | 0.7817 | 0.6029 | 0.737 | None | 5.1473917961120605 | 6.341380998492241 | 157.6943571499276 | True | 
padded_agent_static | padded_agent_static_fixed_k_plugin | 29696 | fp16 | None | None | val | 1789 | 1789 | 0 | 4 | 88.974 | 0.826 | 0.782 | 0.601 | 0.7363 | None | 2.1719040870666504 | 3.3626407384872437 | 297.38532236122006 | True | 
padded_agent_static | padded_agent_static_fixed_k_plugin | 29696 | int8 | train | 200 | val | 1789 | 1789 | 0 | 4 | 71.1101 | 0.6594 | 0.6184 | 0.4303 | 0.5694 | 0.1669 | 1.5994880199432373 | 2.8149541467428207 | 355.245573416213 | True | 
dynamic_agent_dim | dynamic_agent_dim_bucket_fixed_k_plugin | 29696 | fp32 | None | None | val | 1789 | 1789 | 0 | 8 | 241.7984 | 0.8263 | 0.7817 | 0.6026 | 0.7369 | None | 4.994880199432373 | 6.140599027276039 | 162.85056157519514 | True | deployment package engine_count=8 differs from full-val evaluated engine_count=5
dynamic_agent_dim | dynamic_agent_dim_bucket_fixed_k_plugin | 29696 | fp16 | None | None | val | 1789 | 1789 | 0 | 8 | 176.1661 | 0.826 | 0.7818 | 0.6013 | 0.7364 | None | 2.126847982406616 | 3.28262522816658 | 304.6342273310689 | True | deployment package engine_count=8 differs from full-val evaluated engine_count=5
dynamic_agent_dim | dynamic_agent_dim_bucket_int8 | 29696 | int8 | train | 50 | val | 1789 | 1789 | 0 | 5 | 91.904 | 0.3299 | 0.3104 | 0.2104 | 0.2836 | 0.4528 | 1.6024960279464722 | 2.7093086391687393 | 369.09785232398497 | True | 
dynamic_agent_dim | dynamic_agent_dim_bucket_int8 | 29696 | int8 | train | 200 | val | 1789 | 1789 | 0 | 5 | 91.8705 | 0.7117 | 0.6683 | 0.4783 | 0.6194 | 0.117 | 1.592095971107483 | 2.7008913457393646 | 370.2481410729433 | True | 
dynamic_agent_single_engine_maxK | single TensorRT engine | 29696 | fp32 | None | None | val | 1789 | 1789 | 0 | 1 | 42.0362 | 0.8262 | 0.7815 | 0.6027 | 0.7368 | None | 5.406720161437988 | 6.48617185652256 | 154.17414495337337 | True | 
dynamic_agent_single_engine_maxK | single TensorRT engine | 29696 | fp16 | None | None | val | 1789 | 1789 | 0 | 1 | 30.3816 | 0.8259 | 0.7818 | 0.6012 | 0.7363 | None | 2.3736319541931152 | 3.4408215433359146 | 290.6282663617855 | True | 
dynamic_agent_single_engine_maxK | single TensorRT engine | 29696 | int8 | train | 50 | val | 1789 | 1789 | 0 | 1 | 26.6178 | 0.6416 | 0.6062 | 0.4239 | 0.5572 | 0.1791 | 1.816864013671875 | 2.8793606907129288 | 347.2993165550233 | True | 
dynamic_agent_single_engine_maxK | single TensorRT engine | 29696 | int8 | train | 200 | val | 1789 | 1789 | 0 | 1 | 26.78 | 0.7048 | 0.6632 | 0.4691 | 0.6124 | 0.1239 | 1.7993279695510864 | 2.8709396719932556 | 348.3180123063029 | True | 

## Answers

- skipped_624_root_cause_is_K_exceeds_fixed_K: true
- new_fixed_K: 29696
- new_fixed_K_covers_full_validation_set: True
- new_fixedK_full_val_completed: True
- gpu_blocker: None
- remaining_skipped_samples: 0
- single_engine_maxK_FP32_included: True
- padded_agent_static_INT8_train_calib200_evaluated: True
- padded_agent_static_INT8_train_calib200_skip_reason: None
- dynamic_bucket_FP16_forward_p50_speedup_vs_single_engine_maxK_FP16_percent: 4.6
- dynamic_bucket_FP16_size_ratio_vs_single_engine_maxK_FP16: 5.7985
- dynamic_bucket_INT8_train_calib200_forward_p50_speedup_vs_single_engine_maxK_INT8_percent: 5.92
- dynamic_bucket_INT8_train_calib200_size_ratio_vs_single_engine_maxK_INT8: 3.4306
- dynamic_bucket_train_calibration_drop_vs_historical: 0.0316
- single_engine_train_calibration_still_worse_than_dynamic: 0.007
- recommended_fp16_path: dynamic_agent_single_engine_maxK single TensorRT engine
- fastest_fp16_path: dynamic_agent_dim dynamic_agent_dim_bucket_fixed_k_plugin
- recommended_fp16_path_reason: single_engine_maxK FP16 is the default recommendation because it preserves FP16 AP, uses one serialized engine, and avoids the multi-route deployment package size increase. dynamic bucket FP16 remains the speed upper-bound option.
- recommended_int8_speed_candidate: None None
- recommend_qdq_modelopt_next: True
- final_recommended_default_path: dynamic_agent_single_engine_maxK FP16 fixedK29696 + PointPillarScatterTRT
- engine_file_size_report_json: /home/lixingfeng/UniAD_examine/heal_compress/tests/quant_deploy/outputs/lidar_pyramid_agent_export_strategy_compare/summary/engine_file_size_and_deployment_package_report.json
- HEAL/OpenCOOD source modified: false
