# All Deployment Engines Full-Val Idle-GPU Report

previous results under GPU contention should not be trusted.

scheme | engine_strategy | precision | calibration | full_val | total_val | evaluated | skipped | engine_count | dynamic_N | bucket_router | single_engine | AP@0.30 | AP@0.50 | AP@0.70 | mAP | execute_p50 | execute_p95 | forward_p50 | forward_p95 | FPS | selected_gpu | contention | reliable_latency | notes
--- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ---
padded_agent_static | padded_agent_static_fixed_k_plugin | fp32 | None | True | 1789 | 1165 | 624 | 4 | False | True | False | 0.8361 | 0.7904 | 0.6091 | 0.7452 | 5.7730560302734375 | 5.830687999725342 | 6.814312189817429 | 6.928727030754089 | 146.74995394168937 | 2 | False | True | AP over evaluated_samples; K>24064 skipped
padded_agent_static | padded_agent_static_fixed_k_plugin | fp16 | None | True | 1789 | 1165 | 624 | 4 | False | True | False | 0.8358 | 0.7908 | 0.6074 | 0.7447 | 1.9134080410003662 | 1.944991946220398 | 2.9458515346050262 | 3.043659031391144 | 339.46042027337876 | 2 | False | True | AP over evaluated_samples; K>24064 skipped
dynamic_agent_dim | dynamic_agent_dim_bucket_fixed_k_plugin | fp32 | None | True | 1789 | 1165 | 624 | 5 | False | True | False | 0.8362 | 0.7905 | 0.6092 | 0.7453 | 5.7325119972229 | 5.761919975280762 | 6.703704595565796 | 6.781861186027527 | 149.17125087245876 | 2 | False | True | AP over evaluated_samples; K>24064 skipped
dynamic_agent_dim | dynamic_agent_dim_bucket_fixed_k_plugin | fp16 | None | True | 1789 | 1165 | 624 | 5 | False | True | False | 0.8358 | 0.7906 | 0.6077 | 0.7447 | 1.8693439960479736 | 1.8940800428390503 | 2.8651803731918335 | 2.9527023434638977 | 349.0181663104135 | 2 | False | True | AP over evaluated_samples; K>24064 skipped
dynamic_agent_dim | dynamic_agent_dim_bucket_int8 | int8 | calib50 | True | 1789 | 1165 | 624 | 5 | False | True | False | 0.7141 | 0.6742 | 0.4912 | 0.6265 | 1.5631359815597534 | 1.5930880308151245 | 2.566862851381302 | 2.647843211889267 | 389.58061178137024 | 2 | False | True | AP over evaluated_samples; K>24064 skipped
dynamic_agent_dim | dynamic_agent_dim_bucket_int8 | int8 | calib200 | True | 1789 | 1165 | 624 | 5 | False | True | False | 0.7446 | 0.7019 | 0.5065 | 0.651 | 1.5567359924316406 | 1.5833920240402222 | 2.558410167694092 | 2.642463892698288 | 390.8677399063439 | 2 | False | True | AP over evaluated_samples; K>24064 skipped
dynamic_agent_dim | dynamic_agent_dim_bucket_int8 | int8 | calib200_mixed_heads_fp16 | True | 1789 | 1165 | 624 | 5 | False | True | False | 0.7451 | 0.7037 | 0.5064 | 0.6517 | 1.5872960090637207 | 1.6342400312423706 | 2.593815326690674 | 2.657506614923477 | 385.5324585794057 | 2 | False | True | AP over evaluated_samples; K>24064 skipped
dynamic_agent_single_engine_maxK | single TensorRT engine | fp32 | None | True | 1789 | 1165 | 624 | 1 | True | False | True | 0.8361 | 0.7904 | 0.6093 | 0.7453 | 5.910048007965088 | 5.930016040802002 | 6.852108985185623 | 6.907258182764053 | 145.94046915511956 | 2 | False | True | AP over evaluated_samples; K>24064 skipped
dynamic_agent_single_engine_maxK | single TensorRT engine | fp16 | None | True | 1789 | 1165 | 624 | 1 | True | False | True | 0.8358 | 0.7907 | 0.6076 | 0.7447 | 1.9547200202941895 | 1.964959979057312 | 2.944324165582657 | 3.0091889202594757 | 339.63651546571754 | 2 | False | True | AP over evaluated_samples; K>24064 skipped
dynamic_agent_single_engine_maxK | single TensorRT engine | int8 | train_calib50 | True | 1789 | 1165 | 624 | 1 | True | False | True | 0.6481 | 0.6135 | 0.4314 | 0.5643 | 1.6291520595550537 | 1.6375999450683594 | 2.5889016687870026 | 2.6588328182697296 | 386.2641876501001 | 2 | False | True | AP over evaluated_samples; K>24064 skipped
dynamic_agent_single_engine_maxK | single TensorRT engine | int8 | train_calib200 | True | 1789 | 1165 | 624 | 1 | True | False | True | 0.7015 | 0.6616 | 0.4696 | 0.6109 | 1.6639039516448975 | 1.6945600509643555 | 2.6270970702171326 | 2.7508437633514404 | 380.64828716715397 | 2 | False | True | AP over evaluated_samples; K>24064 skipped

## Analysis

- single_engine_maxK_16ms_forward_p50_still_present: False
- single_engine_maxK_vs_dynamic_bucket_fp16_forward_ratio: 1.0276226212950972
- single_engine_maxK_fp16_mAP_delta_vs_dynamic_bucket_fp16: 0.0
- single_engine_maxK_fp16_acceptance: True
- dynamic_bucket_fp16_fastest_high_precision: True
- int8_smallest_mAP_drop_mode: calib200_mixed_heads_fp16
- int8_fastest_latency_mode: calib200
- recommend_int8_as_deployment_candidate: True
- recommend_int8_as_default_deployment: False
- recommend_qdq_modelopt: yes for INT8, especially single_engine_maxK where train_calib200 mAP drop exceeds 0.1; try mixed precision whitelist before full ModelOpt Q/DQ
- recommended_deployment_path: dynamic_agent_single_engine_maxK FP16
