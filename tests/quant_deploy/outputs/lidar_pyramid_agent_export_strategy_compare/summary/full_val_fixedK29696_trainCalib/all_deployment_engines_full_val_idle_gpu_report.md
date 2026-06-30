# All Deployment Engines Full-Val Idle-GPU Report

previous results under GPU contention should not be trusted.

scheme | engine_strategy | precision | calibration | full_val | total_val | evaluated | skipped | engine_count | dynamic_N | bucket_router | single_engine | AP@0.30 | AP@0.50 | AP@0.70 | mAP | execute_p50 | execute_p95 | forward_p50 | forward_p95 | FPS | selected_gpu | contention | reliable_latency | notes
--- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ---
padded_agent_static | padded_agent_static_fixed_k_plugin | int8 | train_calib200 | True | 1789 | 1789 | 0 | 4 | False | True | False | 0.6594 | 0.6184 | 0.4303 | 0.5694 | 1.5994880199432373 | 1.789952039718628 | 2.8149541467428207 | 3.105565905570984 | 355.245573416213 | 0 | False | True | AP over evaluated_samples; fixed_K=29696; skipped=0

## Analysis

- single_engine_maxK_16ms_forward_p50_still_present: False
- single_engine_maxK_vs_dynamic_bucket_fp16_forward_ratio: None
- single_engine_maxK_fp16_mAP_delta_vs_dynamic_bucket_fp16: None
- single_engine_maxK_fp16_acceptance: False
- dynamic_bucket_fp16_fastest_high_precision: False
- int8_smallest_mAP_drop_mode: train_calib200
- int8_fastest_latency_mode: train_calib200
- recommend_int8_as_deployment_candidate: False
- recommend_int8_as_default_deployment: False
- recommend_qdq_modelopt: yes for INT8, especially single_engine_maxK where train_calib200 mAP drop exceeds 0.1; try mixed precision whitelist before full ModelOpt Q/DQ
- recommended_deployment_path: dynamic_agent_dim fixed-K bucket router PointPillarScatterTRT FP16
