# Agent Mode Fixed-K Plugin Ablation Report

mode | precision | frames | true_dynamic_N | per_N_engine | AP@0.30 | AP@0.50 | AP@0.70 | mAP | execute p50 | execute p95 | forward p50 | forward p95 | FPS | notes
--- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ---
padded_agent_static_fixed_k_plugin | fp32 | 50 | False | False | 0.8721 | 0.8534 | 0.7450 | 0.8235 | 5.1333 | 5.9208 | 6.9053 | 8.9115 | 144.8153 | fixed max_cav=2 padded
padded_agent_static_fixed_k_plugin | fp16 | 50 | False | False | 0.8723 | 0.8523 | 0.7312 | 0.8186 | 2.1810 | 3.9609 | 3.9628 | 6.7782 | 252.3490 | fixed max_cav=2 padded
dynamic_agent_dim_fixed_k_plugin | fp32 | 50 | False | True | 0.8721 | 0.8534 | 0.7476 | 0.8244 | 4.9859 | 5.0422 | 6.4706 | 7.5576 | 154.5446 | per-N fallback
dynamic_agent_dim_fixed_k_plugin | fp16 | 50 | False | True | 0.8723 | 0.8523 | 0.7322 | 0.8189 | 2.1481 | 3.5758 | 3.6728 | 5.2375 | 272.2700 | per-N fallback

## Answers

- current_existing_fixed_k_plugin_engine_executes_PointPillarScatterTRT: None
- current_existing_results_agent_export_mode: None
- dynamic_agent_dim_still_slower_after_fixed_k_plugin: False
- valid_agent_mask_extra_overhead_obvious: True
- dynamic_agent_dim_per_N_fp16_50_speedup_vs_padded: 1.0789422836518103
- true_dynamic_N_single_engine_feasible: False
- dynamic_agent_dim_per_N_engine_worth_complexity: maybe for latency-only deployments; not default because it requires N1/N2 engine routing
- recommended_deployment_path: padded_agent_static --max_cav 2 + fixed-K bucket router + PointPillarScatterTRT
- need_agent_dimension_dynamic_optimization: False
- can_enter_int8_qdq: False
- int8_qdq_status: not_run_by_request; freeze FP32/FP16 first
