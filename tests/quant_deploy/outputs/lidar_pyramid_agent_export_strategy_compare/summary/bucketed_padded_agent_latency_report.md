# Bucketed Padded Agent Latency Report

bucket_id | min_voxels | opt_voxels | max_voxels | frames
--- | --- | --- | --- | ---
0 | 1 | 9728 | 9728 | 14
1 | 9729 | 23040 | 23040 | 26
2 | 23041 | 23552 | 23552 | 10
3 | 23553 | 23553 | 24064 | 0

precision | AP@0.30 | AP@0.50 | AP@0.70 | mAP | forward_p50 | execute_p50 | total_runner_p50 | FPS
--- | --- | --- | --- | --- | --- | --- | --- | ---
fp32 | 0.8721 | 0.8534 | 0.7464 | 0.8240 | 186.2597 | 184.2483 | 186.7987 | 5.3688
fp16 | 0.8722 | 0.8523 | 0.7310 | 0.8185 | 189.0994 | 187.0387 | 189.6356 | 5.2882

## Plugin Decision

- remaining_bottleneck: BEV backbone Conv
- bucketed_speedup: 0.9371
- need_pointpillar_scatter_plugin: False
- need_fused_vfe_scatter_plugin: False
- need_bevwarp_plugin: False
- need_bevpool_plugin: False
- recommended_next_step: Do not write plugin yet; inspect remaining Shape/Reformat/Conv bottlenecks and TensorRT profile settings.
