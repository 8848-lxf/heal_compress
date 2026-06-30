# Fixed-K Voxel Mask Scatter Plugin Report

bucket_id | fixed_K | frames
--- | --- | ---
0 | 9728 | 14
1 | 23040 | 26
2 | 23552 | 10
3 | 24064 | 0

precision | AP@0.30 | AP@0.50 | AP@0.70 | mAP | forward_p50 | execute_p50 | total_runner_p50 | FPS | frames
--- | --- | --- | --- | --- | --- | --- | --- | --- | ---
fp32 | 0.8720 | 0.8533 | 0.7476 | 0.8243 | 7.0506 | 5.1292 | 7.0506 | 141.8316 | 50
fp16 | 0.8722 | 0.8523 | 0.7310 | 0.8185 | 3.7895 | 2.1780 | 3.7895 | 263.8858 | 50

## Safety

- pointpillar_scatter_plugin_equivalent: True
- plugin_max_abs_error: 0.0078
- invalid_padded_voxel_affects_spatial_features: False
- valid_voxel_mask is consumed by PointPillarScatterTRT in this graph.
