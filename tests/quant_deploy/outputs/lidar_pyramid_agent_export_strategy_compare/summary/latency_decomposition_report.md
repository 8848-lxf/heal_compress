# Latency Decomposition Report

- conclusion: TRT engine-only and fixed-shape runner are fast, but real 50-frame varying voxel shapes are slow inside TensorRT execution; the main issue is dynamic shape/profile execution, not Python copy/allocation overhead.
- int8_qdq_modelopt_plugin: not run

scheme | precision | engine_only_p50 | fixed_shape_execute_p50 | execute_cuda_event_p50 | h2d_p50 | d2h_p50 | set_shape_p50 | alloc_p50 | total_runner_p50 | PyTorch_forward_p50 | bottleneck | recommendation
--- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ---
padded_agent_static | fp32 | 7.2672 | 7.2059 | 172.6679 | 0.0000 | 0.6796 | 0.0389 | 0.0000 | 174.9262 | 35.5058 | dynamic_shape_profile_execution | Engine is fast for fixed shape, but real varying voxel shapes are slow in TensorRT execution; bucket/pad voxel count near opt shape or rebuild tighter profiles.
padded_agent_static | fp16 | 2.8590 | 1.8514 | 173.9551 | 0.0000 | 0.6715 | 0.0372 | 0.0000 | 177.9034 | 35.5058 | dynamic_shape_profile_execution | Engine is fast for fixed shape, but real varying voxel shapes are slow in TensorRT execution; bucket/pad voxel count near opt shape or rebuild tighter profiles.
dynamic_agent_dim | fp32 | 5.5203 | 4.7544 | 353.7111 | 0.0000 | 0.6682 | 0.0319 | 0.0000 | 356.0999 | 19.1494 | dynamic_shape_profile_execution | Engine is fast for fixed shape, but real varying voxel shapes are slow in TensorRT execution; bucket/pad voxel count near opt shape or rebuild tighter profiles.
dynamic_agent_dim | fp16 | 1.7946 | 1.5309 | 356.5035 | 0.0000 | 0.6232 | 0.0324 | 0.0000 | 360.9167 | 19.1494 | dynamic_shape_profile_execution | Engine is fast for fixed shape, but real varying voxel shapes are slow in TensorRT execution; bucket/pad voxel count near opt shape or rebuild tighter profiles.
