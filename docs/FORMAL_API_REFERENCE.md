# Formal API reference

## Tracing

- `trace_model`
- `build_dependency_graph`
- `build_dependency_scopes`
- `build_coupled_channel_units`
- `build_atomic_prune_units`
- `serialize_trace_result`
- `load_trace_result`

## Pruning

- `load_model`
- `score_pruning_units`
- `select_pruning_request`
- `build_physical_pruning_plan`
- `legalize_pruning_plan`
- `materialize_pruning`
- `replay_pruning`
- `build_physical_structure_snapshot`
- `compute_physical_hashes`
- `validate_physical_model`

## ONNX and precision

- `export_signal_maxk_onnx`
- `export_pruned_signal_maxk_onnx`
- `build_onnx_origin_map`
- `apply_canonical_node_names`
- `validate_onnx_against_physical_snapshot`
- `generate_precision_profile`
- `build_canonical_precision_mapping`
- `insert_explicit_qdq`
- `trace_qdq_root_initializer`
- `validate_qdq_against_physical_snapshot`

## TensorRT and evaluation

- `build_trt_command`
- `build_trt_engine`
- `validate_engine_structure`
- `validate_precision_realization`
- `validate_engine_provenance`
- `load_trt_engine`
- `run_engine_smoke`
- `evaluate_engine`
- `compute_detection_metrics`
- `summarize_latency`

All public functions have typed results and raise package-specific exceptions.
See their docstrings for accepted in-memory objects and path forms. TensorRT
execution APIs import TensorRT lazily and never run as a side effect of import,
configuration, command generation or validation parsing.

