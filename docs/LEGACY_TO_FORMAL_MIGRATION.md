# Legacy to formal migration

Use these replacements in release code:

| Legacy source/entry | Formal replacement |
|---|---|
| `tracer.generic_tracer.trace_model` | `tracer.api.trace_model` |
| `tracer.dependency_tracer.build_dependency_graph` | `tracer.api.build_dependency_graph` |
| `pruning.taylor_importance.*` | `pruning.api.score_pruning_units` |
| `pruning.greedy_budget_selector.*` | `pruning.api.select_pruning_request` |
| `pruning.physical_prune_plan.GlobalPhysicalPrunePlan` | `pruning.api.build_physical_pruning_plan` + `materialize_pruning` |
| `tools.latency_lut.physical_structure_v2` | `pruning.api.build_physical_structure_snapshot` and `compute_physical_hashes` |
| `quant_deploy.pruned_signal_maxk_exporter` | `quantization.api.export_pruned_signal_maxk_onnx` and origin-map APIs |
| v11 LUT builder canonical/QDQ helpers | `quantization.api.build_canonical_precision_mapping` and `insert_explicit_qdq` |
| v11/v12 trtexec helpers | `quantization.api.build_trt_command` / `build_trt_engine` |
| v12 structure/precision audits | `quantization.api.validate_engine_structure` / `validate_precision_realization` |

Formal functions take dataclass configuration or explicit keyword arguments;
they do not accept an `argparse.Namespace`. Paths must be supplied by the
caller. Historical scripts may remain as integration drivers, but release code
must not import them.

Saved pruning replays must include `group_keep_map` for every independently
ranked grouped convolution. A legacy artifact without that map is deliberately
rejected rather than reconstructed with shared local positions.

