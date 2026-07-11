# Formalization inventory

This inventory records where production-capable logic was found and the one
formal package that owns it after consolidation.  Files under `tests/` and
versioned programs under `tools/` remain historical evidence or integration
drivers; they are not release dependencies.

| 功能 | 当前文件 | 当前版本 | 正式目标 | 保留原因 | 迁移状态 |
|---|---|---|---|---|---|
| runtime module/op trace | `tracer/generic_tracer.py`, `tracer/op_graph.py` | current pre-formal | `tracer/api.py`, `tracer/module_call_tracer.py` | real forward call order and tensor flow | merged |
| static graph | `tracer/dependency_tracer.py`, `tracer/static_graph_builder.py` | current pre-formal | `tracer/static_graph_builder.py`, `tracer/dependency_graph.py` | FX module/function graph | merged |
| dependency closure | `pruning/propagation.py`, `tracer/pruning_group.py` | v9.x-v10.9 | `tracer/dependency_tracer.py`, `tracer/coupled_units.py` | residual, concat and downstream-input recipes | merged |
| coupled/atomic units | `pruning/units.py`, `tests/test_coupled_units_search_readiness_v102.py` | v10.2 | `tracer/types.py`, `tracer/coupled_units.py`, `tracer/atomic_units.py` | most complete unit metadata | merged |
| directional protection | `pruning/protection_policy.py`, `tools/latency_lut/run_v108_complete_taylor_greedy_pruner.py` | v10.8 | `tracer/protection.py`, `pruning/policies/protection.py` | fixed output with dependency input propagation | merged |
| HEAL model loading | `pruning/model_io.py`, `tests/test_prune_and_eval.py` | v10.9 | `pruning/model_io.py` | config/checkpoint/state validation | merged; caller supplies paths |
| Taylor importance | `pruning/taylor_importance.py`, `tests/test_v108_complete_taylor_greedy_pruner.py` | v10.8 | `pruning/importance/first_order_taylor.py` | verified dependency-mean and scope-mean normalization | merged |
| L1/L2/Fisher importance | `search/importance.py`, `tests/test_taylor_fisher_importance_connection.py` | current | `pruning/importance/` | optional non-default scorers | merged |
| global budget selector | `pruning/greedy_budget_selector.py`, `tools/latency_lut/run_v109_param_budget_round4_pruner.py` | v10.9 | `pruning/selection/` | normalized global ranking and param/channel budgets | merged |
| dense channel alignment | `tools/latency_lut/run_v109_param_budget_round4_pruner.py` | v10.9 | `pruning/policies/alignment.py` | validated round-to-4 behavior | merged |
| grouped selection | `pruning/grouped_pergroup8_policy.py`, `tests/test_grouped_conv_independent_topk_v81.py` | v8.1-v10.9 | `pruning/selection/grouped_conv.py` | per-group independent local ranking | merged and generalized to explicit allowed set |
| grouped physical surgery | `pruning/grouped_conv.py`, `pruning/physical_prune_plan.py` | v9.7-v10.9 | `pruning/materialization/grouped_conv.py` | balanced per-group slicing and replay metadata | merged |
| one-shot physical plan | `pruning/physical_prune_plan.py`, `tests/test_global_physical_prune_plan_v94.py` | v9.4-v10.9 | `pruning/materialization/planner.py`, `legalizer.py`, `executor.py` | original-index merge and one materialization transaction | merged |
| physical snapshot/hash/ledger | `tools/latency_lut/physical_structure_v2.py` | v12 | `pruning/artifacts/` | latest physical-truth and deterministic hash semantics | merged |
| signal-maxK exporter | `quant_deploy/pruned_signal_maxk_exporter.py`, `tests/quant_deploy/export_dynamic_single_engine_maxk_onnx.py` | v11-v12 | `quantization/export/` | fixed-K inputs, dynamic agents and plugin interface | merged |
| ONNX origin map | `quant_deploy/pruned_signal_maxk_exporter.py` | v11 origin-map fix | `quantization/export/origin_mapping.py` | repeated-call-aware module-to-initializer evidence | merged |
| canonical naming | `quant_deploy/pruned_signal_maxk_exporter.py`, `tests/test_onnx_export_origin_mapping.py` | v11-v12 | `quantization/export/canonical_naming.py` | deterministic unique TRT-visible identity | merged |
| precision profiles | `tools/latency_lut/run_v11_mixed_precision_lut_dataset_builder.py` | v11 | `quantization/precision/profile_generator.py` | 0/20/50/80 percent FP16/INT8 profiles | merged |
| canonical precision map | `tools/latency_lut/run_v11_mixed_precision_lut_dataset_builder.py` | v11 origin-aware | `quantization/precision/grouping.py` | fail-closed module/node/layer mapping | merged |
| explicit Q/DQ | `tools/latency_lut/run_v11_mixed_precision_lut_dataset_builder.py`, `tools/latency_lut/insert_full_graph_qdq.py` | v11-v12 | `quantization/precision/qdq_inserter.py` | activation and weight Q/DQ with reports | merged |
| Q/DQ root tracing | `tools/latency_lut/physical_structure_v2.py` | v12 | `quantization/precision/qdq_trace.py` | initializer truth through pass-through ops | merged |
| TensorRT command/build | `tools/latency_lut/run_v11_mixed_precision_lut_dataset_builder.py` | v11-v12 | `quantization/tensorrt/command.py`, `builder.py` | obey constraints, canonical layer precision, shapes/plugins | merged; execution opt-in only |
| structure checker | `tools/latency_lut/physical_structure_v2.py` | v12 | `quantization/tensorrt/structure_checker.py` | snapshot-v2 hard truth, no sampling fallback | merged |
| precision checker | `tools/latency_lut/audit_full_engine_structure_precision_v4.py` | v12 | `quantization/tensorrt/precision_checker.py` | requested-versus-realized parsing | merged |
| runtime/bindings | `quantization/utils/trt_runtime.py`, `tests/quant_deploy/dynamic_single_engine_maxk_common.py` | v11 | `quantization/tensorrt/runtime.py`, `bindings.py` | reusable runtime and binding allocation | merged; optional TensorRT dependency |
| evaluator/metrics/latency | `quantization/eval/evaluate_single_engine_maxk.py`, `tools/latency_lut/run_abcd_small_eval_v100.py` | v10-v12 | `quantization/evaluation/` | postprocess, metrics and latency summaries | merged; no evaluation run in formalization |
| hashing/atomic writes | `tools/latency_lut/physical_structure_v2.py` | v12 | `pruning/artifacts/io.py`, `quantization/artifacts/io.py` | stable JSON SHA256 and atomic replacement | merged |

Classification used during migration:

- Production algorithm: moved behind a typed API in one of the three formal packages.
- Compatibility entry point: imports the formal API and may emit
  `DeprecationWarning`.
- Test-only/integration: remains under `tests/` or versioned `tools/` and may
  depend on datasets, checkpoints, CUDA, plugins or TensorRT.
- Obsolete: retained in history but not imported by a formal package.

