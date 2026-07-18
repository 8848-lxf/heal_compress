# H800 explicit Q/DQ production acceptance handoff

## Round 1 — production contract and static-context closure

Starting point: branch `feature/heal-compress-h800`, HEAD `25b3dc1f28aaced9d4c585d7d4379dd9d726316e`. Existing H800 audit and ablation artifacts were treated as read-only evidence; no GA/Pareto work and no old 22/48 baseline rerun were started.

Implemented production-path changes:

- Added a formal `trusted_explicit_qdq_int8` all-keep baseline profile containing the established 27 INT8 weighted modules.
- Separated INT8 compute precision from FP16 functional output for `pyramid_backbone.single_head_0/1`; their weighted-output Q/DQ is disabled before the canonical `Sigmoid -> Add -> GridSample` path.
- Added fixed train200 entropy/KL calibration identity (exact manifest IDs/hash/order and deterministic seed), layout-aware per-channel weight semantics, and cache-version separation.
- Added graph Q/DQ topology hashing and production boundary reports joined to TensorRT EngineInspector realization.
- Added strict all-keep state-dict key/shape/dtype/value/tensor-hash identity validation.
- Added warmup-then-reset evaluation manifests, so 200 warmup frames do not consume the first 200 frames of the 1789-frame validation set.
- Added a `--baseline-only` production entry that exits before Fisher/proxy/GA work.
- Deployment identity now changes when the Q/DQ topology or merge contract changes.

Fresh real-model static verification in `univ2x-opt` on physical H800 GPU 6:

- weighted entries: 69;
- precision genes: 69;
- maximal legal INT8 genes: 67;
- protected FP16 genes: 2;
- pruning-scope false coupling: 0;
- multi-member force-same-precision groups: 0;
- `single_head_0/1` both carry `output_precision_policy=FP16` and `insert_activation_output_qdq=false` in the production search-space contract.

Evidence directory: `outputs/H800_explicit_qdq_acceptance_20260714_023005/`. Key files are `continuation_state.json`, `static_context_v2/context_report.json`, and `production_quantization_space_audit.{json,md}`. These output artifacts are intentionally not tracked by Git.

Tests completed:

- 83 quantization/QDQ/boundary/manifest tests passed;
- 31 baseline/Stage-2/search contract tests passed;
- Python compilation and `git diff --check` passed at this checkpoint.

Next required step: verify the isolated `modelopt` CUDA/GCC/TensorRT toolchain, then use the formal production entry to fresh-build the 27-layer entropy explicit-Q/DQ artifact and run the 10-frame gate. Canonical ONNX mapping, Add/Concat merge resolution, shrink boundary fusion, and realized 27/42 coverage remain intentionally unverified until that engine exists.

Completed: 2026-07-14 02:54:55 CST

---

## Round 2 — exact train200 TensorRT entropy provenance

The fixed train200 NPZ tensors were verified file-by-file in manifest order. Replaying those exact tensors through ModelOpt histogram/KL did not reproduce Legacy scales or accuracy, so the remaining discrepancy was narrowed to calibrator semantics rather than checkpoint, preprocessing, frame order, or BN folding.

Implemented a formal isolated TensorRT entropy calibration path:

- `search/integration/tensorrt_entropy_calibration_worker.py` runs `IInt8EntropyCalibrator2` in the `modelopt` environment, verifies every NPZ size/SHA256/fixedK input, builds a fresh cache and calibration-only engine, and records TensorRT/CUDA/GCC/G++/plugin/GPU/profile provenance.
- `search/integration/calibration_provider.py` invokes that worker through the isolated modelopt environment, rejects stale/dependency-mismatched caches, parses the TensorRT cache without fuzzy names, and requires exact canonical input/output tensor matches. Per-channel weight scales still come from the final folded ONNX initializer.
- The production context/config/evaluator now select `tensorrt_entropy_calibration2`, require the exact NPZ manifest, and force the first build.

Accepted toolchain evidence from the formal build:

- TensorRT `10.9.0.34`, CUDA `11.8`, H800 compute capability `9.0`, physical GPU 6;
- Python, `nvcc`, `gcc`, and `g++` all resolve inside `/home/lixingfeng/miniconda3/envs/modelopt`;
- plugin SHA256 `61d9adf44855ab2a595220718270d361c993f9ff281e986cdf8a62d5ca317ecd`;
- fresh cache SHA256 `2cd64360d302b05b5c52d259f19281b8a9a5be50087ce83448c213178d17028b`, 200 samples, no cache reuse, engine deserialize/context creation passed.

One earlier exploratory build resolved `nvcc` through `/usr/local/cuda`; it was stopped immediately and permanently marked `rejected_contaminated_tool_resolution` / `accepted_as_evidence=false`. It is preserved only as rejected provenance.

The clean replay compared 315 positive cache entries with Legacy: 282 of 287 common entries were bit-exact; all five differences were constant/helper tensors. More importantly, all 54 activation input/output boundaries used by the 27-layer production profile are exact Legacy matches. This proves that ModelOpt histogram KL and TensorRT EntropyCalibration2 are different recipes even on identical input tensors.

Completed: 2026-07-14 06:08:00 CST

---

## Round 3 — production v4 boundary and full-validation acceptance

Fresh production artifact:

`outputs/H800_explicit_qdq_acceptance_20260714_023005/production_trt_entropy_v4/h800_explicit_qdq_production_acceptance_20260713_131745/baselines/original_trusted_explicit_qdq_int8/`

Structural and deployment gates:

- all-keep and `pruned_unit_count=0`;
- original/physical parameters both `5,464,791`;
- state-dict key order, shape, dtype, exact value, and per-tensor hashes all match;
- 27 requested/realized INT8 weighted layers, 42 canonical FP16 layers, zero unresolved mappings;
- 79 Q + 79 DQ, all 27 weights per-channel and matched to final folded initializers;
- precision realization passed: 50 reformat layers, 12 hidden casts;
- merge realization passed: 19 FP16 Add merges and one TensorRT common-scale fused INT8 Concat under the declared policy-A contract;
- boundary audit passed with no issues. The shrink graph retains pre-ReLU Q nodes, but EngineInspector proves both shrink Conv/QDQ/ReLU paths are single fused INT8 layers; their effective boundary is `fused_weighted_output_plus_relu`. Two non-shrink pre-ReLU graph placements remain explicit warnings, not failures.

Production v4 `qdq.onnx` SHA256 is `1643be3fdceb3f18ffba05f034e116128898a88faa778fae0b0cebd7f2816ee2`, bit-identical to the established A2 27-layer exact-scale ablation. Production and the prior ModelOpt-entropy 27-layer run also have the same Q/DQ topology hash; only calibration semantics/scales differ.

Fresh gates on GPU 6:

| route | frames | AP@0.3 | AP@0.5 | AP@0.7 | mAP | forward p50 ms | skipped |
|---|---:|---:|---:|---:|---:|---:|---:|
| production v4 smoke | 10 | 0.860634 | 0.836214 | 0.507921 | 0.734923 | 2.546505 | 0 |
| production v4 gate | 200 | 0.793189 | 0.717492 | 0.332663 | 0.614448 | 2.823827 | 0 |
| Legacy single maxK FP16 | 1789 | 0.826011 | 0.781865 | 0.601124 | 0.736334 | 2.612188 | 0 |
| Legacy implicit INT8 train200 | 1789 | 0.704774 | 0.663810 | 0.470345 | 0.612976 | 2.363873 | 0 |
| production v4 explicit Q/DQ | 1789 | 0.784988 | 0.708115 | 0.350254 | 0.614452 | 2.859337 | 0 |

The production-to-Legacy delta is `+0.001476` mAP, inside the `0.005` accuracy-equivalence threshold. AP@0.7 is `-0.120091`, so localization accuracy is explicitly not equivalent. Production p50 is `+0.495464 ms` / `1.2096x`; recipe, coverage, and latency are not equivalent.

The reused A2 full result has the same 1789 frame IDs, Q/DQ ONNX, and scales: production differs by `+0.000795` mAP and `-0.051849 ms` p50, attributable to a fresh TensorRT engine/tactic build and runtime variation rather than graph or scale differences.

Ten-frame diagnostic parity reused the same input NPZs and ORT/PyTorch/Legacy references, but rebuilt only the production diagnostic engine. No catastrophic tensor was found. Shrink/head-input cosine is `0.958309`, zero ratio is `0.785796`, and neither tensor is near-all-zero.

Completed: 2026-07-14 06:08:30 CST

---

## Round 4 — source delivery and cross-server continuation

Production code now contains the H800 fixes rather than depending on ablation scripts:

- Q/DQ topology and boundary audit: `quantization/config.py`, `quantization/precision/qdq_inserter.py`, `quantization/reports/qdq_boundary.py`, `quantization/types.py`;
- trusted baseline, all-keep identity, compute/output split, merge contracts, realized validation, and fresh TRT calibration: `search/baselines/original_engines.py`, `search/integration/{calibration_provider,lidar_pyramid_context,runtime_environment,tensorrt_entropy_calibration_worker}.py`, `search/stage2/lidar_pyramid_real_evaluator.py`;
- fixed warmup-reset evaluation protocol: `search/integration/{data_provider,evaluation_worker}.py`;
- formal baseline-only entry/config and reproducible CLI argument recording: `search/cli.py`, `search/orchestration/lidar_pyramid_search.py`, `search/configs/lidar_pyramid_h800_explicit_qdq_acceptance.yaml`;
- inventory/report corrections and regression tests are included in the changed scripts/tests.

Final test evidence before commit:

- `155 passed`: all `tests/test_search*.py`, `tests/test_two_stage_joint_search.py`, and formal lidar-pyramid orchestration tests after the final CLI regression test;
- `107 passed`: modified production/QDQ/manifest contract files as an explicit focused suite;
- `119 passed`: formal package, ONNX origin mapping, quant deployment utilities, and formal tooling smoke suites in the final run; an earlier supplemental baseline/cache/runtime-shape run also passed 65 tests;
- `9 passed`: toolchain isolation and CLI reproduction command tests after the final CLI fix;
- compileall and `git diff --check` passed.

Do not sync generated `outputs/`, checkpoints, ONNX files, TensorRT engines, caches, or tensor dumps through Git. To reproduce calibration/A-B validation on another server, separately synchronize these read-only dependencies:

1. `/home/lixingfeng/UniAD_examine/Auto_Search/original_models/dairv2s/LiDAROnly/lidar_pyramid/net_epoch_bestval_at17.pth`
2. `/home/lixingfeng/UniAD_examine/Auto_Search/original_models/dairv2s/LiDAROnly/lidar_pyramid/config.yaml`
3. `tests/quant_deploy/outputs/lidar_pyramid_agent_export_strategy_compare/artifacts/calibration/train_calib_single_engine_maxK29696_200/` including all 200 NPZ files and `manifest.json`
4. Legacy FP16/INT8 engines, calibration cache, engine layer-info files, and the historical PointPillarScatterTRT plugin under `tests/quant_deploy/outputs/lidar_pyramid_agent_export_strategy_compare/` when reproducing the A/B comparison
5. Current production plugin or its source/build prerequisites; verify SHA256 before reuse

The 67-layer maximal explicit-entropy result remains a negative control (`mAP=0.509769` at 200 frames) and was not promoted to full validation. The accepted baseline is the 27-layer profile; the 67 legal genes remain available to a later mixed-precision search. GA/Pareto was not run in these rounds.

The delivery commit containing this handoff should be identified as the current branch HEAD after synchronization (`git log -1 --oneline`).

Completed: 2026-07-14 06:09:00 CST

---

## Round 5 — canonical 70-layer matched-coverage correction

This round supersedes two conclusions in Rounds 3–4. The old 27 INT8 / 43 FP16 route is an accuracy-safe mixed-precision control, not a Legacy-equivalent INT8 baseline. Engine fusion is also not sufficient evidence that a graph-level pre-ReLU Q/DQ boundary is semantically correct: production Q/DQ must own and consume the actual post-ReLU/post-merge tensor.

Canonical coverage was rebuilt from Legacy EngineInspector, the base ONNX graph, origin mapping, topology and calibration cache:

- canonical compute entries: 70;
- parameterized weighted ONNX entries: 69;
- protected parameter-free functional compute entry: one affine-grid `torch.bmm` MatMul group before `GridSample`;
- unmapped weighted entries: 0;
- Legacy exact layer set: 67 INT8 / 3 FP16;
- Legacy FP16 exceptions: pillar-VFE linear, `pyramid_backbone.single_head_2`, and `pyramid_backbone.functional_affine_grid_matmul`.

`encoder_m1.pillar_vfe.pfn_layers.0.linear` is separately mapped to its ONNX MatMul initializer. The three initializer-free `/MatMul*` affine-grid nodes are represented by one stable synthetic canonical entry and marked `mapped_but_protected_fp16`. Evidence is in:

`outputs/H800_explicit_qdq_acceptance_20260714_023005/matched_coverage_audit_20260714_094952/`

including `canonical_70_layer_precision_profile.json`, `legacy_vs_explicit_layer_set_diff.csv`, and `functional_matmul_mapping_audit.json`.

The production boundary implementation now resolves direct `Conv -> Relu`, `Conv -> Add -> Relu`, and merge paths to their semantic activation output. The inserter applies input/weight Q first, inserts output Q/DQ after the resolved semantic node, and fails closed when the calibration scale owner differs from the Q input. FP16 Add/Concat contracts insert explicit input casts and constrain the merge plus its unique post-merge ReLU. The two shrink boundaries are now:

- `shrink_conv.layers.0.double_conv.0` -> `/shrink_conv/layers.0/double_conv/double_conv.1/Relu_output_0`;
- `shrink_conv.layers.0.double_conv.2` -> `/shrink_conv/layers.0/double_conv/double_conv.3/Relu_output_0`.

No pre-ReLU output Q/DQ remains in the accepted E67 graph; only terminal cls/reg/dir heads quantize raw weighted outputs. The corrected static topology hash is `2cb4cabc8d939a730e474f48c2bbacf001f0093ee81c21a6249fe3c4f9cf1c3c`.

The first semantic-boundary E67 build exposed a second real issue: TensorRT fused the final Add+ReLU internally in FP32 despite Half merge inputs. Constraining the post-merge ReLU fixed the fallback; final precision, merge and boundary audits all pass. Failed intermediate directories were preserved and were not treated as accepted evidence.

Completed: 2026-07-14 12:45:30 CST

---

## Round 6 — E67-LS/E67-ENT matched-coverage acceptance and source delivery

Fresh production matched-coverage artifacts on fixed physical GPU 6:

- E67-LS: `outputs/H800_explicit_qdq_acceptance_20260714_023005/E67_LS_semantic_merge_v3_20260714_115500/`;
- E67-ENT: `outputs/H800_explicit_qdq_acceptance_20260714_023005/E67_ENT_semantic_merge_v2_20260714_115810/`;
- E27-LS control: `outputs/H800_explicit_qdq_acceptance_20260714_023005/E27_LS_semantic_merge_control_20260713_213135/`;
- E27-ENT control: `outputs/H800_explicit_qdq_acceptance_20260714_023005/E27_ENT_semantic_merge_control_20260713_213454/`;
- final report: `outputs/H800_explicit_qdq_acceptance_20260714_023005/matched_coverage_final_report_20260714_124504/`.

All four production builds are all-keep with exact checkpoint identity: 5,464,791 parameters on both sides, ordered state-dict keys identical, every tensor shape/dtype/value/hash exact, and `pruned_unit_count=0`.

E67-LS and E67-ENT match the exact canonical Legacy 67/3 layer set with zero unresolved entries. They have identical base ONNX, Q/DQ topology, per-channel weight specification, all activation scales, zero points and Q/DQ ONNX bytes. E67-ENT was nevertheless calibrated fresh with TensorRT EntropyCalibration2 over the fixed train200 manifest; the Legacy cache was not reused. Engine hashes differ because TensorRT tactics are rebuilt.

Corrected 200-frame gates:

| route | coverage | AP@0.3 | AP@0.5 | AP@0.7 | mAP | p50 ms | frames/skips |
|---|---|---:|---:|---:|---:|---:|---:|
| E67-LS | 67/3 | 0.754070 | 0.704562 | 0.488691 | 0.649108 | 2.786131 | 200/0 |
| E67-ENT | 67/3 | 0.755606 | 0.703319 | 0.484407 | 0.647778 | 2.814116 | 200/0 |
| E27-LS control | 27/43 | 0.796380 | 0.753992 | 0.550357 | 0.700243 | 3.044917 | 200/0 |
| E27-ENT control | 27/43 | 0.796927 | 0.753994 | 0.541525 | 0.697482 | 3.045852 | 200/0 |

Corrected full validation used the same manifest hash `e5cbece0ceaf2ac1b2c47305a3fa3bdc5f616baa1ce86b0754ba565a013ac463`, warmup-reset protocol, 1789 evaluated frames and zero skips:

| route | coverage | AP@0.3 | AP@0.5 | AP@0.7 | mAP | p50 ms | frames/skips |
|---|---|---:|---:|---:|---:|---:|---:|
| Legacy implicit L67 | 67/3 | 0.704774 | 0.663810 | 0.470345 | 0.612976 | 2.363873 | 1789/0 |
| E67-LS explicit | 67/3 | 0.731194 | 0.687489 | 0.483209 | 0.633964 | 2.776000 | 1789/0 |
| E67-ENT explicit | 67/3 | 0.731035 | 0.687310 | 0.481256 | 0.633200 | 2.772061 | 1789/0 |

E67-ENT minus Legacy is `+0.020224` mAP, `+0.010911` AP@0.70 and an observed `+0.408188 ms` / `1.1727x` p50. Therefore the independent verdicts are:

- `coverage_equivalent=true`;
- `scale_equivalent=true`;
- `accuracy_equivalent=false` because absolute mAP delta exceeds 0.005, even though explicit is more accurate;
- `latency_equivalent=false`;
- `quantization_recipe_equivalent=false`;
- `localization_accuracy_not_equivalent=true`.

GPU6 had concurrent background work during full explicit latency measurement, so the p50 values are observations rather than a strict isolated latency equivalence experiment. This does not affect AP/mAP or the false latency-equivalence verdict.

Corrected ten-frame tensor parity is bit-identical between E67-LS and E67-ENT diagnostic graphs. The first notable error is early backbone, not shrink. Shrink/head-input cosine is `0.971041`, SQNR `12.4343`, zero ratio `0.742456`, saturation ratio 0; no feature collapse remains. Moving from the superseded raw-boundary E67-LS (`mAP=0.570854` at 200 frames) to the semantic-boundary E67-LS (`0.649108`) adds `+0.078254`. Accuracy recovery therefore came from both calibration repair and boundary/merge repair; it was not calibration alone.

Production source changes include stable functional MatMul canonical mapping, semantic activation-boundary resolution, FP16 merge contracts, per-channel Q/DQ insertion, canonical realized-precision validation after fusion, deployment/signature metadata, and reusable matched-coverage audit/build/parity/report scripts. E27 raw inspector rows can be 25/43 after fusion, but its required canonical realization is strictly 27/43/0; raw counts remain recorded for diagnostics.

Focused final regression command passed `126` tests across production quantization groups, axis/calibration/boundary/merge/deployment contracts, baseline precision validation, formal lidar-pyramid orchestration and warmup-reset manifests. `py_compile` and `git diff --check` also pass. No outputs, checkpoints, ONNX files, calibration caches, `.plan` engines or tensor dumps are included in Git.

Final operational decision:

- `trusted_explicit_qdq_baseline=true` for the corrected E67 production path;
- E27 remains an accuracy-safe mixed-precision control only;
- `mixed_precision_GA_may_resume=false` in this delivery, pending explicit user acceptance of a trusted but Legacy recipe/accuracy-non-equivalent E67 baseline;
- no GA or Pareto search ran in this round.

The delivery commit containing this round is the branch HEAD after commit; retrieve it with `git log -1 --oneline` after synchronization.

Completed: 2026-07-14 12:46:00 CST

---

## Round 7 — strongly typed Phase-1 graph and builder contract

This phase starts the requested strongly typed deployment conversion. No GA, Pareto search, pruning materialization, full-validation rerun, or existing H800 artifact overwrite was performed.

The three previously ambiguous FP16 entries were resolved as follows:

- `encoder_m1.pillar_vfe.pfn_layers.0.linear` is a learned PFN `nn.Linear` followed by BN/ReLU/max. It was protected only to reproduce the Legacy 67/3 layer set; no TensorRT operator limitation has been proven.
- `pyramid_backbone.single_head_2` is the level-2 1x1 occupancy-score Conv used by cooperative pyramid fusion. The prior reason string `coordinate_grid_generation_requires_fp16_output` is inaccurate: this Conv does not generate the affine grid. It was also protected only for Legacy coverage matching.
- `pyramid_backbone.functional_affine_grid_matmul` is not an unresolved learned layer. It is a stable canonical group for the three initializer-free `torch.bmm` calls in `quantization.export.heal_lidar_pyramid._warp`, one per pyramid level, used to transform the sampling grid before `GridSample`. TensorRT fuses them to one GEMM row. The entry remains explicitly mapped and FP16-protected as parameter-free geometry compute.

The production FP16 merge contract means branch compute precision remains independently selectable. Each residual/concat input is dequantized/cast to FP16, Add/Concat executes in FP16, and a downstream INT8 weighted layer owns a new Q at the calibrated post-merge semantic tensor. It does not force the entire residual block or concat branch to FP16.

Production changes in this phase:

- `quantization/config.py`: added `QDQConfig.explicit_fp16_compute_casts` and `TensorRTBuildConfig.strongly_typed`; bumped policy versions.
- `quantization/precision/qdq_inserter.py`: added explicit graph-side FP16 casts for parameterized FP16 Conv/ConvTranspose/Gemm/MatMul inputs, protected functional MatMul operands, INT8-compute/FP16-output splits, and the existing FP16 merge boundaries. Original FP32 initializer bytes remain unchanged. The Q/DQ report now records all strong-typing casts and a `strong_typing_graph_contract_hash`.
- `quantization/tensorrt/command.py`: strongly typed mode emits `--stronglyTyped` and deliberately omits weak-typing flags `--fp16`, `--int8`, `--precisionConstraints`, `--layerPrecisions`, and `--layerOutputTypes`.
- `search/baselines/original_engines.py` and `search/stage2/lidar_pyramid_real_evaluator.py`: original baselines and production Stage-2 candidate builds now select strongly typed TensorRT mode.
- `search/stage2/{trt_modelopt,trt_build_worker}.py`: every production engine request now carries the TensorRT root, and the worker records a fail-closed modelopt build-environment manifest with Python/nvcc/GCC/G++ paths and versions, CUDA variables, TensorRT version/root, LD/CMake paths, GPU capability, plugin/QDQ hashes, builder config, command and command hash.
- `tests/test_formal_packages_cpu.py` and `tests/test_search_baseline_engines.py`: added regression coverage for mutually exclusive strongly/weak builder flags, graph-side FP16 parameterized and functional compute typing, initializer preservation, and production baseline defaults.

Search objective decision for the next phase:

```text
minimize normalized joint weight Taylor loss
subject to R_BOPS <= target
```

Physical parameter retention will be reported and may be used only as a deterministic epsilon/lexicographic tie-break. It is removed from the primary weighted scalar objective. Pruning perturbations (`delta_w=-w`) being larger than quantization perturbations is meaningful under one task-loss Taylor scale: if quantization reaches a BOPS budget with lower predicted AP loss, an AP-first search should prefer it. For budgets where W8A8 alone is sufficient, a no-pruning optimum is therefore valid; the 0.05 target is expected to require physical pruning. If pruning at every budget is later required, it must be stated as an explicit pruning constraint/Pareto axis rather than hidden in an arbitrary objective weight.

Validation completed in `univ2x-opt`:

```bash
pytest -q tests/test_formal_packages_cpu.py tests/test_search_baseline_engines.py \
  tests/test_search_quantization_groups.py tests/test_search_baseline_precision_validation.py
# 88 passed
python -m py_compile <all changed production modules>
git diff --check
```

TensorRT 10.9.0.34 exposes `--stronglyTyped`. A plain interactive `conda activate modelopt` inherited an earlier `/usr/local/cuda/bin` entry and initially resolved `which nvcc` to the system CUDA. The environment does contain `/home/lixingfeng/miniconda3/envs/modelopt/bin/nvcc` (CUDA 11.8.89), plus Conda GCC/G++ 11.2. The production `modelopt_python_command()` rebuilds PATH with `$CONDA_PREFIX/bin` first and the new worker rejects any resolved tool outside the modelopt prefix. No plugin was compiled with the inherited system nvcc. A real strongly typed H800 engine build and 10-frame gate remain pending; this code-only phase is not deployment acceptance.

Completed: 2026-07-17 00:15:04 CST

---

## Round 8 — strongly typed engine acceptance, transitive PFN boundary fix, and full validation

This round completes the real H800 strongly typed build/evaluation phase. No GA, Pareto search, pruning materialization, or existing accepted artifact was rebuilt. PyTorch/HEAL work ran after an explicit `conda activate univ2x-opt`; TensorRT workers ran in `modelopt` with the production environment sanitizer. The accepted build manifest records TensorRT `10.9.0.34`, modelopt CUDA `11.8.89`, Conda GCC/G++ `11.2.0`, H800 compute capability `9.0`, `system_toolchain_used=false`, plugin SHA256 `61d9adf44855ab2a595220718270d361c993f9ff281e986cdf8a62d5ca317ecd`, and `--stronglyTyped`. Weak builder flags `--fp16`, `--int8`, `--precisionConstraints`, `--layerPrecisions`, and `--layerOutputTypes` are absent.

Production source changes completed in this round:

- `quantization/precision/qdq_inserter.py` now makes the explicit graph fully typed: FP16 parameterized/functional compute casts, INT8-compute/FP16-output casts, FP16 merge casts, and dtype-closure casts for type-preserving, arithmetic, comparison, `Where`, and constant-producing paths. The graph contract and every cast class enter Q/DQ metadata/hash.
- `quantization/precision/activation_boundary.py` now follows a unique pre-activation chain through exported `Transpose/BatchNormalization/Reshape/...` nodes. This fixed PFN Linear from raw `MatMul` output Q/DQ to the true post-BN/post-ReLU semantic tensor.
- `quantization/reports/qdq_boundary.py` now rejects raw output Q/DQ before a transitive ReLU and uses the same weighted-compute predicate as structure/precision validation. TensorRT may attach a canonical identity to both a preparation `kgen` and the real GEMM; only the latter counts as weighted execution.
- `search/stage2/lidar_pyramid_real_evaluator.py` resolves fused merge layers by exact branch-input tensor sets when TensorRT omits the Add name, and preserves the original engine-build failure instead of masking it with a missing layer-info error.
- `search/integration/lidar_pyramid_context.py` and `search/baselines/original_engines.py` now distinguish Legacy-matched FP16 profile choices from true legality. `encoder_m1.pillar_vfe.pfn_layers.0.linear` and `pyramid_backbone.single_head_2` may be INT8 in the maximal parameterized profile; the initializer-free affine-grid functional MatMul stays mapped FP16 geometry compute.
- `search/stage2/{trt_modelopt,trt_build_worker}.py` write a fail-closed build environment manifest and reject any Python/nvcc/GCC/G++ path outside `modelopt`.
- `scripts/strongly_typed_parser_diagnostic.py` provides a parser-only diagnostic that does not qualify as acceptance. `scripts/run_matched_coverage_baseline.py` now supports fresh `ST69-ENT` production gates.
- Regression tests cover strong/weak builder flag exclusion, graph dtype closure, PFN transitive semantic boundary resolution, kgen-vs-GEMM report matching, per-channel axes, protected functional MatMul typing, search-space protection semantics, and fused merge realization.

Parser debugging preserved three non-accepted diagnostic directories. The first exposed Half/FP32 Sigmoid arithmetic, the second Equal Half/Float, and the third Where branch Float/Half. The final diagnostic engine parsed and realized 67/3, after which all fixes were exercised again through the formal production path rather than accepted from the diagnostic script.

The fresh maximal-parameterized production artifact is:

`outputs/H800_strongly_typed_explicit_qdq_20260717_0018/ST69_ENT_semantic_pfn_boundary_retry2/`

It uses fresh fixedK29696 train200 TensorRT EntropyCalibration2, per-channel weights, semantic output boundaries, FP16 merge contracts, and a fresh strongly typed engine. Its engine SHA256 is `96902772ae7fbcb4612816dddf14490577f9cd1cabcff8116987b1f2e24dff7a`; Q/DQ ONNX SHA256 is `ed43038b0e99e401b80230610ba67e594a58e0db58a4faa0cdfd0f5eecf8a298`.

All structural audits pass:

- all-keep, `pruned_unit_count=0`, physical/checkpoint identity retained;
- canonical structure `70/70`, unmapped `0`;
- canonical requested/realized precision `69 INT8 / 1 FP16`, mismatch `0`;
- the sole FP16 canonical entry is parameter-free `pyramid_backbone.functional_affine_grid_matmul` geometry compute;
- merge realization `20/20` FP16 contracts, zero issues;
- boundary audit passed, zero issues;
- PFN Q input and activation-scale owner are both `/pillar_vfe/pfn_layers.0/Relu_output_0`; resolution is `post_relu_semantic_boundary_via_unique_pre_activation_chain`.

Gates and full validation:

| route | coverage | frames/skips | AP@0.3 | AP@0.5 | AP@0.7 | mAP | p50 ms |
|---|---|---:|---:|---:|---:|---:|---:|
| ST69-ENT 10-frame | 69/1 | 10/0 | 0.814596 | 0.814596 | 0.649049 | 0.759414 | 2.815299 |
| ST69-ENT 200-frame | 69/1 | 200/0 | 0.714310 | 0.666118 | 0.451087 | 0.610505 | 3.067919 |
| ST69-ENT full | 69/1 | 1789/0 | 0.685840 | 0.641624 | 0.448974 | 0.592146 | 3.109113 |
| ST67-ENT full | 67/3 | 1789/0 | 0.731074 | 0.687417 | 0.482080 | 0.633524 | 2.976560 |
| Legacy implicit L67 | 67/3 | 1789/0 | 0.704774 | 0.663810 | 0.470345 | 0.612976 | 2.363873 |

The full runs use the same fixed manifest hash `e5cbece0ceaf2ac1b2c47305a3fa3bdc5f616baa1ce86b0754ba565a013ac463`, warmup `200`, reset after warmup, 1789 evaluated/latency frames, and zero skips. ST69 full evidence is in `outputs/H800_strongly_typed_explicit_qdq_20260717_0018/ST69_ENT_semantic_pfn_boundary_retry2_full1789/`; the source engine hash and all prerequisite audit reports were checked before evaluation and no deployment artifact was rebuilt.

Independent verdicts, superseding any earlier single `trusted/equivalent` wording:

- strongly typed chain implementation: `passed`;
- ST69 structure/precision/merge/boundary realization: `passed`;
- ST69 coverage equivalent to Legacy 67/3: `false` (`69/1` versus `67/3`);
- ST69 scale recipe equivalent to Legacy: `false`;
- ST69 accuracy equivalent to Legacy: `false` (mAP delta `-0.020830`, threshold `0.005`);
- ST69 latency equivalent to Legacy: `false` (p50 delta `+0.745240 ms`, ratio `1.3153x`);
- ST69 trusted explicit baseline: `false` because full mAP `0.592146 < 0.60`;
- ST67 coverage equivalent to Legacy: `true`, but accuracy equivalent remains `false` (mAP delta `+0.020548`) and latency equivalent remains `false` (`1.2592x`).

Quantizing the two Legacy-profile exceptions causes ST69 versus ST67 deltas of `-0.041378` mAP and `+0.132553 ms` p50. This shows they were Legacy coverage choices rather than unmapped or unsupported layers, but including them is not accuracy-safe under the current recipe. The old maximal result around `0.5098` remains a superseded raw-boundary/weak-typing negative control; it must not be compared as the current ST69 implementation.

Validation after the final code changes:

```bash
pytest -q tests/test_formal_packages_cpu.py tests/test_search_baseline_engines.py \
  tests/test_search_quantization_groups.py tests/test_search_strongly_typed_merge_realization.py
# 88 passed
git diff --check
```

The next phase is the requested legal domain-width pruning search and joint weight Taylor objective. GA/Pareto has still not been launched. The primary optimization remains normalized joint weight Taylor loss under a hard BOPS budget; physical parameter retention is metadata/tie-break only, not a primary scalar objective.

Completed: 2026-07-17 01:16:22 CST

---

## Round 9 — legal domain-width genes, joint Taylor objective, and greedy framework

This round implements the new Stage-1 search coordinates and the first production integration smoke. No GA, greedy Stage-2 deployment, Pareto experiment, physical pruning, ONNX export, calibration, Q/DQ build, TensorRT engine build, or validation-set evaluation was launched.

The real lidar_pyramid static audit ran on physical H800 GPU 6 after explicit `conda activate univ2x-opt`. Its artifact is:

`outputs/H800_domain_width_search_space_audit_20260717_015344/`

The audit used one Fisher batch as an integration smoke, not as the final eight-batch search ranking. It found:

- tracer atomic coupled units: `7383`;
- safe formal atomic units admitted to physical pruning search: `6912`;
- root/axis pruning domains: `24` (`8` dense, `16` regular grouped);
- nontrivial legal retained-width genes: `21`; three other domains have only the original all-keep width;
- total enumerated legal width choices: `300`;
- precision genes: `69`, all parameterized weighted layers, all independently selectable;
- hard-protected parameterized precision groups: `0`;
- multi-member force-same-precision groups: `0`;
- pruning scopes reused as precision groups: `0`;
- the 70th canonical compute entry is mapped `pyramid_backbone.functional_affine_grid_matmul`, an initializer-free geometry `torch.bmm` group with two runtime operands; it is not a precision gene and remains FP16 in both Legacy and explicit engines.

This corrects an obsolete wording: PFN Linear and `pyramid_backbone.single_head_2` are the two FP16 exceptions in the Legacy-matched 67/3 recipe, but they are not currently hard-protected search layers. ST69 proved both can be explicitly INT8, although jointly enabling them is not accuracy-safe under the current entropy recipe.

Production search changes:

- `search/candidate.py`, `search/canonicalization.py`, and `search/hashing.py` add stable retained-domain-width genes and include their exact expansion identity in genotype/phenotype/deployment hashes.
- `search/pruning_space/local_domains.py` replaces free atomic-bit choice with one legal retained-width gene per root/axis domain. Dense widths are aligned to 4 before search. Regular grouped Conv keeps the original group count, uses only the safe channels/group set `{4,8,16,32,64,128,256,512}`, prunes equal counts per group, and freezes exact group keep/prune maps. Width expansion is the final physical atomic mask; there is no later alignment repair or reranking.
- `search/pruning_space/domain_importance.py` computes the fixed, precision-independent pruning ranking with `sum(abs(g*(-w)) + 0.5*E[g^2]*(-w)^2)`. Overlapping parameter slices are unioned so one tensor element is not counted twice.
- all GA operators now mutate/crossover adjacent legal widths rather than atomic bits. The old priority-root/max-96 cap is bypassed in domain-width mode.
- `search/proxy/joint_weight_taylor.py` implements the combined perturbation once: pruned elements use `delta_w=-w`; retained elements use production-layout-aware per-channel `Q_p(w)-w`; the score is `sum(abs(g*delta_w)+0.5*E[g^2]*delta_w^2)` normalized by full searchable-weight removal Taylor mass. Activation Taylor is excluded.
- `search/proxy/{candidate_perturbation,gpu_batch_proxy,batch_channel_resolver}.py` add Conv/Linear axis-0 and ConvTranspose axis-1 per-channel simulation plus a CUDA batched joint-Taylor path. Domain candidates generate exact candidate-level channel masks instead of allocating a prohibitive `6912 atomic x layer x channel` dense action tensor.
- `search/proxy/objective.py` adds `joint_weight_taylor_hard_bops`: minimize normalized joint Taylor loss subject to `R_BOPS_vs_original_FP32 <= target`. Parameter retention is report-only by default; its coefficient is zero.
- `search/greedy/engine.py` adds the robust comparison framework. Starting from all-keep highest precision, every iteration batches all one-step legal neighbors (one adjacent width decrease or one precision downgrade), recomputes their current marginal cost, and selects minimum incremental joint-Taylor loss per positive BOPS reduction. One monotonic path snapshots the first feasible unique candidate for each target `{0.05,0.10,0.15,0.20,0.25,0.30}`.
- `search/orchestration/lidar_pyramid_search.py` now supports `search.method: ga|greedy`. GA evaluates repaired/expanded Top-5 candidates for 500 frames per budget round, chooses one round winner, and re-evaluates every unique round-winner engine for 1789 frames without rebuilding physical/ONNX/calibration/QDQ/engine artifacts. Greedy deploys only each unique final budget candidate and evaluates it directly for 1789 frames.
- `search/stage2/lidar_pyramid_real_evaluator.py` adds an evaluation-only round-winner path that verifies the existing engine hash and explicitly records zero physical/ONNX/calibration/QDQ/engine rebuilds.
- new formal configs: `search/configs/lidar_pyramid_h800_domain_width_joint_{ga,greedy}.yaml`; both pin physical GPU 6 through context `device:auto`, the modelopt TensorRT root, fixedK29696 train200 entropy calibration, strongly typed deployment, hard BOPS targets, 500-frame GA Stage-2, and 1789-frame full validation.
- `scripts/audit_domain_width_search_space.py` builds and records the real production search space without running a search.

Environment note: commands that inherit a login shell can still trigger a broken base-prefix Conda cross-compiler activation. The verified isolation command is a non-login shell followed by explicit `source .../conda.sh && conda activate modelopt`. It resolves Python/nvcc/GCC/G++ entirely under `/home/lixingfeng/miniconda3/envs/modelopt`, TensorRT is `10.9.0.34`, nvcc is `11.8.89`, and GCC/G++ are `11.2.0`. No system compiler or CUDA was used.

Tests completed in `univ2x-opt`:

```bash
PYTHONPATH=. pytest -q tests/test_search_greedy_budget.py \
  tests/test_search_domain_width_genes.py \
  tests/test_search_gpu_batch_integration.py \
  tests/test_search_stage2_physical_validation.py \
  tests/test_two_stage_joint_search.py
# 35 passed
python -m py_compile <changed search modules and audit script>
git diff --check
```

Pending before search execution: real GPU6 batched-proxy construction/evaluation smoke using the 6912-unit domain space, final broader regression suite, handoff/reproduction command refresh, and a clean Git commit. The one-batch ranking artifact is diagnostic only and must not be reused as the final eight-batch Fisher ranking.

Completed: 2026-07-17 01:54:58 CST

---

## Round 10 — CUDA batched proxy acceptance, exact physical parameter accounting, and search delivery gate

This round closes the implementation/verification gate before any real GA or
greedy deployment search.  It did **not** run GA, greedy Pareto evaluation,
physical pruning, ONNX export, calibration, Q/DQ insertion, TensorRT engine
build, or validation-set AP evaluation.  Existing H800 artifacts were not
overwritten.

The production CUDA proxy was built from the real lidar_pyramid checkpoint,
the 7383 traced atomic units, the 6912 safe formal units, the 24 legal
root/axis domains and one diagnostic Fisher batch on physical H800 GPU 6.  The
final performance artifact is:

`outputs/H800_domain_width_cuda_proxy_audit_20260717_021305/`

The exact acceptance facts are:

- `proxy_backend=cuda_batched` and `scalar_evaluate_call_count=0` for a random
  population of 128 legal domain-width/precision candidates;
- no dense `6912 x layer x channel` atomic action tensor is allocated;
- all-keep FP32 has exact `L_joint_weight_taylor=0`, `R_BOPS=1`, and physical
  parameter retention `1`;
- random legal candidates span `R_BOPS=0.0568479..0.3784325` and joint Taylor
  loss `0.2643069..0.4249282`;
- the small two-candidate batch reports about `156.13 candidates/s`; the full
  128-candidate random population reports about `6.35 candidates/s`,
  `20.155 s` CUDA-event time and `838,445,056` peak allocated bytes.  The full
  population cost is dominated by exact layer/shape mask and bilinear Taylor
  evaluation; it remains a true GPU batch and never calls the scalar path.

Physical parameter-retention accounting was corrected after the real audit
showed that 896 parameters outside the virtual weighted/prunable layer universe
were being omitted from both sides of the ratio.  They are structurally
constant, so production now carries them unchanged in every candidate.  The
fresh verification artifact is:

`outputs/H800_domain_width_search_space_audit_20260716_111741_paramfix_retry/`

It records all-keep `parameter_count_base=5,464,791`,
`parameter_count_after=5,464,791`, `R_parameter_retention=1`, and
`constant_untracked_parameter_count=896`.  A width-64-to-60 diagnostic action
reports `parameter_count_after=5,460,175` and physical pruning rate
`0.0008446574`.  The failed predecessor directory ending in `_paramfix` is a
launcher diagnostic only: setting `CUDA_VISIBLE_DEVICES=6` while the formal
context requested physical `cuda:6` renumbered the visible GPU and caused
`invalid device ordinal`; the accepted retry leaves `CUDA_VISIBLE_DEVICES`
unset and uses the configured physical GPU ID.

The finalized Stage-1 objective is deliberately not a four-term weighted
mixture:

```text
delta_w = -w                              for physically pruned elements
delta_w = Q_p(w) - w                      for retained quantized elements
T_joint = sum(abs(g * delta_w) + 0.5 * E[g^2] * delta_w^2)
L_joint = T_joint / (T_remove_all_searchable + epsilon)

minimize L_joint
subject to R_BOPS(original strict FP32) <= target
```

Activation Taylor and SQNR are absent from this new objective.  Physical
parameter retention is report-only in the GA objective (`epsilon=0`) and is at
most a deterministic lexicographic tie-break in the greedy comparison.  The
greedy tie-break now uses `R_parameter_retention`, not mixed-bit model size.
This directly answers the pruning-versus-quantization concern: a no-pruning
candidate is allowed when quantization reaches a budget with lower predicted
task loss; physical pruning is not forced by an arbitrary scalar reward.  The
hard BOPS targets remain `{0.05,0.10,0.15,0.20,0.25,0.30}`.

The earlier “two protected parameterized layers plus one unmapped MatMul”
description must not be reused.  The current production search has 69 learned
precision genes, zero hard-protected learned groups, and zero unresolved
weighted mappings.  PFN Linear and `pyramid_backbone.single_head_2` are only
the two FP16 exceptions selected by the Legacy-matched 67/3 recipe; ST69 proved
that both can be requested and realized as INT8, although doing so is not
accuracy-safe under the current recipe.  The 70th canonical compute entry,
`pyramid_backbone.functional_affine_grid_matmul`, is fully mapped parameter-free
geometry `torch.bmm` with two runtime inputs.  It has no weight initializer and
therefore no learned precision gene; both Legacy and explicit engines realize
it in FP16.

The FP16 residual/concat merge contract means branch compute genes remain
independent.  Every merge input is DQ/cast to FP16, Add/Concat and its declared
post-merge semantic activation execute in FP16, and a downstream INT8 layer
may insert a new Q at that calibrated post-merge tensor.  It does not force all
layers in the participating branches to share FP16.

Final source additions/changes for this phase include:

- domain-width genotype/canonical/hash and all GA operators under
  `search/{candidate,canonicalization,hashing,ga}/`;
- legal dense/grouped domains and fixed pruning-only Taylor ranking under
  `search/pruning_space/`;
- joint weight Taylor, exact physical parameter accounting, layout-aware
  per-channel simulation and CUDA batched scoring under `search/proxy/`;
- robust one-step marginal-loss/BOPS greedy search under `search/greedy/`;
- GA/greedy orchestration, target-independent proxy caching, 500-frame GA
  Top-5, unique 1789-frame round-winner reuse, and greedy unique-full-validation
  policy under `search/orchestration/` and `search/stage2/`;
- formal configs
  `search/configs/lidar_pyramid_h800_domain_width_joint_{ga,greedy}.yaml`;
- real static/CUDA audit entrypoint
  `scripts/audit_domain_width_search_space.py`;
- focused tests for legal width expansion, grouped frozen maps, scalar/GPU
  agreement, exact constant-parameter accounting, greedy budgets, GA dedup and
  evaluation-only full-validation reuse.

Verified environments in a non-login shell with explicit activation:

```text
univ2x-opt Python: /home/lixingfeng/miniconda3/envs/univ2x-opt/bin/python
modelopt Python:   /home/lixingfeng/miniconda3/envs/modelopt/bin/python
modelopt nvcc:     /home/lixingfeng/miniconda3/envs/modelopt/bin/nvcc (11.8.89)
modelopt gcc/g++:  /home/lixingfeng/miniconda3/envs/modelopt/bin/{gcc,g++} (11.2.0)
TensorRT root:     /home/lixingfeng/UniAD_examine/TensorRT-10.9_x86_cu118
TensorRT Python:   10.9.0.34
```

Final regression and syntax checks:

```bash
source /home/lixingfeng/miniconda3/etc/profile.d/conda.sh
conda activate univ2x-opt
PYTHONPATH=. pytest -q $(rg --files tests | \
  rg '^tests/(test_search.*\\.py|test_two_stage_joint_search\\.py|test_formal_packages_cpu\\.py)$' | sort)
# 223 passed, 3 non-failing dependency warnings
python -m py_compile <all changed Python files>
git diff --check
```

Reproduction/next-stage commands are recorded but were intentionally not run:

```bash
# Real search-space and GPU scorer audit only
source /home/lixingfeng/miniconda3/etc/profile.d/conda.sh
conda activate univ2x-opt
unset CUDA_VISIBLE_DEVICES
PYTHONPATH=. python scripts/audit_domain_width_search_space.py \
  --config search/configs/lidar_pyramid_h800_domain_width_joint_ga.yaml \
  --output-dir outputs/H800_domain_width_audit_<timestamp> \
  --fisher-batches 8 --population-smoke-size 128

# Formal GA (do not run until the implementation checkpoint is accepted)
PYTHONPATH=. python -m search.cli \
  --config search/configs/lidar_pyramid_h800_domain_width_joint_ga.yaml \
  --output-root outputs --gpu-id 6

# Formal greedy comparison (same acceptance restriction)
PYTHONPATH=. python -m search.cli \
  --config search/configs/lidar_pyramid_h800_domain_width_joint_greedy.yaml \
  --output-root outputs --gpu-id 6
```

Current gate: production implementation and static/CUDA integration are ready
for source synchronization; actual GA/greedy Pareto experiments remain
`not_yet_run` and require the next explicit execution decision.

Completed: 2026-07-17 02:20:03 CST

---

## H800 round: formal greedy deployment, GPU evaluation protocol, and multi-GPU scheduling

This round continued from the existing H800 artifacts; it did not retrace the
old explicit-Q/DQ root cause and did not launch GA.  Formal Stage-1 and greedy
artifacts are now real rather than planned:

- formal CUDA-batched Stage-1 gate:
  `outputs/H800_domain_width_formal_gate_20260717_030521`;
- 7,383 traced atomic units, 6,912 safe units, 24 local domains, 21
  non-trivial legal domain-width genes, and 69 precision genes;
- 1,024 candidates evaluated in eight CUDA batches with
  `scalar_evaluate_call_count=0`; measured throughput was about 7.015
  candidates/s;
- formal greedy run:
  `outputs/h800_domain_width_joint_greedy_20260717_030912`;
- 204 greedy steps and 16,915 unique marginal actions reached all six BOPS
  targets.  The redundant phenotype copy was removed from the proxy cache,
  reducing it from about 2.1 GB to about 21 MB without changing cache identity.

Strongly-typed FP32 uncovered a real graph-typing omission: FP16 upstream
tensors could feed an explicitly FP32 Conv.  Production Q/DQ insertion now
adds FP32 casts only to dynamic compute operands and never casts weight
initializers.  QDQ policy identity was advanced, the strong-typing records are
hashed, and the strict FP32 engine now builds, deserializes, and validates.  A
fused Concat that has no standalone EngineInspector row is accepted only when
the strongly-typed ONNX graph proves all branches have the declared FP16 merge
casts.  Exact engine reuse now checks Q/DQ ONNX hash, builder config, TensorRT
root, plugin hash, engine hash, and precision/structure validation before a
cache hit is allowed.

Evaluation was moved to protocol
`fixed-manifest-gpu-postprocess-workers8-v3`:

- each formal evaluation DataLoader uses exactly eight workers;
- `persistent_workers=true`, `prefetch_factor=2`, pinned memory, and one Torch
  thread per loader worker are enabled;
- rotated NMS and AP IoU matching are CUDA-backed and fail closed when the
  HEAL CUDA extension is unavailable;
- only range filtering after NMS and final VOC curve reduction remain on CPU;
- the protocol version, backend audit and DataLoader settings enter the
  request/result and evaluation cache identity, so older CPU evaluations are
  rejected.

The new protocol has been verified on GPU 6 with real 1,789-frame results:

```text
strict FP32: 1789/1789, skip=0, mAP=0.7366637934, p50=8.091868 ms
strict FP16: 1789/1789, skip=0, mAP=0.7366280976, p50=5.423147 ms
greedy candidate 01ad532f...:
             1789/1789, skip=0, mAP=0.7365685220, p50=6.881644 ms
```

All three result JSON files report `dataloader_num_workers=8`, persistent
workers, GPU AP IoU, and a passing CUDA postprocess audit.  The greedy process
is continuing serially on GPU 6 for the remaining unique BOPS candidates; the
completed artifacts must be reused rather than rebuilt.

Future GA parallelism has also been implemented but has not yet been launched:

- Stage-1 populations are split into deterministic contiguous shards across
  idle GPUs and scored concurrently by cloned CUDA proxy tables while
  preserving candidate order;
- Stage-2 assigns candidates round-robin into one sequential queue per GPU;
  queues run concurrently across GPUs, preventing simultaneous engine/eval
  jobs on the same device;
- duplicate candidate hashes reuse the first cross-round deployment result;
- final round-winner latency validation remains serial to avoid contention;
- the GA configuration currently audits candidate GPUs `[1,3,4,5,6,7]` and
  selects only devices with at least 60,000 MiB free and at most 10% GPU
  utilization.

Focused tests after these changes:

```text
tests/test_search_evaluation_provider.py
tests/test_search_gpu_batch_integration.py
tests/test_search_multi_gpu_orchestration.py
12 passed
```

New or materially changed production files in this round include:

- `quantization/config.py` and `quantization/precision/qdq_inserter.py`:
  explicit FP32 dynamic-input closure and hashed strong-typing evidence;
- `search/baselines/original_engines.py`: strict FP32 validation with the
  mapped parameter-free functional BMM exception;
- `search/integration/evaluation_provider.py` and
  `search/integration/evaluation_worker.py`: GPU postprocess protocol and the
  eight-worker DataLoader;
- `search/stage2/lidar_pyramid_real_evaluator.py`: protocol-safe eval cache,
  exact engine reuse, and merge realization validation;
- `search/proxy/{batch_channel_resolver,gpu_batch_proxy}.py` and
  `search/orchestration/lidar_pyramid_search.py`: multi-GPU Stage-1/Stage-2
  execution;
- `search/configs/lidar_pyramid_h800_domain_width_joint_ga.yaml`: idle-GPU
  pool policy;
- focused tests including the new evaluation-provider, exact-engine-reuse and
  multi-GPU scheduling suites.

Current next gate: let the in-progress greedy full-validation process finish,
run the complete search regression set and `git diff --check`, then create the
required clean source commit.  GA remains paused until that gate passes.

Completed checkpoint: 2026-07-17 04:13:40 +0800 CST

---

## H800 round completion: six-budget greedy full validation

The resumed formal greedy process completed successfully on physical GPU 6.
Every candidate used the same 1,789-frame validation manifest, 200 warmup
frames with iterator reset, zero skipped frames, CUDA rotated NMS/AP IoU, and
an eight-worker persistent DataLoader.  The final measured frontier is:

| BOPS target | realized R_BOPS | parameter prune | mAP | AP@0.70 | forward p50 ms |
|---:|---:|---:|---:|---:|---:|
| 0.05 | 0.049994 | 0.385670 | 0.699808 | 0.563222 | 3.170403 |
| 0.10 | 0.098478 | 0.220708 | 0.736742 | 0.602197 | 3.295959 |
| 0.15 | 0.149413 | 0.210133 | 0.736825 | 0.602484 | 4.627601 |
| 0.20 | 0.196774 | 0.210133 | 0.736688 | 0.602271 | 4.991628 |
| 0.25 | 0.249080 | 0.210133 | 0.736569 | 0.601983 | 6.881644 |
| 0.30 | 0.282141 | 0.210133 | 0.736558 | 0.602197 | 6.866993 |

Strict references under the identical protocol were mAP 0.736664 / p50
8.091868 ms for FP32 and mAP 0.736628 / p50 5.423147 ms for FP16.  The best
greedy F2 candidate is BOPS target 0.10, hash
`0a0f2015b79da78289b9a1f3e9a44473634b89f2294c75f788ffcc10ada1d1e7`,
with F2 0.121552.  The 0.05 endpoint is valid but is not accuracy-equivalent:
its mAP loss is about 0.03686 and should remain a low-resource Pareto point.

Artifact source:
`outputs/h800_domain_width_joint_greedy_20260717_030912`.  Existing physical,
ONNX, calibration, Q/DQ, engine, and evaluation artifacts in that directory
are valid and must not be rebuilt without an identity mismatch.  The source
tree now also writes the explicit greedy best candidate into future
`full_validation_results.json` and `best_candidate.json` outputs.

To prevent duplicate strict baseline engine builds between GA 500-frame
screening and final 1,789-frame validation, the full-validation evaluator now
accepts proven earlier baseline engine roots.  It checks Q/DQ hash, builder
configuration, TensorRT root, plugin hash, engine hash, and validation reports,
then hard-links (or copies across filesystems) the exact engine build into the
fresh evaluation directory.  It refuses to overwrite any occupied target
artifact.

Final source regression after all changes:

```bash
source /home/lixingfeng/miniconda3/etc/profile.d/conda.sh
conda activate univ2x-opt
PYTHONPATH=. pytest -q $(rg --files tests | \
  rg '^tests/(test_search.*\\.py|test_two_stage_joint_search\\.py|test_formal_packages_cpu\\.py)$' | sort)
# 231 passed, 3 dependency warnings
git diff --check
```

GA has still not been launched in this checkpoint.  The next executable stage
is the configured six-GPU formal GA; Stage-1 shards CUDA proxy batches across
idle GPUs, Stage-2 runs one serial candidate queue per GPU with eight
DataLoader workers per evaluation, and final latency validation remains
serial.

Completed checkpoint: 2026-07-17 04:42:15 +0800 CST

---

## H800 GA preflight correction: compact domain-width genotype

The first six-GPU GA attempt is preserved at
`outputs/h800_domain_width_joint_ga_20260716_134325` and was intentionally
stopped before generation 0 completed.  Runtime evidence showed all six proxy
tables allocated on GPUs `[1,3,4,5,6,7]`, but CUDA utilization remained idle
for more than nine minutes while CPU bookkeeping processed each candidate.
The root cause was a representation defect: the new 21-value domain-width
genotype still serialized 7,383 redundant all-keep atomic mask entries in
every raw candidate.  These entries were not searchable and did not affect
the expanded phenotype, but they polluted repair, hash, deduplication and GA
operator work.

Production genotype handling is now compact for any search space with legal
pruning domains:

- `CandidateGenotype.pruning_genes` remains empty;
- only `pruning_width_genes` participates in the pruning genotype identity;
- canonicalization still deterministically expands the fixed Taylor ranking
  into the exact atomic `pruned_unit_ids` mask;
- the legacy redundant all-one representation is accepted on load and
  canonicalizes to exactly the same phenotype;
- initialization, immigrants, crossover and mutation no longer reconstruct
  the redundant 7,383-value coordinate.

Focused regression: 30 tests passed, including a new exact phenotype equality
test between the legacy redundant representation and the compact genotype.
The aborted run directory was not deleted or reused; a new formal GA directory
must be created after committing this correction.

Completed checkpoint: 2026-07-17 04:55:07 +0800 CST

---

## H800 GA preflight correction: batched explicit-mask encoding

The second preflight directory
`outputs/h800_domain_width_joint_ga_20260716_135612` is also preserved and was
stopped during the first multi-GPU proxy call.  Compact raw genotypes removed
hash overhead, but the exact channel-mask encoder still issued a separate
Torch advanced-index assignment for every pruned atomic unit.  With thousands
of units per phenotype, all six CUDA scorers waited on millions of fine-grained
CPU tensor operations.

`TorchBatchedProxyScorer._encode` now:

- caches atomic-id and flattened layer/channel effect indices once per scorer;
- writes all pruned action indices in one operation per candidate;
- merges all output-channel effects into one flattened mask assignment per
  candidate;
- does the same for input-channel effects;
- preserves overlapping/duplicate effect semantics because the destination is
  Boolean.

The multi-device clones own independent cache dictionaries, so lazy cache
construction cannot race across scoring threads.  A new exact mask test covers
overlapping output effects and multi-layer input/output effects.  Focused
proxy/domain tests: 21 passed.  A new formal GA directory is required; neither
preflight directory may be reused as a completed search.

Completed checkpoint: 2026-07-17 05:02:51 +0800 CST

---

## H800 GA preflight correction: shared full-manifest prefix protocol

The third GA directory `outputs/h800_domain_width_joint_ga_20260716_140356`
proved the Stage-1 fixes: round 0 completed five generations with 36 CUDA
batches, zero scalar calls, and about 224.7 candidates/s in the final batch.
It was then stopped before candidate Stage-2 because the strict FP32 reference
evaluation reported
`eval_manifest_count_mismatch:warmup=200!=200:eval=1789!=500`.

The shared manifest is intentionally sized for final full validation.  A
500-frame round candidate must use the first 500 evaluation IDs in exactly the
same order, not require a separate 500-entry manifest.  Production evaluation
protocol v4 now:

- accepts a manifest with at least the requested warmup/evaluation counts;
- validates uniqueness on the full available manifest;
- selects deterministic ordered prefixes of exactly the requested sizes;
- fails when the manifest is too short;
- records available counts and `deterministic_prefix` policy in evaluation
  results;
- includes the new protocol version in evaluation/deployment cache identity.

The earlier greedy 1,789-frame v3 artifacts remain valid historical results;
they are not relabeled.  New GA evaluations use
`fixed-shared-manifest-prefix-gpu-postprocess-workers8-v4`.  Focused manifest,
cache and Stage-2 tests passed.  The failed FP32 evaluation is preserved and
must not be interpreted as an engine/model failure.

Completed checkpoint: 2026-07-17 05:09:57 +0800 CST

---

## H800 multi-GPU Stage-2 correction: thread-safe ONNX/cache publication

The resumable formal directory
`outputs/h800_domain_width_joint_ga_20260716_141052` completed round-0
Stage-1 with 36 CUDA batches, zero scalar evaluations, and approximately
240.58 candidates/s.  Its six-GPU Stage-2 launch then exposed a process-level
PyTorch ONNX exporter race: one candidate completed, three independent
candidates failed with `OnnxExportError`/`AssertionError`, and a fifth was
still calibrating when the run was deliberately stopped.  This is a
concurrency defect rather than an invalid candidate or model-structure result.

Production corrections:

- `search/stage2/lidar_pyramid_real_evaluator.py` serializes only the legacy
  PyTorch ONNX export and shared ONNX cache publication with a process-wide
  reentrant lock.  Calibration, Q/DQ generation, TensorRT builds and
  evaluation remain parallel across GPUs.
- `search/orchestration/lidar_pyramid_search.py` gives all per-GPU evaluators
  the same artifact and real-evaluation cache objects, so identical physical
  models and deployment identities are reused across worker queues.
- `search/cache/artifact_cache.py` and `search/cache/proxy_cache.py` now guard
  in-memory indices and JSONL append operations with reentrant locks.
- `tests/test_search_cache_reuse.py` adds a 64-entry, eight-thread cache-writer
  regression; `tests/test_search_multi_gpu_orchestration.py` verifies the
  ONNX export critical section.

The existing directory is the required resume source: completed Stage-1 and
the successful candidate must be reused, while only the failed/incomplete
Stage-2 candidates are rebuilt.  Focused cache, multi-GPU and physical
validation tests: 17 passed.  Evaluation DataLoader policy remains eight
workers per evaluator under protocol v4.

Completed checkpoint: 2026-07-17 05:26:00 +0800 CST

---

## H800 formal GA result and GA-vs-greedy root-cause audit

The resumed formal directory
`outputs/h800_domain_width_joint_ga_20260716_141052` ultimately completed all
six BOPS rounds, 30 Stage-2 candidate evaluations, and six deduplicated
round-winner full validations.  Every successful evaluation used eight
DataLoader workers, GPU NMS/AP IoU, 0 skipped frames, and the fixed shared
manifest.  The final GA winner was not usable: full-validation mAP was
`0.0173567420` versus strict-FP32 `0.7366637934`.

Full-validation mAP by GA target/round winner:

- target 0.30: `0.0033104460`;
- target 0.25: `0.0045920567`;
- target 0.20: `0.0040900210`;
- target 0.15: `0.0050685185`;
- target 0.10: `0.0173567420`;
- target 0.05: `0.0`.

These results do **not** show that GA is inherently worse than greedy.  The
primary confirmed cause is corrupted resume/dedup semantics:

- the first, interrupted round-0 Stage-1 produced safe candidates, including
  all-keep/all-FP16 phenotype `0cfe6a69...`;
- that phenotype completed a real 500-frame evaluation with mAP
  `0.7269672958` and p50 `5.4917377 ms`;
- because the round had not yet written a complete Stage-2 summary, `--resume`
  reran Stage-1 instead of loading the existing `stage1_topk.json`;
- `seen_raw_genotypes.jsonl` from the interrupted Stage-1 was then treated as
  an exclusion set rather than as cache/archive evidence, so the previously
  evaluated safe seeds could not re-enter round 0;
- the rerun overwrote round-0 Stage-1 selection with destructive candidates,
  and those candidates became the only warm-start elites for rounds 1--5.

The same directory preserves both old and replacement round-0 config files.
The old Top-5 structures pruned 0--256 units in 0--2 domains; the replacement
Top-5 pruned 1,984--2,728 units in 15 domains.  The raw archive also proves the
difference: the first round-0 search reached F1 `0.0001561564`, whereas the
replacement Top-5 F1 range was `0.136245--0.142645`.

Secondary contributors, which still need correction after resume is fixed:

- 1,017 of 1,024 initial candidates are random immigrants centered near 50%
  width in every domain with uniformly random FP32/FP16/INT8 genes;
- five generations are insufficient for this approximately 90-gene space to
  recover the narrow safe basin found by adjacent-action greedy search;
- hard feasibility accepts severe BOPS undershoot, so candidates may waste
  budget while pruning many early/grouped domains;
- current mutation changes approximately 8% of all width and precision genes
  per child, rather than one adjacent compression action;
- weight-only Taylor omits activation/merge quantization sensitivity and is
  unreliable for simultaneous large perturbations across 13--18 pruning
  domains plus 20--33 INT8 groups.

The controlled greedy run uses the same search space, joint weight-Taylor
proxy, physical pruning, explicit-Q/DQ builder, and evaluator.  It found much
smaller proxy losses and safe structures: at targets 0.30/0.25/0.20/0.15/0.10
its F1 values were `1.64e-5`, `2.59e-5`, `5.64e-5`, `9.77e-5`, `1.81e-4`,
with full mAP approximately `0.7366--0.7368`; at target 0.05, F1 was
`0.002857` and full mAP `0.699808`.  It pruned only 2--4 domains instead of
the GA replacement candidates' 12--18 domains.

Do not run another heavy GA before implementing and testing:

1. immutable round Stage-1 checkpoints loaded directly on partial resume;
2. cache-backed reuse of seen genotypes instead of genotype exclusion;
3. global feasible-frontier seeding for each nested BOPS target;
4. adjacent/monotonic mutation or greedy-frontier warm starts;
5. a pre-Stage-2 assertion that GA proxy quality is competitive with the
   existing greedy reference at the same budget.

No production source was changed during this root-cause audit.  All current
GA/greedy artifacts remain read-only evidence.

Completed checkpoint: 2026-07-17 12:27:00 +0800 CST

---

## BOPS band-feasibility correction and subnet compression audit

The intended Stage-2 admission rule was reconfirmed as a two-sided hard band:
`abs(R_BOPS_vs_original_FP32 - target) <= 0.005`.  Static inspection found
that the current production implementation does **not** implement that rule.
`search/proxy/objective.py`, `search/proxy/gpu_batch_proxy.py`, and the GA
orchestrator all compute only `max(0, R_BOPS - target)`.  Consequently, the
current `hard_feasibility` mode rejects over-budget candidates but accepts
arbitrarily severe undershoot.

Applying the intended band retrospectively changes the formal-run status:

- GA round winners at targets 0.30/0.25/0.20/0.15/0.10 have R_BOPS
  0.201438/0.144210/0.162514/0.127768/0.068686 and must all be rejected;
- the GA target-0.05 winner has R_BOPS 0.049122 and is the only admitted GA
  round winner;
- greedy targets 0.05/0.10/0.15/0.20/0.25 are inside the band;
- greedy target 0.30 has R_BOPS 0.282141 and must be rejected.

Therefore the existing formal GA result is not a budget-matched Pareto run;
five of six round winners are inadmissible under the requested contract.

The current joint objective was also verified exactly.  In
`joint_weight_taylor_hard_bops` mode, feasible-candidate F1 is only normalized
joint weight perturbation Taylor loss:

`sum(|g * delta_w| + 0.5 * E[g^2] * delta_w^2) / full-searchable-removal-mass`,

where pruned elements use `delta_w=-w` and retained quantized elements use
`delta_w=Q_p(w)-w`.  `parameter_retention_tiebreak_epsilon=0.0`, so physical
parameter retention/pruning rate is report-only and contributes no additive
F1 term.  BOPS is intended as the hard resource constraint rather than a
weighted F1 term.  Greedy uses parameter retention only as a late
lexicographic tie-break, not as part of F1.

Full-validation compression audit (parameter baseline 5,464,791; latency
speedup uses each run's strict-FP16 baseline):

| search | target | R_BOPS | band | parameter reduction | parameter x | mixed-weight x | BOPS x | mAP | p50 ms | speedup vs FP16 |
|---|---:|---:|:---:|---:|---:|---:|---:|---:|---:|---:|
| GA | 0.30 | 0.201438 | reject | 43.78% | 1.779x | 3.187x | 4.964x | 0.003310 | 5.398 | 1.009x |
| GA | 0.25 | 0.144210 | reject | 60.55% | 2.535x | 5.010x | 6.934x | 0.004592 | 4.255 | 1.280x |
| GA | 0.20 | 0.162514 | reject | 55.10% | 2.227x | 3.812x | 6.153x | 0.004090 | 4.874 | 1.118x |
| GA | 0.15 | 0.127768 | reject | 39.94% | 1.665x | 3.236x | 7.827x | 0.005069 | 4.357 | 1.250x |
| GA | 0.10 | 0.068686 | reject | 59.98% | 2.499x | 5.659x | 14.559x | 0.017357 | 3.625 | 1.503x |
| GA | 0.05 | 0.049122 | admit | 63.36% | 2.729x | 6.358x | 20.358x | 0.000000 | 3.479 | 1.566x |
| greedy | 0.05 | 0.049994 | admit | 38.57% | 1.628x | 5.458x | 20.002x | 0.699808 | 3.170 | 1.711x |
| greedy | 0.10 | 0.098478 | admit | 22.07% | 1.283x | 3.255x | 10.155x | 0.736742 | 3.296 | 1.645x |
| greedy | 0.15 | 0.149413 | admit | 21.01% | 1.266x | 2.500x | 6.693x | 0.736825 | 4.628 | 1.172x |
| greedy | 0.20 | 0.196774 | admit | 21.01% | 1.266x | 2.333x | 5.082x | 0.736688 | 4.992 | 1.086x |
| greedy | 0.25 | 0.249080 | admit | 21.01% | 1.266x | 2.169x | 4.015x | 0.736569 | 6.882 | 0.788x |
| greedy | 0.30 | 0.282141 | reject | 21.01% | 1.266x | 2.120x | 3.544x | 0.736558 | 6.867 | 0.790x |

`mixed-weight x` is `1/R_Size_vs_original_FP32`; it includes both physical
pruning and mixed weight bit-width.  `BOPS x` is theoretical and must not be
reported as measured latency speedup.  GA full-validation strict-FP16/FP32
p50 references are 5.447545/8.911942 ms; greedy references are
5.423147/8.091868 ms.  Relative to strict FP32, the measured speedup ranges
are 1.651--2.562x for the listed GA winners and 1.176--2.552x for greedy.

No production code was changed in this audit.  Before another heavy search,
the BOPS band must be represented explicitly in CPU scoring, CUDA batched
scoring, selection/repair, greedy budget capture, Stage-2 admission, configs,
and tests; candidates outside the band must never enter Stage-2.

Completed checkpoint: 2026-07-17 13:10:56 +0800 CST

---

## GA hard-band, resume, local-search, and FP32-reference production repair

The user accepted the proposed GA repair package with two explicit exclusions:

- no GA-vs-greedy anti-regression gate (G13), because their search quality is
  intentionally being compared;
- no per-GPU baseline remeasurement/fairness policy (G15).  One original FP32
  reference is measured and shared; different GPUs still build/evaluate
  candidates in parallel, while each GPU's assigned queue remains strictly
  sequential (`build candidate -> evaluate candidate -> next candidate`).

F1/parameter retention uses the selected B policy, not direct 0.8/0.2 scalar
addition.  Joint Taylor is the primary objective.  Only candidates within 5%
of the best feasible Taylor value (plus absolute epsilon `1e-8`) use lower
physical parameter retention as a secondary key.  This avoids a raw
`0.2*R_parameter_retention` term overwhelming Taylor values around
`1e-5--1e-3` while still encouraging modest physical pruning among
accuracy-proxy-equivalent candidates.

Production changes:

- `search/proxy/objective.py` and `search/proxy/gpu_batch_proxy.py` add
  `hard_band_feasibility` with
  `abs(R_BOPS_vs_original_FP32-target) <= bops_tolerance_abs` and expose
  absolute delta, tolerance, violation, and feasibility.  Formal tolerance is
  `0.005`.  F1 remains raw joint Taylor; constraint-first selection is separate.
- `search/ga/ranking.py` adds Taylor-primary epsilon-lexicographic ranking.
  Infeasible candidates are ranked only by distance to the BOPS band so they
  can evolve toward feasibility, but they are never admitted to Stage-2.
- `search/ga/{engine,initialization,mutation,selection}.py` now treats seen
  hashes as cache evidence rather than an exclusion list, accepts budget seed
  candidates, builds 90% of the initial population around greedy-frontier
  seeds, mutates only 1--2 genes per child, uses adjacent precision changes
  (`FP32 <-> FP16 <-> INT8`), preserves semantic block crossover, uses
  budget-seeded immigrants, and supports 15 generations with five-generation
  early stopping after a five-generation minimum.
- the six BOPS targets now use independent GA populations, while sharing the
  expensive proxy/artifact caches.  A proxy-only greedy frontier is generated
  once and its exact/nearest candidate seeds each budget; previous-round bad
  elites cannot contaminate subsequent budgets.
- `search/greedy/engine.py` now captures a budget only inside the same
  two-sided `+/-0.005` band.  Discrete targets skipped by the greedy action path
  are explicitly unreachable; nearest candidates are retained only as GA
  seeds, not accepted as greedy budget winners.
- `search/stage1/repair_selection.py` re-scores repaired phenotypes, re-applies
  band eligibility, rejects a round rather than backfilling fewer than five
  feasible candidates, and selects three exploitation plus two genotype-
  diverse candidates.  Selection roles are recorded.
- resume now uses atomic JSON writes and `round_state.json` phases.  A
  `stage1_complete` checkpoint reloads immutable Top-5 rather than rerunning
  Stage-1.  Objective/hash mismatch fails closed.  Completed raw genotypes may
  re-enter population and hit proxy cache.  Partial Stage-2 resumes through
  the existing layered artifact/evaluation caches.
- `Stage2ObjectiveConfig` now records both accuracy and latency references.
  The H800 formal configs use original strict FP32 for both.  When both
  references are FP32 the evaluator builds/evaluates that baseline once and
  shares it with all Stage-2 GPU workers; strict FP16 is auxiliary only.
- round and full-validation artifacts now report actual physical parameter
  reduction/compression, mixed-weight storage compression versus FP32,
  theoretical BOPS compression versus W32A32, and measured p50 speedup versus
  the original strict FP32 engine.  Theoretical BOPS compression remains
  explicitly distinct from measured speedup.

Formal configuration changes:

- `search/configs/lidar_pyramid_h800_domain_width_joint_ga.yaml`:
  `hard_band_feasibility`, tolerance 0.005, 15 generations, 1--2 local mutation
  actions, five-generation early stopping, independent budgets, greedy-frontier
  warm start, 90% seeded initialization, 3+2 Stage-2 selection, FP32 latency
  reference;
- `search/configs/lidar_pyramid_h800_domain_width_joint_greedy.yaml`:
  identical BOPS band and FP32 latency reference.

Validation completed in `univ2x-opt`:

```bash
PYTHONPATH=.:.. pytest -q tests/test_search*.py tests/test_two_stage_joint_search.py
# 169 passed, 2 dependency deprecation warnings

python -m compileall -q search

PYTHONPATH=.:.. python -m search.cli \
  --config search/configs/lidar_pyramid_h800_domain_width_joint_ga.yaml \
  --output-root /tmp/heal_compress_ga_dryrun --dry-run
# successful; evaluated=0
```

No new formal GA, Stage-2 engine build, or validation evaluation was launched
in this checkpoint.  Existing GA results remain read-only historical evidence
and are invalid for the new budget contract (`invalid_bops_admission_semantics`);
they must not be resumed into the repaired run because their objective manifest
lacks the new hard-band contract.

The next heavy command, after syncing this commit, is:

```bash
source /home/lixingfeng/miniconda3/etc/profile.d/conda.sh
conda activate univ2x-opt
PYTHONPATH=.:.. python -m search.cli \
  --config search/configs/lidar_pyramid_h800_domain_width_joint_ga.yaml \
  --output-root outputs
```

Completed checkpoint: 2026-07-17 14:28:33 +0800 CST

---

## Formal GA runtime serialization fix

The repaired formal search was launched into the new, read-only-preserved run
directory `outputs/h800_domain_width_joint_ga_20260716_233739`.  It completed
real model/Fisher initialization and evaluated about 15,000 CUDA-batched proxy
candidates while constructing the shared greedy frontier, then stopped before
the first GA generation or any Stage-2 build.

The stop exposed a production serialization defect rather than an experiment
failure: `GreedySearchResult.to_dict()` attempted to read
`self.config.bops_tolerance_abs`, although result objects do not own the search
config.  `search/greedy/engine.py` now carries the effective absolute BOPS
tolerance as immutable result provenance and serializes that value directly.
`tests/test_search_greedy_budget.py` exercises the full result serialization so
this path cannot regress unnoticed.

Validation in the explicitly activated `univ2x-opt` environment:

```bash
PYTHONPATH=.:.. pytest -q \
  tests/test_search_greedy_budget.py tests/test_search_ga_band_contract.py
# 6 passed

PYTHONPATH=.:.. pytest -q tests/test_search*.py \
  tests/test_two_stage_joint_search.py
# 169 passed, 2 dependency deprecation warnings

git diff --check
# clean
```

The failed run directory was not deleted or overwritten and is not eligible for
formal result reporting because no GA round completed.  The next launch must
create a new independent output directory.

Completed checkpoint: 2026-07-17 14:42:02 +0800 CST

---

## Completed repaired domain-width GA search and full validation

The repaired formal search completed successfully from production commit
`d767626c3b566d27db2aeb53de6a1e9dbc64869c`:

```bash
source /home/lixingfeng/miniconda3/etc/profile.d/conda.sh
conda activate univ2x-opt
PYTHONPATH=.:.. python -m search.cli \
  --config search/configs/lidar_pyramid_h800_domain_width_joint_ga.yaml \
  --output-root outputs
```

The complete result directory is:

```text
outputs/h800_domain_width_joint_ga_20260716_234248
```

The run returned code 0.  It completed six independent hard-band GA rounds,
30 unique Stage-2 candidate deployments/evaluations (five per budget), six
round-winner full validations, and final-winner selection.  The original FP32
full-validation reference is 1789/1789 frames with zero skips, mAP
`0.7366065145`, AP@0.70 `0.6021423571`, and forward p50 `8.656952 ms`.

Stage-1 execution evidence:

- backend `cuda_batched`, batch size 128, GPU workers `[1,3,4,5,6,7]`;
- 48,316 proxy cache misses/unique evaluated phenotypes and 2,102 cache hits;
- 282 batch evaluator calls, 1,687 GPU batches, and exactly zero scalar
  evaluator calls;
- measured aggregate throughput `319.56 candidates/s`;
- actual generations before convergence/early stop were 8, 15, 14, 9, 6,
  and 6 for targets 0.30 through 0.05, respectively;
- every round admitted exactly five repaired unique candidates inside the
  two-sided `abs(R_BOPS-target) <= 0.005` gate and selected three exploitation
  plus two diversity candidates.

Full-validation frontier (all rows are 1789/1789, zero skips):

| target | actual R_BOPS | abs delta | winner hash | canonical INT8/FP16/FP32 | parameter prune | parameter x | mixed-weight x | BOPS x | mAP | AP@0.70 | p50 ms | speedup vs FP32 | F2 |
|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.30 | 0.304223 | 0.004223 | `cb7b234ae178` | 0/26/44 | 21.01% | 1.266x | 2.014x | 3.287x | 0.736734 | 0.602359 | 7.115 | 1.217x | 0.164368 |
| 0.25 | 0.254517 | 0.004517 | `9a41b6c0bac0` | 1/29/40 | 20.68% | 1.261x | 2.082x | 3.929x | 0.736701 | 0.602170 | 5.485 | 1.578x | 0.126711 |
| 0.20 | 0.204300 | 0.004300 | `386f5ec9f802` | 3/29/38 | 21.35% | 1.271x | 2.373x | 4.895x | 0.736639 | 0.602136 | 5.030 | 1.721x | 0.116206 |
| 0.15 | 0.154526 | 0.004526 | `f7d76246679b` | 10/31/29 | 21.35% | 1.271x | 2.740x | 6.471x | 0.736839 | 0.602607 | 4.920 | 1.760x | 0.113657 |
| 0.10 | 0.104957 | 0.004957 | `702e8db80f45` | 14/47/9 | 22.07% | 1.283x | 3.294x | 9.528x | 0.736747 | 0.602536 | 3.465 | 2.499x | 0.080044 |
| 0.05 | 0.054807 | 0.004807 | `abfefc9d324d` | 32/37/1 | 44.13% | 1.790x | 5.783x | 18.246x | 0.707696 | 0.572820 | 3.139 | 2.758x | 1.228937 |

Here `parameter x` is original FP32 parameter count divided by the physically
materialized parameter count, `mixed-weight x` includes both physical pruning
and selected weight precision, `BOPS x` is W32A32 theoretical BOPS compression,
and `speedup` is measured forward p50 speedup against the same-run strict FP32
engine.  These quantities must not be interchanged.

The configured final F2 winner is the 0.10-budget candidate
`702e8db80f4561f01db09e66e9d8cd3cecd85636b3cdee3348f8a7fc157012e3`.
It preserves full-validation mAP within `+0.000140` of FP32 while providing
9.528x theoretical BOPS compression, 3.294x mixed-weight storage compression,
22.07% physical parameter pruning, and 2.499x measured forward speedup.  Its
identity is:

```text
physical_hash   002a622fa4ce22494012a32d23f379e6050af8ef46ec43b1a162296c154d0eb6
engine_hash     444f64eb6e10496a62a770fea05a926b2ca0add9aa4bf8ebd8795c3a3772b51b
deployment_hash 6d19ad68ae92fd9ad174cbbc1ff7a99b7d367780a4035a4f903d6213e3b1a339
```

Its compact pruning genotype changes exactly three of the 24 legal domain
width genes and performs no late alignment repair:

```text
pyramid_backbone.resnet.layer2.0.conv3::out       256 -> 252
shrink_conv.layers.0.double_conv.0::out            256 -> 68
shrink_conv.layers.0.double_conv.2::out            256 -> 124
```

This expands deterministically to 324 pruned atomic units.  The materializer
ledger contains 33 applied dependency-closed entries, zero repaired entries,
zero merged entries, and zero skipped entries.  Repaired phenotype -> request
-> physical plan equality, grouped keep/prune-map freezing, physical structure,
real forward, and engine structure checks all passed.

Deployment acceptance for every round winner:

- TensorRT 10.9 production command contains `--stronglyTyped`, `--noTF32`, the
  fixedK29696 profiles, and the project PointPillarScatter static plugin;
- requested INT8 weighted-layer count equals realized INT8 count for every
  winner; all canonical engine checks match 70/70 layers with zero unresolved
  mappings;
- precision realization, merge realization (20 merge boundaries), and engine
  structure checks pass without mismatch;
- semantic Q/DQ boundary audit passes with zero raw weighted/Conv-output Q/DQ
  boundaries; all INT8 weights are layout-aware per-channel;
- the final 0.10 candidate realizes 14 INT8, 47 FP16, and 9 FP32 canonical
  weighted layers.  This consists of 69 searched precision genes plus the
  mapped/protected functional MatMul FP16 entry, not an unmapped layer;
- final validation reused the already verified round deployment artifacts:
  six deduplicated winners and `candidate_engine_rebuild_count=0` (also zero
  physical, ONNX, calibration, Q/DQ, and engine rebuild flags per row).

Measured Pareto interpretation:

- 0.15 is the highest-mAP accuracy-safe point (`0.736839`, 6.471x BOPS,
  1.760x speedup);
- 0.10 is the configured balanced winner and the fastest accuracy-safe point;
- 0.05 is the maximum-compression point but loses `0.028911` mAP and should not
  be labeled accuracy-safe;
- on measured mAP/latency, the 0.20, 0.25, and 0.30 points are dominated by the
  0.15 point.  They remain valid budget-specific search results, not failures.

Relative to the prior greedy full-validation frontier, the repaired GA is no
longer generally worse.  Its 0.30 result is valid while the greedy 0.30 result
missed the hard band (`R_BOPS=0.282141`).  At 0.05 the GA improves mAP from
`0.699808` to `0.707696`; at 0.25 it improves mAP from `0.736569` to `0.736701`
and normalized speedup from about 1.176x to 1.578x.  The 0.10--0.20 accuracy
differences are below 0.00005, while the repaired GA has equal or better
same-run-normalized speedup.  Absolute cross-run latency should still be read
with the corresponding run-specific FP32 reference.

Primary machine-readable results:

```text
outputs/h800_domain_width_joint_ga_20260716_234248/final_full_validation_results.json
outputs/h800_domain_width_joint_ga_20260716_234248/final_winner.json
outputs/h800_domain_width_joint_ga_20260716_234248/global_summary.json
outputs/h800_domain_width_joint_ga_20260716_234248/run_manifest.json
```

Completed checkpoint: 2026-07-17 15:52:20 +0800 CST

---

## Separate greedy and GA full-validation latency summaries

The greedy and GA runs measured their strict FP32 references independently.
Their latency values must not be mixed: greedy speedup uses greedy FP32 p50
`8.091868 ms`, while GA speedup uses GA FP32 p50 `8.656952 ms`.  Every row
below is 1789/1789 frames with zero skips.  Sizes use decimal MB
(`1 MB = 10^6 bytes`).  `physical FP32 MB` is the physically pruned parameter
count stored at 32 bits; `mixed-weight MB` additionally applies each selected
FP32/FP16/INT8 weight precision.  Speedup is reference p50 divided by candidate
p50.

Greedy full-validation results:

| budget | actual BOPS | physical FP32 MB | canonical INT8/FP16/FP32 | parameter prune | mixed-weight MB | mixed-weight x | BOPS x | AP30/50/70 | mAP | p50/p90/p99 ms | speedup |
|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---|---:|
| FP32 reference | 1.000000 | 21.859 | 0/0/70 | 0.00% | 21.859 | 1.000x | 1.000x | 0.826195/0.781629/0.602167 | 0.736664 | 8.092/8.782/27.107 | 1.000x |
| 0.30 (invalid) | 0.282141 | 17.266 | 0/28/42 | 21.01% | 10.312 | 2.120x | 3.544x | 0.826042/0.781435/0.602197 | 0.736558 | 6.867/7.347/22.830 | 1.178x |
| 0.25 | 0.249080 | 17.266 | 0/31/39 | 21.01% | 10.077 | 2.169x | 4.015x | 0.826120/0.781603/0.601983 | 0.736569 | 6.882/7.385/22.452 | 1.176x |
| 0.20 | 0.196774 | 17.266 | 2/31/37 | 21.01% | 9.370 | 2.333x | 5.082x | 0.826251/0.781543/0.602271 | 0.736688 | 4.992/5.422/19.424 | 1.621x |
| 0.15 | 0.149413 | 17.266 | 3/37/30 | 21.01% | 8.745 | 2.500x | 6.693x | 0.826304/0.781686/0.602484 | 0.736825 | 4.628/5.095/18.494 | 1.749x |
| 0.10 | 0.098478 | 17.035 | 13/50/7 | 22.07% | 6.716 | 3.255x | 10.155x | 0.826150/0.781880/0.602197 | 0.736742 | 3.296/3.402/15.095 | 2.455x |
| 0.05 | 0.049994 | 13.429 | 33/36/1 | 38.57% | 4.005 | 5.458x | 20.002x | 0.789874/0.746329/0.563222 | 0.699808 | 3.170/3.256/15.286 | 2.552x |

The greedy 0.30 row is not a valid 0.30-budget result because its absolute
BOPS error is `0.017859 > 0.005`; it is retained only as the nearest greedy
path snapshot.  The other five greedy rows pass the formal band.

GA full-validation results:

| budget | actual BOPS | physical FP32 MB | canonical INT8/FP16/FP32 | parameter prune | mixed-weight MB | mixed-weight x | BOPS x | AP30/50/70 | mAP | p50/p90/p99 ms | speedup |
|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---|---:|
| FP32 reference | 1.000000 | 21.859 | 0/0/70 | 0.00% | 21.859 | 1.000x | 1.000x | 0.826153/0.781525/0.602142 | 0.736607 | 8.657/9.263/28.389 | 1.000x |
| 0.30 | 0.304223 | 17.266 | 0/26/44 | 21.01% | 10.854 | 2.014x | 3.287x | 0.826222/0.781621/0.602359 | 0.736734 | 7.115/7.555/22.639 | 1.217x |
| 0.25 | 0.254517 | 17.339 | 1/29/40 | 20.68% | 10.499 | 2.082x | 3.929x | 0.826220/0.781713/0.602170 | 0.736701 | 5.485/5.939/21.166 | 1.578x |
| 0.20 | 0.204300 | 17.193 | 3/29/38 | 21.35% | 9.210 | 2.373x | 4.895x | 0.826216/0.781564/0.602136 | 0.736639 | 5.030/5.499/19.641 | 1.721x |
| 0.15 | 0.154526 | 17.193 | 10/31/29 | 21.35% | 7.977 | 2.740x | 6.471x | 0.826282/0.781627/0.602607 | 0.736839 | 4.920/5.235/18.311 | 1.760x |
| 0.10 | 0.104957 | 17.035 | 14/47/9 | 22.07% | 6.636 | 3.294x | 9.528x | 0.826102/0.781602/0.602536 | 0.736747 | 3.465/3.626/15.546 | 2.499x |
| 0.05 | 0.054807 | 12.212 | 32/37/1 | 44.13% | 3.780 | 5.783x | 18.246x | 0.796773/0.753495/0.572820 | 0.707696 | 3.139/3.242/14.660 | 2.758x |

All six GA rows pass `abs(R_BOPS-target) <= 0.005`.  The configured GA final
winner is the 0.10 row.  The 0.15 row has the highest measured mAP, while both
search methods show an accuracy cliff at 0.05.

Completed checkpoint: 2026-07-18 01:47:37 +0800 CST

---

## Formal pruning / mixed-quantization contribution ablation

Completed a fresh same-GPU full-validation ablation for all six accepted GA
winners and all six greedy winners. Every searched configuration was evaluated
as P+Q, exact-mask strict-FP32 P-only, and original-all-keep Q-only. The run
uses H800 GPU 7, fixedK29696, train200 entropy calibration, warmup/reset,
DataLoader workers=8, CUDA postprocess, and 1789 evaluated / 0 skipped frames.

The FP32 reference was rebuilt and remeasured in this run: `mAP=0.736616873`,
`p50=7.031603 ms`. The invalid old greedy 0.30 snapshot (`0.282141`) was
replaced by exact replay of path step 104 (`R_BOPS=0.301922`, delta `0.001922`),
which passed full validation with `mAP=0.736713448`, `p50=6.582260 ms`.

Main conclusion: at BOPS 0.10--0.30, P-only, Q-only, and P+Q all retain FP32
accuracy. At BOPS 0.05, P-only remains near `0.736` mAP while Q-only and P+Q
fall to `0.699--0.708`; the accuracy cliff is primarily caused by the
aggressive quantization profile rather than physical structured pruning. The
largest absolute pruning/quantization interaction is only `0.001323` mAP.

Production additions:

```text
search/ablation/lidar_pyramid_prune_quant.py
scripts/run_lidar_pyramid_prune_quant_ablation.py
tests/test_search_lidar_pyramid_prune_quant_ablation.py
docs/codex_handoffs/H800_LIDAR_PYRAMID_PRUNE_QUANT_ABLATION_20260719.md
```

Machine-readable evidence root:

```text
outputs/h800_lidar_pyramid_prune_quant_ablation_20260718_124729/
```

See the dedicated handoff and `ablation_report.md` for all AP30/AP50/AP70/mAP,
p50/p90/p99, precision coverage, parameter pruning, speedup, contribution, and
interaction tables. Outputs, checkpoints, ONNX files, Q/DQ files, calibration
caches, and engines remain excluded from Git.

Implementation/results commit: `a811ce8`. Push to the current H800 branch was
attempted, but GitHub authentication was unavailable because the VS Code Git
credential socket on this server was stale; the local branch remains the
authoritative source until credentials are refreshed.

Completed checkpoint: 2026-07-19 05:47:26 +0800 CST

---

## Dual-GPU, evaluation-only fairness rerun

Re-evaluated the accepted strict-FP32, six GA P+Q, and six greedy P+Q engines
on H800 GPU 0 and GPU 1. GPU 0 completed its entire sequence before GPU 1
started. Every item ran in a fresh process, followed by explicit CUDA cache
cleanup and nvidia-smi memory-return validation. All 26 evaluations used the
same fixedK29696 1789-frame manifest, warmup200/reset, latency rounds 3,
DataLoader workers 8, and CUDA postprocess; all completed 1789/1789 with zero
skips and the same frame-order hash.

This run invoked zero engine builds. All source engine SHA256 values and six
deployment acceptance reports were verified before evaluation, source hashes
were unchanged afterward, and evaluation directories contain no model/ONNX/
engine files.

Fresh strict-FP32 references were `mAP=0.736783, p50=6.9771 ms` on GPU 0 and
`mAP=0.736601, p50=6.9403 ms` on GPU 1. The largest cross-GPU mAP difference
over all engines was `0.000339514`; the largest p50 difference was
`0.079973 ms`. Both cards reproduce near-lossless accuracy for budgets
0.10--0.30 and the 0.05 accuracy cliff.

Evidence and the complete AP30/AP50/AP70/mAP/p50/p90/p99/speedup tables:

```text
outputs/h800_lidar_pyramid_dual_gpu_eval_only_20260718_154151/
docs/codex_handoffs/H800_LIDAR_PYRAMID_DUAL_GPU_FAIR_EVALUATION_20260719.md
```

Completed checkpoint: 2026-07-19 07:17:50 +0800 CST

---
