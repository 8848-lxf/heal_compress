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
