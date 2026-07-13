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
