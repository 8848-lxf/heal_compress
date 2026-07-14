# 4090 Strongly Typed E67 Readiness Report

## Scope and branch gate

- repository: `/home/lixingfeng/UniAD_examine/heal_compress`;
- branch: `feature/heal-compress-h800-sync-4090`;
- starting HEAD: `2a23ebe`;
- required ancestor `b862b3d8ad061bd12580776226c75f564918298d`: present;
- weakly typed engines are retained only as diagnostic references;
- Stage A and Stage B were not started.

## Production implementation

The production build path now uses typed ONNX plus `--stronglyTyped`:

- `quantization/precision/typed_graph.py` materializes canonical FP16/FP32
  compute, requested output precision, Q/DQ input dtype, residual/concat,
  functional affine-grid MatMul, GridSample and floating elementwise contracts
  with explicit Cast nodes;
- `quantization/tensorrt/command.py` rejects weak production fallback and
  forbids `--fp16`, `--int8`, precision constraints, layer precisions and layer
  output types in strongly typed mode;
- `search/stage2/lidar_pyramid_real_evaluator.py` builds only
  `typed_qdq.onnx` for production Stage-2;
- context builder flags and deployment quantization-contract hashes include
  `strongly_typed`, typed-graph identity and the fixed plugin boundary;
- Stage-A config contains `readiness_selection_required`, so it cannot start
  before the boundary comparison is accepted.

The canonical mapping, pruning groups, quantization groups, H800 semantic QDQ
boundaries, EntropyCalibration2 recipe and per-output-channel weight scales
were not changed.

## Strongly typed plugin minimum graph

Evidence root:

`outputs/4090_strongly_typed_scatter_probe_20260714_234751/`

The plugin was rebuilt fresh with modelopt GCC/G++/nvcc, TensorRT 10.9 headers,
CUDA 11.8 and `CMAKE_CUDA_ARCHITECTURES=89`.

- physical GPU: 4;
- GPU UUID: `GPU-166702d7-bf30-18e0-83ff-83b316d37c0e`;
- TensorRT: 10.9.0.34;
- plugin SHA256:
  `5a5224f15831cb0712f945f623b4e252d0b617a28451de57afc98ab118d9855b`;
- plugin API: `IPluginV2DynamicExt`; TensorRT 10.9 accepted it in strongly
  typed mode, so an evidence-free IPluginV3 migration was not performed;
- INT8 boundary: rejected by the public graph API and unit test.

| boundary | parser/build | deserialize | inspector plugin I/O | parity max abs | engine SHA256 |
|---|---|---|---|---:|---|
| FP16 | pass | pass | Half / Half | 0.0 | `928421581317faa45ec429a703190e4ce49b406c64542a58463414637eec1312` |
| FP32 | pass | pass | Float / Float | 0.0 | `68dd9babacbfbb72926425eb7e80b4b29b30337e6c6204b4d1f3a68e6706df17` |

`plugin_probe_summary.json`, both ONNX files, layer-info files, build logs and
engines are saved under the evidence root and remain ignored by Git. The one
sample latency values are initialization-contaminated diagnostics and are not
used to select a boundary.

## Full E67 structural probe

The prior fresh 4090 E67 QDQ artifact was used only as an input graph to test
the new typed lowering. Its weak engine was not reused. New strongly typed
ONNX files and engines were built with the fresh plugin above on GPU4.

| boundary | typed Casts | unresolved dtype | build/deser | canonical | merge audit | engine SHA256 |
|---|---:|---:|---|---|---|---|
| FP16 | 269 | 0 | pass/pass | 67 INT8 / 3 FP16 / 0 unresolved | 20/20 pass | `4cf999528ea54c89a785f04376928493c7726576802649a5524f1328cab50b5e` |
| FP32 | 269 | 0 | pass/pass | 67 INT8 / 3 FP16 / 0 unresolved | 20/20 pass | `909e62ce4f92e4d17eedfcf87343ea443ba2e8955cc4161f6daafa6c5b735584` |

The FP16 engine inspector reports PointPillarScatterTRT Half/Half. The FP32
engine reports Float/Float. Both engines pass canonical precision realization
and all 20 merge contracts; `/Concat_9` no longer realizes as undeclared
FP32/mixed.

These are structural diagnostics, not final readiness results: they reused the
already-generated local E67 QDQ scales and have not run the required fresh
post-commit calibration, 10-frame smoke or fixed 200-frame evaluation.

## Verification

- formal/search/strongly-typed suite: 333 passed;
- Python compilation: passed for every modified Python file;
- `git diff --check`: passed;
- no generated ONNX, engine, cache, plugin binary or checkpoint is tracked.

## Current verdict

`STRONGLY_TYPED_PLUGIN_FP16_PASS = true`

`STRONGLY_TYPED_PLUGIN_FP32_PASS = true`

`SELECTED_PLUGIN_BOUNDARY = NONE`

`STRONGLY_TYPED_E67_PASS = false`

`READY_FOR_GA = false`

Reason: both full engines pass the structural gate, but the required fresh
same-commit FP16 reference plus FP16-boundary E67 plus FP32-boundary E67
10/200-frame comparison is not complete. Stage A remains unstarted.

--- Round 1 completed: 2026-07-15 01:09:38 CST ---

## Round 2 - post-commit FP16-boundary readiness attempt

Evidence root:

`outputs/4090_strongly_typed_e67_fp16_readiness_20260714_101615/`

The formal plugin was rebuilt for SM89 before the run. The run used GPU4
(`GPU-166702d7-bf30-18e0-83ff-83b316d37c0e`) and plugin SHA256
`91aec743dca383151b995a60d004cd254ce115ed16152873c1f46754bb15022d`.

The strict FP16 baseline failed closed during strongly typed ONNX parsing:
TensorRT reported a Half activation and Float kernel mismatch at canonical
ConvTranspose node
`__canonical__pyramid_backbone_deblocks_0_0__ConvTranspose__call00061`.
No weakly typed fallback was attempted.

The E67 branch supplied useful independent diagnostics before the overall run
was rejected:

- fresh train200 EntropyCalibration2 cache and activation scales;
- typed graph canonical profile 67 INT8 / 3 FP16 / 0 FP32;
- strongly typed parse, build and deserialize passed;
- canonical engine validation 67 INT8 / 3 FP16 / 0 unknown;
- semantic QDQ and all merge realization checks passed;
- FP16 plugin boundary recorded in the typed graph;
- 200 evaluated, 0 skipped;
- AP03 0.755833, AP05 0.706998, AP07 0.493888, mAP 0.652240;
- forward p50/p90/p95 3.9796/4.3700/5.0445 ms.

This E67 result is not an acceptance result because its paired strict FP16
reference failed. A regression test reproduced the root cause, and the typed
graph pass now closes ConvTranspose activation, weight and optional bias dtypes
explicitly. The fix must be committed before a fresh rerun.

`STRONGLY_TYPED_PLUGIN_FP16_PASS = true`

`STRONGLY_TYPED_PLUGIN_FP32_PASS = true`

`SELECTED_PLUGIN_BOUNDARY = NONE`

`STRONGLY_TYPED_E67_PASS = false`

`READY_FOR_GA = false`

--- Round 2 completed: 2026-07-15 01:28:00 CST ---

## Round 3 - fresh dual-boundary acceptance and production selection

Accepted code commit: `34ebd4d2b536fc4d0a780818e730210c33022b87`.

Evidence roots:

- FP16 plugin boundary plus strict FP16 reference:
  `outputs/4090_strongly_typed_e67_fp16_readiness_20260714_102836/`;
- FP32 plugin boundary:
  `outputs/4090_strongly_typed_e67_fp32_readiness_20260714_103946/`.

Both runs used physical GPU4,
`GPU-166702d7-bf30-18e0-83ff-83b316d37c0e`, driver `580.105.08`,
TensorRT 10.9.0.34, CUDA 11.8 and the freshly rebuilt SM89 plugin with
SHA256 `91aec743dca383151b995a60d004cd254ce115ed16152873c1f46754bb15022d`.
GPU preflight passed immediately before both E67 evaluations. The train200
tensor-manifest hash was identical in both runs:
`eb56308111e20ad7c789b18a8860289fe7357474722dadc43c810868554e0ec5`.
The fixed validation-manifest hash was also identical:
`6f601374e573a5ed7da61eeac07259c0eea34fb5ed72d9bf52265d40a02c9f16`.

Each evaluation performed 20 warmup frames, so the required first ten smoke
frames were executed before measured evaluation. No warmup frame was skipped;
the latency collector was reset after warmup, then exactly 200 measured frames
were evaluated with zero skips.

| engine | AP03 | AP05 | AP07 | mAP | p50 ms | p90 ms | p95 ms | size bytes |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| strict FP16, scatter FP16 | 0.810839 | 0.769922 | 0.595456 | 0.725406 | 5.6439 | 5.8622 | 6.1784 | 39489228 |
| E67, scatter FP16 | 0.755352 | 0.705780 | 0.492622 | 0.651251 | 4.0027 | 4.2952 | 4.8178 | 26422260 |
| E67, scatter FP32 | 0.755723 | 0.706405 | 0.490781 | 0.650970 | 3.9881 | 4.1067 | 4.4139 | 26282852 |

E67 engine SHA256 values:

- scatter FP16:
  `653e8ee2251fa5ec1894f6ebdc20a22853a62a4619310560c40dcbaf3145394a`;
- scatter FP32:
  `579dddaa7324b25b245ec1cc9b7a3be607433213a538422e8aa2701de6d61c84`.

Both E67 variants passed all acceptance audits:

- typed graph canonical profile 67 INT8 / 3 FP16 / 0 FP32;
- engine canonical validation 67 INT8 / 3 FP16 / 0 unknown;
- plugin-adjacent QDQ count zero and unresolved tensor dtype count zero;
- semantic QDQ boundary, per-output-channel weights and EntropyCalibration2
  scale lineage passed;
- all merge contracts passed, including `/Concat_9`;
- no undeclared FP32 canonical fallback;
- Inspector plugin I/O was Half/Half for the FP16 graph and Float/Float for the
  FP32 graph, matching the typed ONNX boundary;
- 200 evaluated and 0 skipped for all three engines;
- no near-zero AP or shrink collapse.

The FP32 plugin boundary was selected. Its mAP differs from FP16 by only
-0.000281, while forward p50/p90/p95 are lower by approximately
0.4%/4.4%/8.4%. Both variants pass correctness and stability, so this follows
the predefined latency tie-break rule. PointPillarScatterTRT remains pure
floating point and is not added to `layer_bitwidth` or canonical 67/3 counts.
`search/configs/lidar_pyramid_4090_ga_stage_a.yaml` now fixes the production
boundary to `fp32`, with a regression assertion in
`tests/test_search_large_population.py`.

Test evidence:

- selected-config and strongly typed focused suite: 23 passed;
- full historical tree in `univ2x-opt`: 823 passed, 4 known unrelated legacy
  failures;
- exact deselection of only those four recorded cases: 823 passed, 4
  deselected;
- the ModelOpt entropy unit passed in the environment that provides the
  `nvidia-modelopt` Python package;
- weakly typed engines were not used as production results.

`STRONGLY_TYPED_PLUGIN_FP16_PASS = true`

`STRONGLY_TYPED_PLUGIN_FP32_PASS = true`

`SELECTED_PLUGIN_BOUNDARY = FP32`

`STRONGLY_TYPED_E67_PASS = true`

`READY_FOR_GA = true`

Stage A has not started at this report boundary. The required four-GPU Top-5
smoke is the next gate.

--- Round 3 completed: 2026-07-15 01:54:00 CST ---
