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
