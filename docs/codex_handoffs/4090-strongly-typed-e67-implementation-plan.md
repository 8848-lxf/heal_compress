# 4090 Strongly Typed E67 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: execute this plan with test-first, fail-closed checkpoints. This repository is operated inline; do not dispatch subagents.

**Goal:** Replace the production weakly typed TensorRT route with a strongly typed ONNX/QDQ/Cast route whose declared tensor dtypes match the realized engine, then gate GA on fresh 4090 E67 evidence.

**Architecture:** Keep canonical precision mapping and semantic QDQ ownership unchanged. Add an explicit ONNX dtype-closure pass for FP16/FP32 regions and the PointPillarScatterTRT boundary, and make the production TensorRT command accept only `--stronglyTyped` without implicit precision flags or layer constraints. Validate the plugin in isolated FP16 and FP32 graphs before applying the same contract to E67.

**Tech Stack:** Python 3.10, ONNX opset 17, TensorRT 10.9.0.34, CUDA 11.8, C++17/CUDA plugin, pytest, RTX 4090 SM89.

## Global Constraints

- Work, commit, and push only on `feature/heal-compress-h800-sync-4090`.
- Preserve ancestor `b862b3d8ad061bd12580776226c75f564918298d`.
- Never push or modify `feature/heal-compress-h800`; never force push.
- PointPillarScatterTRT remains floating point and is not a canonical precision gene.
- The canonical E67 profile remains 67 INT8 / 3 FP16 / 0 unresolved outside the plugin boundary.
- Strongly typed failures fail closed; weakly typed engines are diagnostic-only.
- Stage A and Stage B remain disabled until all readiness gates pass.
- Generated ONNX, engines, caches, plugin binaries, and checkpoints remain ignored.

---

## Design Decision

Three approaches were considered:

1. **Typed ONNX plus `trtexec --stronglyTyped` (selected).** This keeps the existing subprocess isolation, records a reproducible command, and makes ONNX Q/DQ/Cast the sole precision authority.
2. **Python TensorRT builder for production.** It gives richer parser introspection but duplicates the current command builder and increases CUDA process lifetime. It is retained only for a minimal diagnostic probe if trtexec cannot expose the parser boundary.
3. **Immediate IPluginV3 migration.** This is higher risk and unnecessary unless the minimal graph proves TensorRT 10.9 cannot safely instantiate the current `IPluginV2DynamicExt`. TensorRT 10.9 headers explicitly describe strongly typed inference through `getOutputDataType`, so migration must be evidence-driven.

The first root-cause hypothesis is that the production ONNX type contract is incomplete: the plugin feature input lacks `value_info`, while its output is recorded as FLOAT even though `getOutputDataType` mirrors input 0. The minimal graph will confirm or reject that hypothesis before plugin ABI changes.

## Task 1: Strongly Typed Plugin Probe

**Files:**

- Create: `scripts/run_strongly_typed_scatter_probe.py`
- Create: `tests/test_strongly_typed_scatter_probe.py`
- Modify only if the failing probe identifies a plugin defect: `quantization/plugins/pointpillar_scatter_trt/pointpillar_scatter_plugin.{h,cpp}` and `pointpillar_scatter_kernel.cu`

**Interfaces:**

- `build_typed_scatter_onnx(destination, boundary_dtype, sample_shape)` emits a four-input production-signature graph with explicit feature Cast and typed plugin output.
- `strongly_typed_trtexec_command(...)` emits `--stronglyTyped` and rejects INT8 boundary requests.
- `audit_scatter_engine(...)` checks parser/build/deserialization, inspector dtypes, plugin SHA, engine SHA, and tensor parity.

- [ ] Write tests requiring FP16 and FP32 graph types and rejecting INT8.
- [ ] Run the tests and retain the expected failures caused by missing probe APIs.
- [ ] Implement only the graph/probe APIs needed by the tests.
- [ ] Build the current plugin fresh with TensorRT 10.9 headers, SM89, and the modelopt runtime.
- [ ] Run FP32 and FP16 minimal probes on one idle 4090; save ONNX, logs, inspector, dtype audit, hashes, commands, and parity.
- [ ] If either probe crashes, isolate parser, builder, serialization, and runtime phases before changing plugin code.
- [ ] Add a regression test for every confirmed plugin defect, apply the smallest fix, and rerun both probes.

## Task 2: Typed QDQ/Cast Closure

**Files:**

- Create: `quantization/precision/typed_graph.py`
- Create: `tests/test_strongly_typed_qdq_graph.py`
- Modify: `quantization/precision/qdq_inserter.py`
- Modify: `quantization/types.py`
- Modify: `search/integration/trt_compatible_export.py`

**Interfaces:**

- `apply_strongly_typed_precision_contract(input_onnx, output_onnx, mapping, plugin_boundary)` inserts deterministic FP16/FP32 Cast boundaries without changing Q/DQ ownership.
- `audit_typed_precision_contract(...)` reports canonical dtypes, merge dtypes, plugin input/output dtypes, unresolved values, and forbidden plugin Q/DQ.
- `plugin_boundary` is exactly `fp16` or `fp32` and never appears in `layer_bitwidth`.

- [ ] Write tests for FP16 Casts, FP32 Casts, Q/DQ preservation, merge input/output closure, and plugin boundary isolation.
- [ ] Verify the tests fail because the typed graph pass does not exist.
- [ ] Implement deterministic Cast insertion and value-info repair while preserving post-ReLU/post-merge QDQ topology.
- [ ] Test that `/Concat_9` inputs/output have one declared dtype and no implicit fallback path.
- [ ] Test that both scatter variants preserve canonical 67/3 and quantization group identity.

## Task 3: Strongly Typed Production Builder

**Files:**

- Modify: `quantization/config.py`
- Modify: `quantization/tensorrt/command.py`
- Modify: `quantization/tensorrt/builder.py`
- Modify: `search/baselines/original_engines.py`
- Modify: `search/stage2/lidar_pyramid_real_evaluator.py`
- Modify: `search/stage2/trt_build_worker.py`
- Test: `tests/test_formal_packages_cpu.py`
- Create: `tests/test_strongly_typed_builder.py`

**Interfaces:**

- `TensorRTBuildConfig(strongly_typed=True, production=True)` is the only accepted production configuration.
- `build_trt_command` emits `--stronglyTyped` and never emits `--fp16`, `--int8`, `--precisionConstraints`, `--layerPrecisions`, or `--layerOutputTypes` for production.
- The worker records network dtype audit, parser errors, builder flags, inspector data, plugin boundary dtype, and `strongly_typed=true`.

- [ ] Write failing tests for the required command and every forbidden weakly typed option.
- [ ] Write a failing test that production rejects `strongly_typed=false` rather than falling back.
- [ ] Implement the new configuration and command branch without deleting the diagnostic legacy command path.
- [ ] Route formal Stage-2 and E67 baselines exclusively through strongly typed production mode.
- [ ] Invalidate deployment/cache signatures when typed policy or plugin boundary changes.

## Task 4: Fresh E67 Boundary Comparison

**Files:**

- Create: `scripts/run_4090_strongly_typed_e67_readiness.py`
- Create: `search/configs/lidar_pyramid_4090_strongly_typed_e67_readiness.yaml`
- Create: `tests/test_4090_strongly_typed_readiness.py`
- Create/update: `docs/codex_handoffs/4090-strongly-typed-e67-readiness-report.md`

**Interfaces:**

- One readiness run creates strict FP16, E67 scatter-FP16, and E67 scatter-FP32 artifacts under one timestamped output root.
- Both E67 variants consume the same train200 calibration manifest and fixed 200-frame validation manifest.
- Selection requires structural correctness first, then lower latency when AP is within the configured equivalence tolerance.

- [ ] Fresh-build strict FP16 and both typed E67 variants on the same idle 4090.
- [ ] Run 10-frame smoke then 200-frame evaluation for each variant.
- [ ] Audit 70 canonical entries, 67/3/0 profile, semantic QDQ, per-channel weights, entropy lineage, merge dtypes, plugin dtypes, inspector parity, 200/200, and zero skips.
- [ ] Select FP16 or FP32 plugin boundary only if its correctness and stability gates pass.
- [ ] Freeze the selected boundary in readiness and Stage-A configs and tests.
- [ ] Write all five required readiness booleans and keep `READY_FOR_GA=false` on any missing evidence.

## Task 5: GA Admission

**Files:**

- Modify: `search/configs/lidar_pyramid_4090_ga_stage_a.yaml`
- Modify as required by the selected boundary: `search/stage2/candidate_worker.py`
- Update: `docs/codex_handoffs/4090_GA_EXPLICIT_QDQ_PROGRESS_20260714.md`

- [ ] Run a five-unique-candidate multi-GPU Top-5 smoke on GPUs 4/5/6/7 only after readiness is true.
- [ ] Verify serial/parallel metric equivalence and exact deployment/cache isolation.
- [ ] Run Stage A only after the smoke passes: target 0.21 +/- 0.005, five generations, per-generation Top-5 at 300 frames, five winners at 500 frames.
- [ ] Stop without Stage B if any Stage-A gate fails.

## Verification and Delivery

- [ ] Run all related plugin, QDQ, builder, realization, orchestration, worker, and formal package tests.
- [ ] Run `python -m py_compile` for every modified Python file.
- [ ] Run `git diff --check`.
- [ ] Append a timestamped 4090 progress round with files, behavior, evidence, failure reasons, and next gate.
- [ ] Commit and push only `feature/heal-compress-h800-sync-4090` without force.
- [ ] Verify clean status and ancestor `b862b3d8ad061bd12580776226c75f564918298d` after push.

Completed planning checkpoint: 2026-07-14 23:41:05 CST

---
