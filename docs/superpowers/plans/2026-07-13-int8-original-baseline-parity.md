# INT8 Original Baseline Parity Investigation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:systematic-debugging for the investigation. Use superpowers:test-driven-development before any production code change. This run executes inline because the user explicitly requested direct execution after the plan.

**Goal:** Identify the smallest root cause for the original unpruned search-path INT8 mAP collapse by strictly comparing legacy single maxK and search-path original/maximal-legal deployment chains.

**Architecture:** Use a single run root for immutable experiment metadata, copied manifests, rebuilt artifacts, audit reports, small-set metrics, and full validation metrics. Treat legacy single maxK INT8 train200 as the working reference and search-path explicit-QDQ INT8 as the suspect chain. No fixes are allowed until structure, precision profile, Q/DQ, calibration, binding, and numeric drift evidence isolates the failing boundary.

**Tech Stack:** PyTorch, ONNX, TensorRT 10.9, HEAL/OpenCOOD LiDAR pyramid model, fixedK29696 TensorRT plugin, repository quantization/search exporters and evaluators.

## Global Constraints

- All four engines use the same original unpruned checkpoint.
- All four engines use the original unpruned structure and fixedK29696.
- All four engines use the same plugin library, input profile, output binding order, preprocessing, postprocess, calibration_200 manifest and order, validation manifest, TensorRT version, GPU, warmup protocol, and latency protocol.
- Full validation uses warmup 200 frames, excludes warmup from metrics, resets the iterator after warmup, evaluates all 1789 validation frames, and requires skipped_frames = 0.
- If structure differs, stop INT8 analysis and report the structure difference.
- If FP16 differs, stop INT8 analysis and fix export, binding, or postprocess before INT8 investigation.
- Do not change original model structure, tracer, physical pruning logic, or detection postprocess unless controlled evidence proves those layers are the root cause.

---

### Task 1: Artifact Layout And Existing Artifact Inventory

**Files and directories:**
- Create run root: `outputs/int8_original_baseline_parity_20260713`
- Read configs, scripts, and outputs matching legacy single maxK, search-path original/maximal-legal, fixedK29696, calibration_200, and validation metrics.
- Write inventory: `outputs/int8_original_baseline_parity_20260713/artifact_inventory.json`

**Steps:**
- [ ] Record git status and repository revision.
- [ ] Locate legacy single maxK FP16 and INT8 train200 configs, ONNX, engines, logs, and metrics.
- [ ] Locate search-path original/maximal-legal FP16 and INT8 explicit-QDQ configs, ONNX, engines, logs, and metrics.
- [ ] Locate checkpoint, plugin path, TensorRT binary, calibration_200 manifest, validation manifest, profile shapes, and binding maps.
- [ ] Decide per artifact whether to reuse, rebuild, or mark missing.

### Task 2: Strict Four-Way Control Definition

**Artifacts:**
- A1: legacy single maxK FP16.
- A2: legacy single maxK INT8 train200.
- B1: search-path original-model FP16.
- B2: search-path original-model INT8.

**Steps:**
- [ ] Generate `control_matrix.json` with checkpoint path/hash, model config path/hash, plugin path/hash, TensorRT version, GPU, input profile, calibration manifest hash, validation manifest hash, output bindings, preprocessing adapter, postprocess adapter, warmup, and eval frame count for each chain.
- [ ] Fail the control if any non-intentional field differs.
- [ ] Rebuild missing or mismatched A1/A2/B1/B2 artifacts into the run root.

### Task 3: Structure Parity Reports

**Outputs:**
- `reports/legacy_fp16_structure.json`
- `reports/legacy_int8_structure.json`
- `reports/search_fp16_structure.json`
- `reports/search_int8_structure.json`
- `reports/structure_diff.md`

**Steps:**
- [ ] Report PyTorch parameter count, physical_hash, pruned_unit_count, and weighted tensor names/shapes for the original checkpoint through both pipelines.
- [ ] Report ONNX initializer names/shapes and Conv, ConvTranspose, Linear, BN channel signatures.
- [ ] Report weighted layer count and output names, shapes, and order.
- [ ] Confirm pruned_unit_count = 0, physical parameter count unchanged, all weighted tensor shapes unchanged, and physical_hash equals the original model structure hash.
- [ ] Stop INT8 analysis if legacy original structure differs from search-path original structure outside Q/DQ, Cast, or TensorRT fusion representation.

### Task 4: Small Validation Reproduction

**Outputs:**
- `metrics/small_validation_table.csv`
- `metrics/small_validation_table.md`

**Steps:**
- [ ] Run A1, A2, B1, and B2 on the same fixed small validation subset.
- [ ] Use the same warmup, iterator reset behavior, plugin, profile, bindings, preprocessing, and postprocess.
- [ ] Confirm legacy FP16 approximates 0.7363 on full-run reference and legacy INT8 approximates 0.6124 on full-run reference where existing metrics are reused.
- [ ] If B1 FP16 does not align with A1 FP16 on small validation outputs and AP, stop INT8 analysis and investigate FP16 export, binding, or postprocess.

### Task 5: Realized Precision Profile Diff

**Outputs:**
- `reports/legacy_realized_precision_profile.json`
- `reports/search_realized_precision_profile.json`
- `reports/realized_precision_profile_diff.csv`
- `reports/realized_precision_profile_diff.md`

**Steps:**
- [ ] Canonicalize weighted layers across PyTorch modules, ONNX initializers, and TensorRT layer names.
- [ ] For each canonical weighted layer, list legacy requested precision, legacy realized precision, search requested precision, search realized precision, match status, and fallback reason.
- [ ] Count weighted INT8 layers and weighted FP16 fallback layers for legacy and search.
- [ ] Verify whether search 22 INT8 + 48 FP16 matches legacy exactly.
- [ ] If sets differ, rebuild B2 with the legacy precision profile before continuing Q/DQ and numeric drift analysis.

### Task 6: ONNX Q/DQ Graph Audit

**Outputs:**
- `reports/legacy_qdq_audit.json`
- `reports/search_qdq_audit.json`
- `reports/qdq_graph_diff.json`
- `reports/qdq_graph_diff.md`

**Steps:**
- [ ] Count QuantizeLinear and DequantizeLinear nodes.
- [ ] For each Q/DQ, record tensor, weight or activation role, scale value summary, scale shape, scale dtype, zero point, axis, per-tensor or per-channel mode, symmetric or asymmetric mode, and graph location.
- [ ] Compare Q/DQ placement around Conv, Add, Concat, fusion, BEV, heads, and plugin boundaries.
- [ ] Detect duplicate quantization, missing dequantization, shared scale misuse, quantized output heads, and plugin boundary mismatches.

### Task 7: Calibration Diff

**Outputs:**
- `reports/calibration_scale_diff.csv`
- `reports/calibration_scale_diff.md`

**Steps:**
- [ ] Verify both chains use the same calibration_200 manifest with identical ordering.
- [ ] Compare tensor min, max, absmax, scale, method, cache key, dynamic range, and missing status.
- [ ] Flag 0, NaN, Inf, extremely small, or extremely large scales.
- [ ] Focus review on BEV/fusion activations, residual/add sides, concat inputs, cls/reg/dir heads, scatter/plugin boundaries, and grouped convolution input/output tensors.
- [ ] Fail if any default scale=1 or silent scale fill is observed.

### Task 8: Layerwise Numeric Alignment

**Outputs:**
- `reports/numeric_alignment/*.json`
- `reports/numeric_alignment_summary.md`

**Steps:**
- [ ] Run identical fixed inputs through legacy FP16 versus search FP16.
- [ ] Run identical fixed inputs through legacy FP16 versus legacy INT8.
- [ ] Run identical fixed inputs through search FP16 versus search INT8.
- [ ] Run identical fixed inputs through legacy INT8 versus search INT8.
- [ ] Save and compare scatter output, early backbone, backbone stages, fusion/BEV, shrink layer, cls head, reg head, dir head, and final engine outputs.
- [ ] Record shape, min, max, mean, std, MAE, relative L2, cosine similarity, NaN/Inf, saturation ratio, and zero ratio.
- [ ] Identify the first layer or group where INT8 drift becomes severe.

### Task 9: Quantization Group Ablation

**Outputs:**
- `reports/ablation_results.csv`
- `reports/ablation_results.md`

**Steps:**
- [ ] Execute only if FP16 aligns and legacy/search realized precision profiles align but INT8 still diverges.
- [ ] Start from all-FP16 search engine.
- [ ] Enable one quantization group at a time on the fixed small validation set.
- [ ] Save output drift and AP per group.
- [ ] Use binary search over groups if single-group evaluation is too slow.

### Task 10: Engine Inspector Diff

**Outputs:**
- `reports/legacy_engine_inspector.json`
- `reports/search_engine_inspector.json`
- `reports/engine_inspector_diff.md`

**Steps:**
- [ ] Export TensorRT Engine Inspector data for A2 and B2.
- [ ] Compare layer name, type, input/output dtype, precision, tensor format, tactic, fusion, Q/DQ fusion, plugin precision, binding names, binding order, and binding shapes.
- [ ] Confirm actual INT8 weighted layers, FP16 fallback layers, plugin boundaries, detection head outputs, and binding order.

### Task 11: Full Validation And Trust Decision

**Outputs:**
- `metrics/full_validation_table.csv`
- `metrics/full_validation_table.md`
- `reports/root_cause.md`

**Steps:**
- [ ] Run aligned A1, A2, B1, and B2 on all 1789 validation frames after small validation and structural/profile parity are established.
- [ ] Use warmup 200 frames, do not count warmup in metrics, reset the iterator after warmup, require evaluated = 1789 and skipped = 0.
- [ ] Report engine, mAP, AP@0.30, AP@0.50, AP@0.70, p50, evaluated, and skipped.
- [ ] Accept parity only if |Delta mAP FP16| <= 0.002 and |Delta mAP INT8| <= 0.005.
- [ ] If parity fails, mark search-path INT8 as `accuracy_validation_failed` and `trusted_baseline = false`.
- [ ] Answer the ten required final questions with file-backed evidence.
