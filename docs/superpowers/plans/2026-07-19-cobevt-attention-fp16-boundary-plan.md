# CoBEVT Attention FP16 Boundary Audit Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and execute a strongly typed CoBEVT Attention FP16 boundary audit that identifies the minimum accuracy-safe FP32 island.

**Architecture:** Extend the existing CoBEVT canonical typed-ONNX path with one model-family boundary rewriter between canonical typing and auxiliary closure. A dedicated orchestration module owns fresh profile builds, staged evaluation, TensorRT realization joins, diagnostic tensor outputs, parity metrics, and final reports.

**Tech Stack:** Python 3.9, PyTorch, ONNX opset 17, TensorRT 10.9, CUDA 11.8, existing CoBEVT fixed-K exporter/evaluator, pytest.

## Global Constraints

- Branch: `feature/cobevt-attention-dim-pruning-audit`.
- One strongly typed `fixedK=29696` engine per profile; no bucket/chunk routing.
- GPU 2 for TensorRT work; `modelopt` for build/evaluation and `univ2x-opt` for tests.
- No GA, Greedy, tracer, ChannelResolver, physical-pruner, search-gene, INT8, training, or weight changes.
- All profile artifacts are fresh and isolated under `/data/lxf/heal_data/outputs/cobevt_attention_fp16_boundary_audit_<timestamp>/`.

---

### Task 1: Central Profile Contract and Node Discovery

**Files:**
- Create: `search/model_families/lidar_cobevt/attention_precision_boundaries.py`
- Create: `tests/test_lidar_cobevt_attention_precision_boundaries.py`

**Interfaces:**
- Produces: `AttentionBoundaryProfile`, `attention_boundary_profile(name)`,
  `discover_attention_nodes(model, mapping)`, and
  `apply_attention_boundary_contract(input_onnx, output_onnx, mapping, profile)`.

- [ ] Write RED tests that assert all A0-A7 role dtypes, unique names, complete
  role coverage, six-block discovery, and fail-closed missing/duplicate nodes.
- [ ] Run the focused tests and confirm missing imports/functions fail.
- [ ] Implement immutable profile records and deterministic node discovery.
- [ ] Run focused tests and confirm green.
- [ ] Commit `feat: add CoBEVT attention precision boundary profiles`.

### Task 2: Explicit Cast Rewrite and Provenance

**Files:**
- Modify: `search/model_families/lidar_cobevt/attention_precision_boundaries.py`
- Modify: `tests/test_lidar_cobevt_attention_precision_boundaries.py`

**Interfaces:**
- `apply_attention_boundary_contract(...) -> dict[str, Any]` writes typed ONNX
  and returns inserted-cast and per-role provenance records.

- [ ] Add RED synthetic-ONNX tests for A1 QKV output recovery, A2 LayerNorm,
  A3 QK, A4 Softmax, A5 AV, A6 Out, A7 residual, unchanged external nodes,
  and no silent default.
- [ ] Run the tests and confirm each expected boundary fails before implementation.
- [ ] Implement deterministic input/output Cast insertion with unique names,
  tensor type updates, checker validation, and source/output SHA256.
- [ ] Run scalar profile tests and full CoBEVT-family tests.
- [ ] Commit `feat: add explicit CoBEVT attention dtype islands`.

### Task 3: Static and Realized Precision Inventory

**Files:**
- Create: `search/reporting/cobevt_attention_precision_inventory.py`
- Create: `tests/test_lidar_cobevt_attention_precision_inventory.py`

**Interfaces:**
- Produces: `build_attention_precision_inventory(...)` and
  `audit_attention_boundary_realization(...)`.

- [ ] Write RED tests joining module path, ONNX node, tensor dtype, Cast
  provenance, EngineInspector metadata, fusion, and realized input/output dtype.
- [ ] Add fail-closed tests for missing functional nodes and unresolved fused
  precision.
- [ ] Implement JSON and Markdown inventory rendering plus requested/ONNX/TRT
  mismatch summaries.
- [ ] Validate against existing strict FP32, strict FP16, and safe mixed engines.
- [ ] Commit `feat: audit CoBEVT attention precision provenance`.

### Task 4: Fresh Boundary Build/Evaluation Orchestration

**Files:**
- Create: `search/orchestration/lidar_cobevt_attention_precision_audit.py`
- Modify: `search/orchestration/lidar_cobevt_attention_pruning.py`
- Create: `tests/test_lidar_cobevt_attention_precision_audit.py`

**Interfaces:**
- Produces CLI phases `prepare`, `inventory`, `build`, `smoke10`, `fixed50`,
  `fixed500`, `parity`, and `report`.
- Extends existing export/build with `attention_boundary_profile` while leaving
  all historical diagnostic profiles byte-compatible.

- [ ] Add RED tests for new timestamp root, copied manifest hash identity,
  fixed50 derivation, profile-specific directories, freshness signatures,
  staged admission, and 10/10 or 50/50 or 500/500 with zero skip.
- [ ] Implement the minimal reusable hooks in the existing runner.
- [ ] Implement the dedicated audit CLI and run-manifest writer.
- [ ] Confirm historical CoBEVT attention tests remain green.
- [ ] Commit `feat: orchestrate CoBEVT attention boundary audit`.

### Task 5: Diagnostic Tensor Outputs and Metrics

**Files:**
- Create: `search/model_families/lidar_cobevt/attention_tensor_parity.py`
- Create: `tests/test_lidar_cobevt_attention_tensor_parity.py`

**Interfaces:**
- Produces: `append_attention_diagnostic_outputs`, `tensor_error_metrics`,
  `qk_metrics`, `softmax_metrics`, `residual_metrics`, and
  `select_failure_frames`.

- [ ] Add RED tests for output tagging without changing production ONNX,
  generic statistics, top-k/rank/sign metrics, entropy/KL/JS, residual update
  retention, finite handling, and deterministic worst-frame selection.
- [ ] Implement metrics in float64 with epsilon-protected divisions.
- [ ] Add a diagnostic-engine runner using the existing TensorRT runtime and
  mark every result `diagnostic_latency_invalid=true`.
- [ ] Run tests and commit `feat: add CoBEVT attention tensor parity diagnostics`.

### Task 6: Execute A0-A7 Smoke and Staged Follow-up

**Files:**
- Generate only under the timestamped `/data` output root.

**Interfaces:**
- Consumes the audit CLI and produces profile build/evaluation records.

- [ ] Audit GPU 2 processes and record UUID/utilization/memory/temperature/power.
- [ ] Run `prepare` and `inventory`; verify fixed-K, manifest, plugin, checkpoint,
  config, and source commit hashes.
- [ ] Fresh-build A0-A7 in order and require successful EngineInspector audits.
- [ ] Run smoke10 in order A1-A7 and classify deltas against A0.
- [ ] Run fixed50 only for ambiguous/parity-suspicious profiles.
- [ ] Preserve every failure record without rebuilding an identical signature.

### Task 7: Resolve and Evaluate Combination Profiles

**Files:**
- Generate resolved M4/M5 profile JSON under the run output.

**Interfaces:**
- Produces a staged candidate matrix and two or three fixed500 finalists.

- [ ] Construct only evidence-supported M1-M5 combinations.
- [ ] Build and run smoke10/fixed50 for combinations.
- [ ] Select two or three candidates using AP safety first and latency second.
- [ ] Run fixed500 and enforce 500/500, zero skip, finite output, and exact
  requested/realized identity.
- [ ] Measure formal latency only if GPU 2 is isolated; otherwise mark the
  latency requirement blocked without substituting diagnostic/screening values.

### Task 8: Execute Tensor Parity and Root-Cause Report

**Files:**
- Generate: `tensor_parity_summary.json/csv`, `attention_failure_frames.json`,
  `precision_boundary_matrix.json/csv`, `root_cause_report.md`, and
  `final_recommended_precision_contract.json`.
- Update: `docs/codex_handoffs/4090-cobevt-attention-fp16-boundary-audit.md`
- Update: `docs/codex_handoffs/4090-cobevt-attention-dim-pruning-progress-20260718.md`

**Interfaces:**
- Produces the final evidence-backed FP32/FP16 node sets and search-policy advice.

- [ ] Run parity on all smoke10 frames, then retain detailed records for the
  three largest-error frames and required scenario diversity.
- [ ] Identify the earliest tensor with material divergence from A0.
- [ ] Generate root-cause answers, profile failures, final node contract, AP,
  latency, speedup, provenance, and reproduction commands.
- [ ] Run all new tests, `tests/test_lidar_cobevt_*.py`, modified-file
  `py_compile`, `git diff --check`, branch/ancestor/remote checks.
- [ ] Commit and push only `feature/cobevt-attention-dim-pruning-audit`.
