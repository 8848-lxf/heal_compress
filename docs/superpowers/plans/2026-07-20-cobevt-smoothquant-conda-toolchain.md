# CoBEVT SmoothQuant Conda Toolchain Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prove a Conda-only SM89 CUDA/ModelOpt toolchain, close the CoBEVT SmoothQuant INT8 projection deployment path, and add the missing S2 fixed500 evidence without touching Pyramid GA.

**Architecture:** Add a fail-closed toolchain/provenance contract beside the existing CoBEVT SmoothQuant profile contracts, then extend the existing real-activation SmoothQuant runner with the approved alpha grid, normalized multi-metric selection, staged fresh-process exports, and realized-precision admission. Reuse the existing fixedK29696 strongly typed CoBEVT exporter/evaluator and immutable S2 structure manifest; generated artifacts live only in a fresh `/data/lxf/heal_data/outputs/` directory.

**Tech Stack:** Python 3, PyTorch cpp_extension, ModelOpt 0.29, ONNX, TensorRT 10.9, CUDA 11.8, pytest, RTX 4090/SM89.

## Global Constraints

- Every CUDA/ModelOpt/ONNX/TensorRT Python process uses the resolved `modelopt` Conda prefix.
- `/usr/bin/nvcc`, `/usr/local/cuda/bin/nvcc`, CPU fallback, stale engines, native INT8 QK, and additive unit LUT admission fail closed.
- Do not modify, pause, restart, or kill Pyramid GA or external processes.
- S2 uses the existing retained indices and F3 contract; no reranking or mask changes.
- SmoothQuant is admitted only after export, TensorRT build, exact realized INT8 projections, FP32 QK, AP, and isolated full-engine latency all pass.

---

### Task 1: Conda CUDA Toolchain Contract

**Files:**
- Create: `search/model_families/lidar_cobevt/conda_cuda_toolchain.py`
- Create: `tests/test_lidar_cobevt_smoothquant_conda_toolchain.py`

**Interfaces:**
- Produces `audit_conda_cuda_toolchain`, `build_toolchain_environment`, `reject_non_conda_compiler_log`, and manifest/hash helpers.

- [ ] Write tests that reject system nvcc, mismatched CUDA_HOME/CUDACXX, CPU fallback, missing SM89, cross-environment extension caches, and missing compiler hashes.
- [ ] Run the focused tests and verify RED failures due to missing contract functions.
- [ ] Implement the minimal fail-closed contract and rerun focused tests GREEN.
- [ ] Run the current CoBEVT contract regression tests.

### Task 2: SmoothQuant Metrics and Export Debugging

**Files:**
- Modify: `search/model_families/lidar_cobevt/minimal_structure_quant_latency.py`
- Modify: `search/orchestration/lidar_cobevt_smoothquant_projection.py`
- Create: `search/orchestration/lidar_cobevt_smoothquant_conda_toolchain.py`
- Modify: `tests/test_lidar_cobevt_smoothquant_conda_toolchain.py`

**Interfaces:**
- Consumes the toolchain manifest from Task 1.
- Produces deterministic alpha selection over projection relative-L2, QK relative-L2, and Softmax JS; staged E0–E5 export records; explicit requested/realized admission decisions.

- [ ] Add RED tests for the five-alpha grid, normalized three-metric score, smoothing scale completeness, Q/K DQ-to-FP32 contract, native INT8-QK rejection, realized mismatch rejection, and fresh-process stage lineage.
- [ ] Implement only the tested metric and orchestration changes.
- [ ] Re-run focused and existing SmoothQuant tests.

### Task 3: Execute Toolchain and SmoothQuant Evidence

**Artifacts:**
- Create fresh `environment/`, `modelopt_extension/`, `smoothquant/`, `export_debug/`, `tensorrt/`, `evaluation/`, `latency/`, and `failures/` directories under the run root.

- [ ] Audit the resolved modelopt prefix and Conda package state.
- [ ] If necessary, perform a guarded CUDA 11.8 compiler dry-run/install that cannot change core packages.
- [ ] Build, inspect, load, and execute a minimal SM89 cpp_extension using only Conda nvcc.
- [ ] Trigger and audit the ModelOpt CUDA backend with no CPU fallback.
- [ ] Freshly rerun SQ1/SQ2 alpha calibration for `[0.5, 0.6, 0.7, 0.75, 0.8]`; gate SQ3 on SQ2 full success.
- [ ] Execute E0–E5 staged exports in fresh processes and isolate the first failing module if any stage crashes.
- [ ] Build only valid ONNX candidates with strongly typed TensorRT 10.9 and audit realized projection/QK precision.
- [ ] Run smoke10, fixed50, at most one fixed500, and isolated latency only for candidates passing all prior gates.

### Task 4: S2 Fixed500 and Final Contract

**Files:**
- Modify: `search/reporting/cobevt_minimal_structure_quant_latency.py` only if a reusable report helper is required.
- Create: `docs/codex_handoffs/4090-cobevt-smoothquant-conda-toolchain-report.md`

**Artifacts:**
- Create `structure_followup/s2_fixed500.json`, `structure_pareto.csv`, `structure_pareto.md`, `final_smoothquant_structure_contract.json`, and `root_conclusion.md`.

- [ ] Verify S2 retained-index, structure, profile, ONNX, engine, checkpoint, and manifest provenance.
- [ ] Run S2 fixed500 with 500/500 and zero skips, rebuilding only if provenance requires it and never changing its mask.
- [ ] Preserve `unit_lut=not_additive` and use full-engine action anchors.
- [ ] Generate the final contract/report from actual gate outcomes.

### Task 5: Verification and Delivery

**Files:**
- Test all modified Python modules and reports.

- [ ] Run all new tests and the existing CoBEVT regression suite.
- [ ] Run `py_compile` on every changed Python file and `git diff --check`.
- [ ] Verify output provenance, no system nvcc strings in successful build logs, and no Pyramid process mutation.
- [ ] Commit, fetch/reconcile the target remote branch, push without force, and verify a clean synchronized worktree.
