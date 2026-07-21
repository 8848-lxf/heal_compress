# CoBEVT Native F16A32 Tactic Pinning Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Enumerate, pin, and audit native TensorRT 10.9 tactics for all real CoBEVT QK/AV shapes without confusing tactic pinning with accumulator proof.

**Architecture:** A Conda-built C++ helper follows TensorRT's `sampleEditableTimingCache` to emit profiling records and edit timing-cache entries. Python modules validate provenance, classify evidence, orchestrate micro/full-engine gates, and write machine-readable matrices. Unsupported or Level-B-only results remain experimental.

**Tech Stack:** TensorRT 10.9.0.34, CUDA 11.8, modelopt Conda nvcc, Python, C++ TensorRT API, ONNX, PyTorch, HEAL CoBEVT evaluator.

## Global Constraints

- Branch `feature/cobevt-native-f16a32-tactic-pinning`, base `ea342d49526355094d92b4b4631bb3ee596e1e6a`.
- Use only `/home/lixingfeng/anaconda3/envs/modelopt/bin/python` and its `nvcc`.
- Keep Pyramid GA and other branches/processes untouched.
- No plugin is native evidence; no 1789-frame validation.
- Exact GPU SM89/TRT10.9/CUDA/shape/graph/cache provenance is mandatory.

### Task 1: Contracts and capability evidence

**Files:**
- Create: `search/model_families/lidar_cobevt/native_tactic_contract.py`
- Create: `search/orchestration/lidar_cobevt_native_tactic_capability.py`
- Test: `tests/test_lidar_cobevt_native_f16a32_tactic_pinning.py`

- [ ] Write failing tests for output/accumulator phenotype names, Level-A/B/C gates, cache compatibility, and partial 6/6 rejection.
- [ ] Implement immutable contracts and local TRT header/Python/sample inventory.
- [ ] Run focused tests and record installed API evidence.

### Task 2: Editable timing-cache helper

**Files:**
- Create: `plugins/cobevt_editable_timing_cache/CMakeLists.txt`
- Create: `plugins/cobevt_editable_timing_cache/tactic_probe.cpp`
- Modify: `tests/test_lidar_cobevt_native_f16a32_tactic_pinning.py`

- [ ] Add failing tests for Conda compiler provenance and profiling/cache record parsing.
- [ ] Implement sample-compatible key/tactic parsing, editable-cache build, profiling log extraction, and cache update.
- [ ] Compile with Conda nvcc/g++ on SM89 and run one canonical micrograph before the full matrix.

### Task 3: Real-shape micro-engine matrix

**Files:**
- Create: `search/orchestration/lidar_cobevt_native_tactic_microbench.py`
- Create: `search/model_families/lidar_cobevt/native_tactic_evidence.py`

- [ ] Add fixtures for QK/AV shape signatures and tactic table classification.
- [ ] Build 12 fresh primitive engines with editable cache and detailed inspector/profiling logs.
- [ ] Enumerate candidate records, preserve partial completeness, and run oracle fingerprints on selected tactics.

### Task 4: Cache edit and deterministic pinning

**Files:**
- Create: `search/orchestration/lidar_cobevt_tactic_pinning.py`
- Modify: `tests/test_lidar_cobevt_native_f16a32_tactic_pinning.py`

- [ ] Add failing tests for key/tactic mismatch, stale cache rejection, and three-build stability.
- [ ] Edit only official `ITimingCache` key/value records, rebuild each selected shape three times, and verify realized tactic IDs.
- [ ] Write cache edit manifests, hashes, and shape-level pinning matrix.

### Task 5: Full-model gate and bounded evaluation

**Files:**
- Create: `search/orchestration/lidar_cobevt_native_f16a32_full_model.py`
- Create: `search/reporting/cobevt_native_f16a32_report.py`

- [ ] Add failing tests for micro-to-full preservation and fixed500 safety classification.
- [ ] Build only exact 6/6 native profiles; reject partial, plugin, or materialized-Cast candidates.
- [ ] Run smoke10/fixed50 and at most six fixed500 profiles, then isolated formal latency only for exact Level-A candidates.

### Task 6: Reports, regression, commit, push

- [ ] Generate all requested CSV/JSON/Markdown artifacts and final search contract.
- [ ] Run all CoBEVT tests, py_compile, and diff checks.
- [ ] Commit implementation/results and push only `feature/cobevt-native-f16a32-tactic-pinning`.
