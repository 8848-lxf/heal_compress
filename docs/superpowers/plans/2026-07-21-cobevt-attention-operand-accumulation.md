# CoBEVT Attention Operand and Accumulator Precision Audit Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce a reproducible Level-A-evidence audit of CoBEVT QK and AV operand/accumulator precision, deployment safety, and latency on TensorRT 10.9.

**Architecture:** Separate immutable precision contracts, local TensorRT capability evidence, exact cuBLASLt/CUTLASS oracle execution, native TensorRT realization, optional plugin-oracle realization, full-model gated evaluation, and final reporting. Each stage emits hashed evidence consumed by the next stage and fails closed on unsupported or unknown semantics.

**Tech Stack:** Python 3, PyTorch, ONNX, TensorRT 10.9.0.34, CUDA 11.8, cuBLASLt, optional CUTLASS, HEAL/OpenCOOD evaluation, RTX 4090 SM89.

## Global Constraints

- Base is `70f93224691a17c8174d44ce73fe3b7ef6d3383f` on an isolated branch and worktree.
- CUDA builds use only `/home/lixingfeng/anaconda3/envs/modelopt/bin/nvcc`; system nvcc is rejected.
- Pyramid GA processes and code paths are not modified, stopped, or restarted.
- No 1789-frame evaluation is run; fixed500 is capped at six gated profiles.
- Only Level-A accumulator evidence can create a formal search gene.
- Native TensorRT, materialized-Cast, fused-unknown, and plugin-oracle phenotypes remain distinct.

---

### Task 1: Precision contracts and evidence policy

**Files:**
- Create: `search/model_families/lidar_cobevt/attention_compute_contract.py`
- Create: `search/model_families/lidar_cobevt/attention_accumulation_profiles.py`
- Test: `tests/test_lidar_cobevt_attention_operand_accumulation.py`

**Interfaces:**
- Produces: immutable phenotype/profile dataclasses, evidence classification, Cast materialization classification, search eligibility.

- [ ] Write failing tests for field separation, F3/R1/I8 classification, evidence levels, Cast materialization, and search fail-closed behavior.
- [ ] Run the focused test and verify failures are due to missing modules.
- [ ] Implement the minimum immutable contracts and validators.
- [ ] Run the focused test and existing accumulation tests.
- [ ] Commit the green contract layer.

### Task 2: Installed TensorRT and toolchain capability audit

**Files:**
- Create: `search/orchestration/lidar_cobevt_attention_accumulation_capability.py`
- Modify: `tests/test_lidar_cobevt_attention_operand_accumulation.py`

**Interfaces:**
- Consumes: phenotype/evidence schema from Task 1.
- Produces: `installed_tensorrt_api_inventory.json`, header/release evidence, and `precision_capability_matrix.csv`.

- [ ] Add failing fixtures for API/header/symbol parsing and absence of accumulator APIs.
- [ ] Verify RED.
- [ ] Implement local-only API/header/library/sample/release-note inventory with hashes.
- [ ] Run focused tests and a real modelopt-environment capability audit.
- [ ] Commit the capability audit.

### Task 3: Real Attention capture and numerical metrics

**Files:**
- Create: `search/orchestration/lidar_cobevt_capture_attention_tensors.py`
- Create: `search/model_families/lidar_cobevt/attention_numerical_boundary.py`
- Modify: `tests/test_lidar_cobevt_attention_operand_accumulation.py`

**Interfaces:**
- Produces: six-block captured tensor bundles, hashes/statistics, probe tensors, QK/Softmax/AV metrics.

- [ ] Add failing tests for six-block completeness, tensor hashes, statistics, and metric edge cases.
- [ ] Verify RED.
- [ ] Implement hooks, manifest binding, statistics, probes, and metrics.
- [ ] Run tests and capture smoke10 on a free GPU.
- [ ] Commit capture/metric support.

### Task 4: Exact cuBLASLt/CUTLASS oracle

**Files:**
- Create: `search/orchestration/lidar_cobevt_attention_gemm_oracle.py`
- Create: `search/model_families/lidar_cobevt/cuda_oracle_contract.py`
- Create: `plugins/cobevt_attention_gemm_oracle/CMakeLists.txt`
- Create: `plugins/cobevt_attention_gemm_oracle/oracle.cu`
- Modify: `tests/test_lidar_cobevt_attention_operand_accumulation.py`

**Interfaces:**
- Produces: serialized exact compute contracts and oracle outputs/timings for O0-O4 and V0-V4.

- [ ] Add failing tests for cuBLASLt/CUTLASS type contracts and system-nvcc rejection.
- [ ] Verify RED.
- [ ] Implement modelopt-nvcc build and cuBLASLt execution; mark unsupported combinations explicitly.
- [ ] Execute real captured tensors and discriminative probes on free GPUs.
- [ ] Commit exact-oracle support.

### Task 5: Native TensorRT micro-engine matrix

**Files:**
- Create: `search/orchestration/lidar_cobevt_attention_trt_microbench.py`
- Create: `search/model_families/lidar_cobevt/attention_trt_realization.py`
- Modify: `tests/test_lidar_cobevt_attention_operand_accumulation.py`

**Interfaces:**
- Consumes: six-block shapes/captures and oracle fingerprints.
- Produces: M0-M5/N0-N5 engine evidence, Cast materialization, native phenotype classification.

- [ ] Add failing tests for native-vs-materialized-vs-unknown and INT8-DQ rejection.
- [ ] Verify RED.
- [ ] Implement strongly typed fresh builds, inspector/profiler parsing, and Level A/B/C classification.
- [ ] Execute the six-shape matrix and numerical comparisons.
- [ ] Commit native micro-engine support.

### Task 6: Plugin oracle fallback

**Files:**
- Create: `plugins/qk_mixed_accum/CMakeLists.txt`
- Create: `plugins/qk_mixed_accum/qk_mixed_accum_plugin.h`
- Create: `plugins/qk_mixed_accum/qk_mixed_accum_plugin.cpp`
- Create: `plugins/qk_mixed_accum/qk_mixed_accum_kernel.cu`
- Create: `plugins/av_mixed_accum/CMakeLists.txt` only if AV native evidence is insufficient.
- Modify: `search/orchestration/lidar_cobevt_attention_trt_microbench.py`
- Modify: `tests/test_lidar_cobevt_attention_operand_accumulation.py`

**Interfaces:**
- Produces: separately labelled Level-A `plugin_oracle` F16A32/I8A32I realization.

- [ ] Add failing serialization/type/shape tests.
- [ ] Verify RED.
- [ ] Implement deterministic cuBLASLt plugins with explicit SM89 compiler provenance.
- [ ] Build, deserialize, run, and compare against exact oracle.
- [ ] Commit plugin-oracle support.

### Task 7: Full-model gated profiles and evaluation

**Files:**
- Create: `search/orchestration/lidar_cobevt_attention_accumulation_full_model.py`
- Modify: `search/model_families/lidar_cobevt/attention_accumulation_profiles.py`
- Modify: `tests/test_lidar_cobevt_attention_operand_accumulation.py`

**Interfaces:**
- Consumes: validated native/plugin phenotypes.
- Produces: A0-A5/B1-B3/C1-C3 build, smoke10, fixed50, and selected fixed500 results.

- [ ] Add failing tests for gate order, fixed500 cap, threshold classification, and profile isolation.
- [ ] Verify RED.
- [ ] Implement fresh export/build/audit/evaluation orchestration using accepted manifests.
- [ ] Execute gated smoke10/fixed50 and at most six fixed500 profiles.
- [ ] Commit full-model orchestration.

### Task 8: Formal latency and final reports

**Files:**
- Create: `search/reporting/cobevt_attention_accumulation_boundary.py`
- Modify: `tests/test_lidar_cobevt_attention_operand_accumulation.py`
- Create: `docs/codex_handoffs/4090-cobevt-attention-operand-accumulation-report.md`

**Interfaces:**
- Produces: all core CSV/JSON/Markdown matrices, search contract, isolated latency evidence, root conclusion.

- [ ] Add failing tests for safety matrices, search eligibility, latency isolation, and required report fields.
- [ ] Verify RED.
- [ ] Implement report/contract writers and isolated latency runner.
- [ ] Measure eligible engines on one isolated GPU and generate final artifacts.
- [ ] Run full CoBEVT tests, py_compile, and diff checks; commit and push the branch.
