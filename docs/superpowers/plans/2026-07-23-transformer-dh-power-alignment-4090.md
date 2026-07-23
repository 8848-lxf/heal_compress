# Transformer d_h Power Alignment 4090 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Execute and audit the power-of-two and high-order `d_h` alignment matrix for CoBEVT and V2XViT on RTX 4090 under P32/P16/P8.

**Architecture:** Additive 4090 modules validate the local toolchain, generate deterministic candidates, invoke the existing H800-tested physical/precision primitives, schedule resumable GPU queues, measure isolated same-profile latency, and write compact evidence. Existing H800 production files remain unchanged.

**Tech Stack:** Python 3.9/3.10, PyTorch CUDA 11.8, ModelOpt 0.29, ONNX, TensorRT 10.9, pytest.

## Global Constraints

- Work only on `feature/h800-transformer-dh-alignment-sweep` in the isolated 4090 worktree.
- Preserve `feature/heal-unified-search-h800` at the recorded remote HEAD.
- Use GPUs 4/5/6/7 without killing external processes.
- Do not run GA, Greedy, or full1789.
- Do not commit checkpoint, ONNX, engine, calibration cache, or `/data` output.
- Label all new runtime evidence RTX 4090/SM89, not H800.

---

### Task 1: Candidate and metric contracts

**Files:**
- Create: `search/model_families/transformer/dh_power_alignment_4090.py`
- Test: `tests/test_transformer_dh_power_alignment_4090.py`

**Interfaces:**
- Produces `power_alignment_widths`, `alignment_traits`, `joint_candidates`, `speedup_metrics`, `neighbor_advantage`, and `search_candidate_gate`.

- [ ] Write failing tests for exact powers, divisibility, projection-width alignment, deduplication, bounds, neighbors, baseline semantics, replay drift, and search eligibility.
- [ ] Run the focused test and confirm the new module import fails.
- [ ] Implement immutable candidate records and fail-closed metric gates.
- [ ] Run the focused test and confirm all contract tests pass.
- [ ] Commit the candidate contract.

### Task 2: 4090 runtime and provenance

**Files:**
- Create: `search/orchestration/lidar_transformer_dh_power_alignment_4090.py`
- Test: `tests/test_transformer_dh_power_alignment_4090.py`

**Interfaces:**
- Consumes the existing H800 inventory/build/evaluate/joint functions.
- Produces `configure_4090_runtime`, `write_4090_provenance`, candidate manifests, and resumable queue specifications.

- [ ] Add failing tests rejecting system `nvcc`, wrong SM architecture, missing TRT/plugin, timing-cache reuse, and formal-search branch drift.
- [ ] Implement local environment discovery and in-process constant adaptation without editing H800 modules.
- [ ] Implement single-family and explicit joint queue manifests with GPU ownership.
- [ ] Run focused and existing Transformer tests.
- [ ] Commit the 4090 runtime and scheduler.

### Task 3: Compact reporter and latency gates

**Files:**
- Create: `search/reporting/transformer_dh_power_alignment_4090.py`
- Modify: `tests/test_transformer_dh_power_alignment_4090.py`

**Interfaces:**
- Consumes structure/build/evaluation/formal-latency artifacts.
- Produces every requested CSV/JSON contract and `root_conclusion_power_alignment.md`.

- [ ] Add failing tests for three speedup definitions, neighbor controls, diagnostic unsafe rows, repeat stability, and compact evidence selection.
- [ ] Implement aggregation, precision interaction, alignment benefit, and final contract logic.
- [ ] Verify missing or conflicting evidence fails closed.
- [ ] Commit the reporter.

### Task 4: Execute structure, build, and evaluation matrix

**Files:**
- Output only: `/data/lxf/heal_data/outputs/h800_transformer_dh_power_alignment_<timestamp>/`

**Interfaces:**
- Produces fresh manifests, rankings, physical structures, ONNX, P32/P16/P8 engines, Inspector evidence, and smoke10/fixed50/fixed500 results.

- [ ] Generate model-specific manifests and fixed-K contracts with eight workers.
- [ ] Generate inventory and one fixed Taylor ranking per model.
- [ ] Start four disjoint serial queues on GPUs 4/5/6/7.
- [ ] Monitor failures, preserve evidence, and resume only incomplete queue items.
- [ ] Confirm all accepted fixed500 rows are 500/500 with zero skip.

### Task 5: Formal latency and independent builds

**Files:**
- Output only under `formal_latency/`, `inspector/`, and `scheduler/`.

**Interfaces:**
- Consumes accepted engines and candidate metadata.
- Produces same-profile latency, total/precision speedups, neighbor comparisons, and repeat-build stability.

- [ ] Select an externally idle RTX 4090 and record five-minute isolation.
- [ ] Measure P32/P16/P8 baselines, candidates, and replay serially.
- [ ] Apply the `max(1%, 3*CV, replay drift)` gate.
- [ ] Fresh-build beneficial candidates and controls three times without timing-cache reuse.
- [ ] Re-measure and classify build stability.

### Task 6: Final evidence, verification, and push

**Files:**
- Create: `docs/codex_handoffs/4090_TRANSFORMER_DH_POWER_ALIGNMENT_20260723.md`

**Interfaces:**
- Produces the final compact Git evidence and remote branch update.

- [ ] Generate all requested result matrices, final contract, and root conclusion.
- [ ] Run focused tests and existing Transformer regressions.
- [ ] Run `python -m compileall` and `git diff --check`.
- [ ] Verify no large artifacts are staged and the formal-search branch HEAD is unchanged.
- [ ] Commit reports and push `feature/h800-transformer-dh-alignment-sweep` without force.

