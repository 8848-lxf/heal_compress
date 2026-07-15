# Joint Taylor GA Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a unified pruning/quantization Taylor proxy, calibrate its fixed exponential scale from full-validation anchors, and validate a fresh generation-0 Top-5 without starting Stage A or B.

**Architecture:** New proxy and anchor modules own the mathematical contracts and artifact schemas. Existing GA, repair, deployment, and Stage-2 paths receive narrow adapters so their established semantics remain unchanged.

**Tech Stack:** Python 3.10, PyTorch 2.0.1/CUDA 11.8, TensorRT 10.9 strongly typed explicit-QDQ deployment, pytest, YAML/JSON/CSV.

## Global Constraints

- Work only on `feature/heal-compress-h800-sync-4090`; preserve ancestor `b862b3d8ad061bd12580776226c75f564918298d`.
- Do not alter tracer, pruning dependency graph, genotype semantics, typed-QDQ realization, or Stage-2 F2.
- Use `univ2x-opt` for code/tests/Fisher and `modelopt` with TensorRT library paths for deployment.
- Use GPUs 4/5/6/7 for parallel deployment and one idle 4090 for isolated formal latency.
- Do not start Stage A or Stage B.

---

### Task 1: Unified Taylor mathematics and Fisher statistics

**Files:**
- Create: `search/proxy/joint_taylor.py`
- Create: `search/proxy/fisher_statistics.py`
- Modify: `search/proxy/candidate_perturbation.py`
- Test: `tests/test_joint_taylor_proxy.py`

**Interfaces:**
- Produces `JointTaylorProxy.evaluate(phenotype) -> JointTaylorResult`, `conditional_group_costs(phenotype)`, `collect_fisher_statistics(...)`, and manifest serialization.

- [ ] Write failing tests for pruned/quantized deltas, cross-layer precision, overlap deduplication, `mean(g^2)`, first/second order reproducibility, no normalization, and SQNR exclusion.
- [ ] Run `conda run -n univ2x-opt pytest -q tests/test_joint_taylor_proxy.py` and confirm feature-missing failures.
- [ ] Implement boolean-union slice masks, per-module pseudo quantization, float64 accumulation, Fisher collection, and audit rows.
- [ ] Re-run the focused test and existing Fisher tests to green.
- [ ] Commit as `feat: add unified pruning quantization taylor proxy`.

### Task 2: Exponential task score and conditional repair

**Files:**
- Create: `search/proxy/task_score.py`
- Modify: `search/stage1/repair_selection.py`
- Modify: `search/orchestration/lidar_pyramid_search.py`
- Test: `tests/test_joint_taylor_stage1.py`

**Interfaces:**
- Produces `compute_exponential_j1`, immutable `ProxyScale`, conditional dense/grouped repair reports, and raw/repaired metric lineage.

- [ ] Write failing tests for J1 direction, exact score landmarks, saturation, fixed tau, raw mask preservation, dense floor repair, independent grouped repair, precision identity, and post-repair recomputation.
- [ ] Confirm focused RED failures.
- [ ] Implement monotonic shortlist repair using `I_g(b)`, recompute joint score/parameters/BOPS, and preserve the existing hard BOPS gate.
- [ ] Re-run focused and repair/grouped/BOPS regression tests.
- [ ] Commit as `feat: integrate exponential joint proxy into stage1 ga` after tau support exists.

### Task 3: Global anchor structure sweep

**Files:**
- Create: `search/anchors/joint_taylor_sweep.py`
- Create: `search/configs/lidar_pyramid_4090_joint_taylor_anchor_sweep.yaml`
- Modify: `search/anchors/runner.py`
- Test: `tests/test_joint_taylor_anchor_sweep.py`

**Interfaces:**
- Produces legal anchor structure manifests, global ranking/audit tables, three-precision deployment requests, and formal-latency isolation checks.

- [ ] Write failing tests for requested/realized prune rates, 0.8 domain cap, nearest legal tie-break, shared physical hashes, and concurrency rejection.
- [ ] Confirm focused RED failures.
- [ ] Implement deterministic global ranking, legal structure selection, artifact writers, engine matrix orchestration, and serial formal-latency replay.
- [ ] Re-run focused plus physical/grouped/worker/typed-QDQ tests.
- [ ] Commit as `feat: add global joint taylor anchor sweep`.

### Task 4: Tau calibration

**Files:**
- Create: `search/proxy/tau_calibration.py`
- Test: `tests/test_joint_taylor_tau_calibration.py`

**Interfaces:**
- Produces `calibrate_tau(anchor_rows, ...) -> ProxyScale`, bisection requests, distribution diagnostics, and `proxy_scale.json`.

- [ ] Write failing tests for absolute 0.1 drop, safe-boundary tie-breaks, under-resolution, no-safe-boundary failure, exact score landmarks, and immutable serialization.
- [ ] Confirm focused RED failures.
- [ ] Implement selection, validation, diagnostics, stable hashing, and fail-closed gate.
- [ ] Re-run focused tests.
- [ ] Commit as `feat: calibrate exponential task score from map boundary`.

### Task 5: Execute anchors and generation-0

**Files:**
- Create: timestamped ignored `outputs/4090_global_joint_taylor_anchor_sweep_<timestamp>/`
- Create: `search/configs/lidar_pyramid_4090_joint_taylor_generation0.yaml`
- Modify: `search/constrained/population.py`
- Modify: `search/orchestration/lidar_pyramid_search.py`
- Test: `tests/test_joint_taylor_generation0.py`

**Interfaces:**
- Consumes fixed `proxy_scale.json` and valid anchor masks; produces a fresh Top-5 report while leaving Stage A/B stopped.

- [ ] Run Fisher collection and validate its manifest.
- [ ] Run the 0.0-0.7 three-precision sweep and any required boundary bisections on GPUs 4-7.
- [ ] Stop all workers and replay formal latency serially on one idle GPU.
- [ ] Calibrate and freeze tau; fail closed if invalid.
- [ ] Add anchor-derived/current-safe/random/mixed seed families and verify uniqueness.
- [ ] Run fresh generation-0, repaired BOPS backfill, smoke10, and formal200 Top-5.
- [ ] Confirm `STAGE_A_STARTED=false` and `STAGE_B_ALLOWED=false`.

### Task 6: Verification, reports, and delivery

**Files:**
- Create: `docs/codex_handoffs/4090-joint-taylor-anchor-generation0-report.md`
- Create: `docs/greedy_joint_compression_design.md`
- Modify: `docs/codex_handoffs/4090_GA_EXPLICIT_QDQ_PROGRESS_20260714.md`

- [ ] Run all requested focused and search regression suites in `univ2x-opt`.
- [ ] Run `py_compile` for every changed Python file and `git diff --check`.
- [ ] Record unavailable tools as environment evidence rather than code failure.
- [ ] Write complete anchor/tau/Top-5 evidence and append a timestamped progress round.
- [ ] Commit reports, verify branch/ancestor/status, fetch remote, and push only `feature/heal-compress-h800-sync-4090` without force.

