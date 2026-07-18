# Stage-2 Engine Build Cap Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Enforce at most five calibration/TensorRT build attempts per generation while moving physical BOPS rejection before engine construction.

**Architecture:** Split generation deployment into a physical preflight phase and a capped build/smoke phase. Preflight can scan ranked candidates without consuming engine slots; after no more than five preflight passes are selected, one concurrent build wave runs with no post-build backfill.

**Tech Stack:** Python 3.9, pytest, existing HEAL physical materializer, strongly typed TensorRT Stage-2 worker pool.

## Global Constraints

- Work only on `feature/heal-compress-h800-sync-4090`.
- Preserve ancestor `b862b3d8ad061bd12580776226c75f564918298d`.
- Do not force push or modify the H800 branch.
- Do not change the BOPS intervals, typed deployment recipe, candidate encoding, or F2 rule.
- Start the corrected search in a fresh timestamped output directory.

---

### Task 1: Lock The Generation Build Contract

**Files:**
- Modify: `tests/test_generation_stage2_candidate_counts.py`
- Modify: `search/orchestration/generation_stage2.py`

**Interfaces:**
- Consumes: ranked `ProxyCandidateRecord` rows and `BopsBandPolicy`.
- Produces: `run_generation_stage2(..., physical_preflight_batch_fn=...)` report with explicit preflight/build counts and skip reasons.

- [ ] Add tests proving failed build/smoke candidates are not replaced and the build callback receives at most `topk` rows.
- [ ] Run the focused tests and observe failure from the current success-count backfill loop.
- [ ] Replace the loop with ranked physical preflight followed by one capped build wave.
- [ ] Verify zero, one, and two-to-five candidate semantics.

### Task 2: Add Real Physical BOPS Preflight

**Files:**
- Modify: `search/stage2/lidar_pyramid_real_evaluator.py`
- Modify: `search/stage2/candidate_worker.py`
- Modify: `search/orchestration/legal_width_six_budget_ga.py`
- Modify: `tests/test_search_stage2_candidate_worker.py`

**Interfaces:**
- Consumes: serialized phenotype, target BOPS interval, physical model and legalized precision profile.
- Produces: `physical_preflight` worker result containing physical identity, physical BOPS, and a reusable artifact lineage for the selected build.

- [ ] Add a failing worker routing test for `physical_preflight`.
- [ ] Expose materialize-and-BOPS preflight without ONNX/QDQ/calibration/TRT.
- [ ] Route preflight through persistent workers and pass only admitted rows to `build_smoke`.
- [ ] Assert preflight rejection creates no engine or calibration artifact.

### Task 3: Verify And Record The Change

**Files:**
- Modify: `docs/codex_handoffs/4090_GA_EXPLICIT_QDQ_PROGRESS_20260714.md`
- Create: `docs/codex_handoffs/4090-stage2-engine-build-cap-20260718.md`

**Interfaces:**
- Consumes: focused test output and the stopped-run artifact counts.
- Produces: reproducible handoff entry and fresh restart command.

- [ ] Run generation, worker, BOPS, process-pool, and six-budget orchestration tests.
- [ ] Run `py_compile` for every changed Python file and `git diff --check`.
- [ ] Record old-run progress, root cause, new invariants, and restart command.
- [ ] Commit, verify branch/ancestor/remote divergence, and push only the 4090 branch.
- [ ] Start the corrected search in a new output directory and verify the first generation never submits more than five build tasks.
