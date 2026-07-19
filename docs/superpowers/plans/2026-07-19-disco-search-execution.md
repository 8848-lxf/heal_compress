# DiscoNet Search Execution Plan

> **For agentic workers:** Execute this plan inline with verification checkpoints; preserve the active Pyramid run.

**Goal:** Complete the DiscoNet Greedy result handoff, then run the approved single-seed six-budget real legal-width GA with at most five Stage-2 deployments per generation.

**Architecture:** Reuse the validated DiscoNet family adapter, legal-width decoder, strongly typed Stage-2 worker, and content-addressed artifact policy. The Greedy full-validation manifest is reused only as a seed/provenance source; the GA starts in a fresh timestamped output root with its own population and generation records. GPU7 is the requested deployment device, while active Pyramid processes remain untouched.

**Tech Stack:** Python/univ2x-opt for Stage-1 and orchestration, modelopt/TensorRT 10.9 for ONNX/QDQ/engine deployment, fixedK29696, GPU AP IoU, 500-frame Stage-2 and 1789-frame final validation.

## Global Constraints

- Work only on `feature/heal-compress-4090-cnn-softfusion-family`; preserve H800 ancestor `b862b3d8ad061bd12580776226c75f564918298d`.
- Do not modify, stop, or reuse active Pyramid processes or its output tree.
- Budgets are `0.05, 0.10, 0.15, 0.20, 0.25, 0.30`.
- GA uses one seed, population 64, offspring 64, 15 generations, and at most five Stage-2 candidates per generation.
- Candidates failing the BOPS admission gate never consume an engine slot; no duplicate backfill.
- Preserve only compact audit summaries in Git; keep ONNX/PTH/engine/cache outputs outside Git.

### Task 1: Commit the completed Greedy evidence

Files:
- Modify `docs/codex_handoffs/4090-CNN-SOFTFUSION-GENERALIZATION-20260719.md` with the six-budget Stage-1/Stage-2 results and artifact-reuse attempt evidence.
- Add `docs/codex_handoffs/4090-DISCONET-GREEDY-RESULTS-20260719.md` to the handoff set.

Verification:
- Confirm `greedy_endpoint_stage2_result.json` reports six successful reused builds and six `1789/1789, skipped=0` validations.
- Run focused regression tests, `py_compile`, and `git diff --check`; commit and push.

### Task 2: Launch the real DiscoNet GA

Files/config:
- Use `search/configs/lidar_disco_4090_joint_six_budget_ga.yaml` unchanged for the approved six budgets.
- Bind the validated Greedy `joint_loss_scale.json` and Greedy manifest only through CLI provenance.

Execution:
- Create a new timestamped output root under `/var/tmp/lxf/heal_data/outputs`.
- Launch `search.cli` in background with one seed, six budgets, 15 generations, GPU7, strongly typed Stage-2, and no external cache reuse.
- Monitor process health, generation/budget completion, Stage-2 counts, GPU7 occupancy, and Pyramid isolation.

### Task 3: Validate, compact, and report GA results

Verification:
- Parse every budget/generation record; confirm BOPS gate, at-most-five Stage-2 attempts, realized precision audits, and full-validation results.
- Produce compact CSV/JSON summaries and Pareto reports using only real full-validation metrics for official points.
- Run relevant tests, `py_compile`, `git diff --check`, append a timestamped handoff round, commit, and push.

F-Cooper remains deferred until its checkpoint is available and DiscoNet GA evidence is complete.
