# 4090 HEAL Model-Family CoBEVT Search And Deployment Progress

This file is the append-only execution handoff for the CoBEVT model-family
compatibility work. Large model, ONNX, TensorRT engine, calibration, and tensor
artifacts remain outside Git. Each completed work round ends with a timestamped
separator and records the exact branch, commits, code paths, tests, experiment
evidence, and unresolved blockers.

## Round 001: Architecture Approval And Isolated Branch Setup

### Scope completed

- Audited the current `lidar_pyramid` search/deployment ownership boundaries.
- Audited the HEAL `HeterModelBaseline` CoBEVT topology, including the shared
  point-pillar encoder/scatter and the window/grid attention fusion blocks.
- Audited the older CoBEVT physical-pruning implementation as evidence for
  attention-head-aligned QKV/FFN/LayerNorm slicing; it will not become a runtime
  dependency of this repository.
- Selected the frozen-pyramid plus model-family-recipe architecture.
- Wrote and pushed the approved architecture specification.
- Created an isolated worktree so the active pyramid search cannot read
  partially implemented CoBEVT modules from newly spawned workers.
- Ran the legal-width and CLI dispatch baseline regression before changing
  production code.

### Branch and lineage

- Original production branch:
  `feature/heal-compress-h800-sync-4090`
- Isolated CoBEVT branch:
  `feature/heal-compress-4090-cobevt-family`
- CoBEVT branch base commit:
  `6d184ac0dc2c459cf713e33a3dae5698863c32ee`
- Approved design commit:
  `202b669`
- Required H800 ancestor:
  `b862b3d8ad061bd12580776226c75f564918298d`
- H800 ancestor check: passed
- Worktree:
  `/home/lixingfeng/UniAD_examine/heal_compress/.worktrees/cobevt-family`

### Design decisions

- Existing pyramid typed-ONNX/QDQ/TensorRT/evaluation modules remain the
  production implementation and are not rewritten.
- Model-family dispatch happens before runner construction. Old configs without
  a family field continue to resolve to the existing pyramid runner.
- CoBEVT receives separate pruning, export, canonical precision, merge/QDQ,
  plugin-capability, and evaluation recipes while reusing validated common
  search/cache/builder/process infrastructure.
- `PointPillarScatterTRT` is probed for reuse and stays floating point outside
  precision genes.
- CoBEVT attention is first lowered to ONNX-native operators. A new plugin is
  permitted only after a minimal strongly typed parser or parity reproducer
  proves it is necessary.
- The missing requested TensorRT path is not used silently. The installed path
  is `/home/lixingfeng/UniAD_examine/HEAL/prune_model/TensorRT-10.9_x86_cu118`.
- The bounded compatibility experiment is one feasible BOPS band, one greedy
  endpoint, and GA population 16 x 3 generations x one seed with generation
  Top-2, followed by real strongly typed engine smoke10 and fixed50 GPU
  evaluation.

### Files added or changed

- `docs/superpowers/specs/2026-07-17-heal-model-family-deployment-design.md`
  defines the approved architecture, operator/plugin decision policy,
  strongly typed deployment sequence, cache identity, bounded smoke, and
  fail-closed rules.
- `.gitignore` now ignores `.worktrees/` to keep the isolated checkout out of
  repository status.
- `docs/4090_HEAL_MODEL_FAMILY_COBEVT_PROGRESS_20260717.md` is this append-only
  implementation and experiment record.

### Baseline verification

Command:

```bash
conda run --no-capture-output -n univ2x-opt pytest -q \
  tests/test_legal_width_inventory.py \
  tests/test_legal_width_genotype.py \
  tests/test_deterministic_width_decoder.py \
  tests/test_precision_structure_orthogonality.py \
  tests/test_search_tool_adapters.py
```

Result: `22 passed in 2.16s`.

### Active pyramid experiment isolation

The existing six-budget pyramid GA was not stopped or restarted. At the Round
001 audit it had completed budgets 0.05, 0.10, and 0.15 and had entered budget
0.20 generation 1. Its original worktree remains on
`feature/heal-compress-h800-sync-4090`.

### Status

- `COBEVT_DESIGN_APPROVED=true`
- `ISOLATED_BRANCH_CREATED=true`
- `PYRAMID_DEPLOYMENT_PATH_MODIFIED=false`
- `COBEVT_IMPLEMENTATION_STARTED=false`
- `COBEVT_STRONGLY_TYPED_ENGINE_PASS=false`
- `COBEVT_GREEDY_SMOKE_PASS=false`
- `COBEVT_GA_SMOKE_PASS=false`

--- ROUND 001 COMPLETE | 2026-07-18T02:43:53+08:00 ---

