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

## Round 002: Lazy Model-Family Dispatch And Pyramid Freeze Guard

### Implementation

- Added `search/model_families/contracts.py` with the minimal `SearchRunner`
  protocol and a canonical, family-specific capability manifest hash.
- Added `search/model_families/registry.py` with fail-closed family parsing and
  lazy runner imports.
- Updated only the runner construction point in `search/cli.py`.
- Preserved the existing `LidarPyramidTwoStageSearch` symbol and constructor
  arguments so old configs and tests retain their exact default behavior.
- CoBEVT modules are not imported when the selected/default family is
  `lidar_pyramid`.

### Tests added

- `tests/test_model_family_registry.py`
  - missing family defaults to pyramid;
  - CoBEVT dispatch does not construct pyramid;
  - unknown family fails closed;
  - capability hashes are deterministic and family-specific.
- `tests/test_pyramid_family_freeze.py`
  - default dispatch never loads CoBEVT;
  - explicit pyramid family passes the same resume/constructor contract.

### RED and GREEN evidence

- RED: `6 failed`, all caused by the intentionally missing
  `search.model_families` package.
- GREEN command:

```bash
conda run --no-capture-output -n univ2x-opt pytest -q \
  tests/test_model_family_registry.py \
  tests/test_pyramid_family_freeze.py \
  tests/test_search_tool_adapters.py
```

- GREEN result: `17 passed in 1.66s`.
- Modified Python files compile successfully.
- `git diff --check` passes.

### Status

- `MODEL_FAMILY_DISPATCH_IMPLEMENTED=true`
- `DEFAULT_PYRAMID_DISPATCH_PRESERVED=true`
- `PYRAMID_QUANTIZATION_PATH_MODIFIED=false`
- `COBEVT_MODULE_LAZY_LOAD=true`

--- ROUND 002 COMPLETE | 2026-07-18T02:54:18+08:00 ---

## Round 003: CoBEVT Model Identity And Fixed-K Contract

### Implementation

- Added `search/model_families/lidar_cobevt/model_capability.py`.
  - Hashes and validates the exact checkpoint/config before model creation.
  - Requires `heter_model_baseline` with `fusion_method=cobevt`.
  - Loads through the HEAL model factory while preserving missing/unexpected
    state-key evidence.
  - Exposes a shared point-pillar scatter capability that is floating point and
    excluded from precision genes.
- Added `search/model_families/lidar_cobevt/input_contract.py`.
  - Derives fixed K from an ordered manifest's positive voxel counts.
  - Rounds to an explicit alignment and records source maximum, overflow count,
    record count, and a stable manifest hash.
  - Contains no pyramid `29696` default.

### Real checkpoint evidence

- Checkpoint SHA256:
  `67b0f2f00d74fea4912b4fdb902c1150146c733201e5dc36d3358f5ba605cfd4`
- Config SHA256:
  `a0ee9d64fd1b01af95b1c997937ab07e0e810440236accfb598d5462ba086c33`
- Checkpoint state keys: 224
- Constructed model state keys: 224
- Missing keys: 0
- Missing parameter keys: 0
- Unexpected keys: 0
- Fusion module: `CoBEVT`
- Scatter module: `encoder_m1.scatter` / `PointPillarScatter`
- Scatter BEV output shape: `[64, 256, 512]`
- Scatter allowed boundary dtypes: FP32, FP16
- Scatter precision gene: false

Actual train200/fixed50 fixed K remains pending manifest scanning; no fixed K
value is claimed in this round.

### Tests

- RED: 10 failures caused by the missing CoBEVT package; a real checkpoint load
  test was added before implementation.
- GREEN: `11 passed` for model/fixed-K tests.
- Combined model-family regression: `17 passed in 7.56s`.
- Warnings are existing third-party `timm`, matplotlib pyparsing, and
  `torch.meshgrid` deprecations; no model load or coverage warning occurred.
- Modified Python files compile successfully.
- `git diff --check` passes.

### Status

- `COBEVT_CHECKPOINT_IDENTITY_PASS=true`
- `COBEVT_CHECKPOINT_FULL_WEIGHT_COVERAGE=true`
- `COBEVT_SCATTER_CAPABILITY_IDENTIFIED=true`
- `COBEVT_FIXED_K_ALGORITHM_PASS=true`
- `COBEVT_ACTUAL_MANIFEST_FIXED_K_AVAILABLE=false`

--- ROUND 003 COMPLETE | 2026-07-18T03:00:57+08:00 ---
