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

## Round 004: CoBEVT Head-Aligned Legal Width And Physical Replay

### Implementation

- Added `search/model_families/lidar_cobevt/pruning_recipe.py`.
- The CoBEVT fusion domain is rooted at
  `shrinker_m1.layers.0.double_conv.2` and extends through all three window/grid
  attention stages, FFNs, LayerNorms, relative-position bias embeddings,
  fusion MLP head, and cls/reg/dir input channels.
- Legal widths for the current 256-dimensional embedding and `dim_head=32` are
  `[64, 96, 128, 160, 192, 224, 256]`, satisfying the 80% per-domain cap.
- Width decode consumes a fixed head ranking and has no precision argument.
- Smaller legal widths produce nested prune sets.
- Physical replay slices QKV output as three corresponding blocks, updates
  attention head metadata and positional-bias columns, and validates every
  dependent module after surgery.
- Unsupported/inconsistent fusion closures fail closed; they are not sent to
  normal repair.

### Real 256 -> 192 physical evidence

- Original attention heads: 8
- Physical attention heads: 6
- Original embedding: 256
- Physical embedding: 192
- Synchronized physical operations: 54
- Original parameters: 10,500,260
- Predicted physical parameters: 9,286,336
- Materialized physical parameters: 9,286,336
- Parameter retention: 0.8843910532
- Structure hash:
  `44bd08fac491bc74dbe3e26c0abbe8483c518504ea2f5145936233cfa75daf7e`
- Structure audit issues: 0
- Precision input to decoder: none

### Tests

- Initial RED: 7 failures and 2 setup errors for the missing pruning recipe.
- Additional RED: fusion-domain inventory test failed because the domain API
  did not yet exist.
- GREEN command covered CoBEVT widths/replay plus existing generic physical
  replay and legal-width inventory.
- GREEN result: `14 passed in 7.77s`.
- Modified Python files compile successfully.
- `git diff --check` passes.

### Status

- `COBEVT_FUSION_LEGAL_WIDTH_DOMAIN_PASS=true`
- `COBEVT_MASK_DEPENDS_ON_PRECISION=false`
- `COBEVT_PHYSICAL_PARAMETER_IDENTITY_PASS=true`
- `COBEVT_FUSION_FORWARD_PARITY_PENDING=true`

--- ROUND 004 COMPLETE | 2026-07-18T03:12:20+08:00 ---

## Round 005: Real CoBEVT Typed ONNX And Strongly Typed FP32 Capability

### Implementation

- Added `quantization/export/heal_lidar_cobevt.py` as a separate six-input
  CoBEVT export wrapper.
  - Reuses the accepted `PointPillarScatterTRT` plugin ABI without importing or
    modifying the pyramid export wrapper.
  - Adds `record_len` so attention masks distinguish real and padded agents.
  - Pads pairwise transforms to `max_cav=2` independently of pyramid routing.
  - Lowers warp grid construction to native base-grid + MatMul + GridSample.
- Added `search/model_families/lidar_cobevt/export_recipe.py` with actual ONNX
  export, checker, hashing, and operator inventory.
- Added `search/model_families/lidar_cobevt/operator_probe.py` with custom-op
  audit, scatter parser conversion, and a strongly typed command policy that
  omits weak precision controls and handles static graphs without shape flags.

### Real ONNX evidence

- Output directory:
  `/data/lxf/heal_data/outputs/4090_lidar_cobevt_capability_20260718_032028`
- Source ONNX SHA256:
  `659b0015eee1d16700f67216cab2c82da1c7c8e6789bf842773220d5ab40d034`
- Parser ONNX SHA256:
  `c976aec0c53cfb92dc7bb95d13fb6053357fef02080ee1160891f821992133eb`
- Inputs: six, including `record_len`
- Outputs: cls/reg/dir
- Nodes: 3,700
- Relevant native operators:
  - MatMul: 27
  - Einsum: 12
  - Softmax: 6
  - LayerNormalization: 13
  - GridSample: 1
- Registered custom ops: `trt::PointPillarScatterTRT` only
- Unregistered custom ops: 0
- ONNX checker: passed

### Real TensorRT evidence

- TensorRT: 10.9.0
- GPU UUID: `GPU-a4b5f0d9-c77c-2c99-b4c3-1b3811b835e3`
- Build mode: `--stronglyTyped --noTF32`, no weak precision flags
- ONNX parser: passed
- Scatter plugin creation: passed
- Detected network tensors: 6 inputs, 3 outputs
- Build time: 15.2783 seconds
- Engine generation time: 15.1185 seconds
- Engine size reported by TensorRT: 62.2923 MiB
- Engine SHA256:
  `53416351540aca2cbdfcbf5b62f4e26bf4f642c77500b049e767832a3e90c7cd`
- Engine deserialization: passed
- EngineInspector layers: 135
- Scatter input/output dtype: Float/Float
- Inspector tensor dtypes: Bool, Int32, Float
- New CoBEVT plugin required: false

### Failure diagnosis retained

1. PyTorch opset-17 export initially rejected `aten::affine_grid_generator`.
   The root cause was isolated and replaced with the existing mathematically
   equivalent native base-grid/MatMul representation.
2. Direct `trtexec` initially segfaulted during plugin initialization because
   TensorRT root libraries were absent from `LD_LIBRARY_PATH`. Reusing the
   formal worker environment removed the segfault before ONNX parsing.
3. The first valid parser run rejected dynamic shape flags because the graph is
   static. Static command generation now omits those flags.

No try/catch fallback, weak typing, graph relaxation, or new plugin was used.

### Tests

- RED: 8 missing-module failures for wrapper/probe interfaces.
- Additional RED: unsupported `affine_grid_generator` real ONNX export.
- Additional RED: static probe incorrectly emitted empty shape flags.
- GREEN result: `16 passed in 2.43s` across CoBEVT export/probe and existing
  strongly typed scatter tests.
- Modified Python files compile successfully.
- `git diff --check` passes.

### Status

- `COBEVT_TYPED_ONNX_EXPORT_PASS=true`
- `COBEVT_TRT_PARSER_PASS=true`
- `COBEVT_STRONGLY_TYPED_FP32_ENGINE_PASS=true`
- `COBEVT_ENGINE_DESERIALIZE_PASS=true`
- `COBEVT_NEW_PLUGIN_REQUIRED=false`
- `COBEVT_REAL_GPU_INFERENCE_PENDING=true`

--- ROUND 005 COMPLETE | 2026-07-18T03:25:50+08:00 ---

## Round 006: Canonical Precision And Strict FP16 Strongly Typed Build

### Implementation

- Extended `search/model_families/lidar_cobevt/export_recipe.py` to capture all
  weighted module calls, build a canonical ONNX origin map, rename canonical
  nodes, and persist the complete mapping hash.
- Added `search/model_families/lidar_cobevt/quantization_recipe.py`.
  - Maps every captured weighted Conv/Linear call to one CoBEVT-owned precision
    group.
  - Removes unverified INT8 actions from the search capability instead of
    permitting nominal INT8 with a floating fallback.
  - Keeps raw cls/reg/dir outputs protected and the parameter-free affine-grid
    MatMul fixed at FP16.
  - Keeps `PointPillarScatterTRT` outside precision genes and rejects adjacent
    Q/DQ.
  - Adds a CoBEVT-only auxiliary dtype closure for LayerNormalization
    parameters, attention elementwise bias, and Where branches. The shared
    pyramid typed-graph implementation was not modified.
- Added `search/model_families/lidar_cobevt/deployment_recipe.py`.
  - Production builder is fixed to `--stronglyTyped --noTF32`.
  - Weak `--fp16`, `--int8`, precision constraints, layer precision, and layer
    output-type controls are absent.
  - Cache identity binds model family, recipe version, fixed K, physical hash,
    precision hash, calibration signature, and build signature.
- Added `search/stage2/lidar_cobevt_real_evaluator.py` with a fail-closed
  smoke10-before-fixed50 protocol, GPU AP/IoU, and exactly 8 DataLoader workers.

### Real canonical and TensorRT evidence

- Artifact directory (not tracked by Git):
  `/data/lxf/heal_data/outputs/4090_lidar_cobevt_capability_20260718_032028`
- Canonical weighted entries: 53
- Parameter-free functional compute entries: 1
- Canonical renamed nodes: 54
- Canonical origin-map hash:
  `2ea7a5e6973adbb365b2cd6778f8e74564521ad12f7840097db3a157704cf1c6`
- Strict FP16 requested weighted entries: 53
- Scatter boundary: FP32 / Float-to-Float
- CoBEVT auxiliary closure:
  - LayerNormalization nodes audited: 13
  - LayerNormalization scale/bias Casts inserted: 14
  - attention same-type elementwise Casts inserted: 6
  - attention Where branch Casts inserted: 6
  - scatter-adjacent Q/DQ count: 0
- Typed ONNX SHA256:
  `0ed8c1b0aa7b50903b7ea5d260eedcba3b2c2ac3b05ced6021ef2aa3b2916830`
- Parser ONNX SHA256:
  `a76d1822124d6ba15d171a141658204840642d2d8a301dbe54a682a68f887f45`
- TensorRT: 10.9.0
- GPU UUID: `GPU-a4b5f0d9-c77c-2c99-b4c3-1b3811b835e3`
- Parser time: 0.138181 seconds
- Engine generation time: 34.4587 seconds
- Total build time: 34.7017 seconds
- Engine size: 39 MiB
- Engine SHA256:
  `c138ce42cf59f743453280e1c269fc5c7588882e6eb1d7edd46be323a656d1c9`
- Engine deserialization: passed
- Engine layers: 303
- Engine I/O tensors: 9
- Inspector tensor dtype occurrences:
  - Half: 424
  - Float: 239
  - Int32: 12
  - Bool: 8
- Scatter inspector boundary: Float inputs and Float output
- Plugin SHA256:
  `91aec743dca383151b995a60d004cd254ce115ed16152873c1f46754bb15022d`

### Root-cause evidence

The first strict FP16 parser attempt failed at LayerNormalization because the
activation was Half while scale/bias remained Float. After explicit parameter
Casts, parsing advanced to a Half/Float attention Add. After the elementwise
closure, parsing advanced to a Float/Half Where. Closing the Where branches
allowed the complete graph to parse and build. Each change was guarded by a
separate RED-to-GREEN regression test; no weakly typed fallback was introduced.

### Tests

- New focused tests: canonical precision groups, auxiliary typed/QDQ contract,
  strongly typed builder policy, cross-family cache isolation, and evaluator
  protocol.
- Focused plus existing strongly typed regression result:
  `33 passed, 5 tracer warnings in 2.01s`.
- The warnings are expected PyTorch static-export tracer warnings; no test,
  parser, engine, or dtype audit warning was converted into a pass.

### Status

- `COBEVT_CANONICAL_PRECISION_MAPPING_PASS=true`
- `COBEVT_STRONGLY_TYPED_FP16_ENGINE_PASS=true`
- `COBEVT_ENGINE_DESERIALIZE_PASS=true`
- `COBEVT_SCATTER_QDQ_COUNT=0`
- `COBEVT_CROSS_FAMILY_CACHE_ISOLATION_PASS=true`
- `COBEVT_MIXED_INT8_DEPLOYMENT_AVAILABLE=false`
- `COBEVT_REAL_GPU_INFERENCE_PENDING=true`
- `COBEVT_SMOKE10_FIXED50_PENDING=true`

--- ROUND 006 COMPLETE | 2026-07-18T03:49:57+08:00 ---

## Round 007: Real CoBEVT Legal-Width Greedy And GA Stage-1 Smoke

### Implementation

- Added `search/model_families/lidar_cobevt/stage1_capability.py`.
  - Represents the 256-channel fusion embedding as one legal-width domain of
    eight complete 32-channel attention heads.
  - Enumerates legal retained head counts `[2, 3, 4, 5, 6, 7, 8]`, equivalent
    to physical embedding widths `[64, 96, 128, 160, 192, 224, 256]`.
  - Builds one full parameter dependency closure per head across shrink output,
    all QKV and output projections, FFNs, LayerNorms, relative-position bias,
    fusion MLP, and cls/reg/dir input channels.
  - Collects empirical Fisher as `mean(g^2)` and mean gradient as `mean(g)`.
  - Produces immutable first- and second-order prune-only rankings independent
    of precision genes.
  - Exposes 53 independent FP32/FP16 precision groups. INT8 is removed because
    CoBEVT mixed INT8 deployment has not yet passed realization.
  - Uses exact CoBEVT physical closure parameter counting; the predicted count
    is regression-tested against a materialized 6-head model.
- Added `search/orchestration/lidar_cobevt_smoke.py`.
  - Probes reachable budgets before running search.
  - Reuses the existing `run_six_budget_greedy` and
    `run_legal_width_stage1_seeds` implementations.
  - Freezes one BOPS band, one greedy endpoint, GA population 16, three
    generations, one seed, and Top-2 intent for later Stage-2.
  - Does not invoke normal candidate repair.
- Added `search/configs/lidar_cobevt_4090_model_family_smoke.yaml` with a fresh
  output/cache policy and the accepted strongly typed Stage-2 protocol.
- CoBEVT greedy uses frontier size 1. This preserves the exact per-step
  `delta-L/delta-BOPS` action ordering while avoiding combinatorial beam
  expansion across 53 binary precision groups.

### Real Fisher and search capability

- Successful output directory:
  `/data/lxf/heal_data/outputs/4090_lidar_cobevt_stage1_smoke_20260717_131405`
- Checkpoint SHA256:
  `67b0f2f00d74fea4912b4fdb902c1150146c733201e5dc36d3358f5ba605cfd4`
- Config SHA256:
  `a0ee9d64fd1b01af95b1c997937ab07e0e810440236accfb598d5462ba086c33`
- Device: GPU7
- Fisher sample count: 1
- Fisher micro-batch size: 1
- Fisher manifest hash:
  `066569cb5c5f1dbb30947c97f1b048add27a55d5c3b34ea3dc12f79e78bc7dbf`
- Weighted runtime calls: 53
- Precision groups: 53
- Frozen Stage-1 joint-loss scale: `12.092770715175625`
- Stage-1 objective:
  `J1 = -0.8 * (L_joint/L_scale) + 0.2 * R_prune`
- SQNR contribution to the main objective: 0
- Width-space hash:
  `37b4fd2b89d6fbfba83e612dc28ade9782800bbb97aeed26e56a77ad2a6084f8`
- Fixed second-order decoder ranking hash:
  `55865b333d42326880d324355c7c64753b2ca42f9b507d33a3075fd71952a1d6`

### Reachable-budget evidence

- Probe candidates: 378
- BOPS range: `[0.2328965473, 0.4910058553]`
- Candidate counts within `target +/- 0.0075`:
  - 0.10: 0
  - 0.15: 0
  - 0.20: 0
  - 0.25: 156
  - 0.30: 2
- Selected smoke budget: 0.25
- Legal interval: `[0.2425, 0.2575]`

### Real greedy result

- Unique proxy evaluations: 990
- BOPS primary candidates: 1
- Endpoint BOPS: `0.2543881352`
- Endpoint candidate hash:
  `8ca730ecab81a5a327f55f8636519dfb8cd968963872d769070818e65daa605d`
- Endpoint structure: all eight fusion heads retained
- Endpoint `R_prune`: 0
- Endpoint requested precision: mixed FP32/FP16, with no INT8
- Endpoint requested and Stage-1 legalized precision profiles: identical
- Normal repair invocation count: 0

The first real attempt used a generic beam width of 8 and exhausted 4,096
states before reaching the 0.25 band; its nearest BOPS was 0.47468255. The
reachable probe had already proven 156 feasible candidates, so this was
classified as greedy branch expansion exhaustion, not budget infeasibility.
The CoBEVT-only frontier-size correction was verified by a RED-to-GREEN test
and the fresh run above.

### Real GA result

- Seeds: 1 (`4090`)
- Population: 16
- Generations: 3
- Total proxy evaluations: 48
- Per-generation candidate counts: 16, 16, 16
- Unique genotypes: 48
- Unique phenotypes: 48
- Unique structures observed: 6
- BOPS-feasible phenotype archive: 9
- Archive unique structures: 2
- Archive unique precision profiles: 8
- Overall BOPS range: `[0.2471485387, 0.7140382998]`
- Normal candidate count: 48
- Repair invocation count/rate: 0 / 0.0
- Repair changed structure/precision: 0 / 0
- Generation 0 BOPS-feasible: 9
- Generation 1 BOPS-feasible: 0
- Generation 2 BOPS-feasible: 0

The later-generation supply exhausted because fresh legal mutations in this
small FP32/FP16 space moved outside the narrow 0.25 band. These candidates are
retained as explicit BOPS failures and are not promoted to Stage-2. The nine
archive candidates remain available for deduplicated physical deployment.

### Tests

- Added tests for exact smoke scale/config, reachable-band selection, generic
  greedy/GA delegation, fresh-run enforcement, head parameter closures,
  precision/structure orthogonality, and physical parameter identity.
- Focused model-family/runner regression result: `16 passed`.
- Normal search candidate repair invocation rate remains exactly zero.

### Status

- `COBEVT_LEGAL_WIDTH_STAGE1_SPACE_PASS=true`
- `COBEVT_REAL_FISHER_COLLECTION_PASS=true`
- `COBEVT_GREEDY_STAGE1_PASS=true`
- `COBEVT_GA_16X3X1_STAGE1_PASS=true`
- `COBEVT_STAGE1_FEASIBLE_PHENOTYPES=9`
- `COBEVT_MIXED_INT8_DEPLOYMENT_AVAILABLE=false`
- `COBEVT_STAGE2_STARTED=false`
- `COBEVT_REAL_GPU_INFERENCE_PENDING=true`

--- ROUND 007 COMPLETE | 2026-07-18T04:15:54+08:00 ---

## Round 008: Separate GPU Evaluation, Mixed-Concat Closure, And Real Stage-2

### Code changes

- Added `search/integration/lidar_cobevt_evaluation_provider.py`.
  - Pins a selected physical GPU through `CUDA_VISIBLE_DEVICES` while the
    subprocess uses logical `cuda:0`.
  - Requires GPU AP/IoU, strict mode, and exactly 8 DataLoader workers.
  - Dispatches only to the separate CoBEVT evaluation worker.
- Added `search/integration/lidar_cobevt_evaluation_worker.py`.
  - Uses `prepare_cobevt_maxk_inputs`, never the pyramid input preparer.
  - Loads one target TensorRT engine without constructing a per-GPU FP32
    reference model.
  - Executes GPU post-processing/AP, warmup reset, exact manifest admission,
    nonfinite/empty output checks, and p50/p90/p95 latency collection.
- Updated `search/stage2/lidar_cobevt_real_evaluator.py` so smoke10 and fixed50
  receive distinct count-matched manifests.
- Updated `search/model_families/lidar_cobevt/quantization_recipe.py` with a
  CoBEVT-only mixed floating Concat contract. If any branch is FP16, the merge
  inputs are explicitly closed to FP16; integer shape Concat nodes are not
  changed.
- Added/updated four test modules for provider/worker protocol, phase-specific
  manifests, and the mixed backbone Concat contract.
- Implementation commit: `55a1886d200f5cb5352d1c9bc5264879abb8851f`.

### Manifest and fixed-K evidence

- Artifact root:
  `/data/lxf/heal_data/outputs/4090_lidar_cobevt_model_family_smoke_20260718_042721`
- train200 plus validation70 scanned records: 270.
- Source maximum voxel count: 25412.
- CoBEVT fixed K: 25600 (alignment 256), overflow count 0.
- fixed50 manifest hash:
  `14e70f4974c7aec17fa9579b58a4282ebaeb188e59f4dde44a018482eaa63127`.
- GPU AP/IoU backend and 8 DataLoader workers were used for every evaluation.

### Failure reproduction and fix

The first real greedy mixed build failed closed at
`/backbone_m1/Concat`: deblock0 produced Float while deblock1/2 produced Half.
Strict FP16 had already built, isolating the problem to a missing mixed merge
contract rather than fixed K, plugin, or TensorRT environment. A one-node RED
test reproduced the mismatch. Adding one explicit Float-to-Half Cast allowed
the same graph to parse/build with `--stronglyTyped --noTF32`; weak precision
flags and precision fallback remained forbidden.

### Real engine and fixed50 evidence

All five engines passed deserialize, canonical requested/realized precision,
Float-to-Float scatter, smoke10 `10/10, 0 skip`, and fixed50 `50/50, 0 skip`.

| candidate | R_BOPS | R_param | weighted FP32/FP16 | mAP | AP07 | screening p50 ms |
|---|---:|---:|---:|---:|---:|---:|
| strict FP32 | 1.000000 | 1.000000 | 53/0 | 0.591499 | 0.410487 | 8.324 |
| strict FP16 | 0.250000 | 1.000000 | 0/53 | 0.192366 | 0.113453 | 4.622 |
| greedy endpoint | 0.254388 | 1.000000 | 30/23 | 0.590689 | 0.410128 | 6.630 |
| GA generation-0 rank1 | 0.250001 | 1.000000 | 1/52 | 0.193831 | 0.115006 | 4.592 |
| GA generation-0 pruned | 0.247149 | 0.938587 | 0/53 | 0.175288 | 0.114382 | 4.443 |

The pruned GA candidate deterministically removes fixed-ranking head 7,
changes fusion width 256 to 224, and matches predicted/physical parameters at
9,855,410. The greedy endpoint is the accuracy-preserving result: relative to
same-GPU strict FP32 it changes mAP by -0.000810 and screening p50 by
8.324 to 6.630 ms (1.255x).

### Search supply and limitations

- GA generation 0 supplied 9 feasible phenotypes.
- Generations 1 and 2 supplied zero candidates in `[0.2425, 0.2575]`; no
  out-of-band candidate was promoted or duplicated.
- Mixed INT8 remains deployment-unavailable and absent from genes.
- Latency is parallel screening latency, not isolated formal latency.
- No full validation was run; fixed50 results are not final model claims.

### Verification

- Complete focused regression: `89 passed, 26 warnings`.
- Modified Python files: `py_compile` passed.
- `git diff --check` passed.
- H800 ancestor retained; pyramid worktree/controller remained untouched.
- Large ONNX/PLAN/log outputs remain under `/data` and outside Git.

### Status

- `COBEVT_STRONGLY_TYPED_MODEL_FAMILY_PASS=true`
- `COBEVT_MIXED_FP32_FP16_REALIZATION_PASS=true`
- `COBEVT_GREEDY_REAL_STAGE2_PASS=true`
- `COBEVT_GA_REAL_STAGE2_PASS=true`
- `COBEVT_REAL_GPU_INFERENCE_PASS=true`
- `COBEVT_SMOKE10_FIXED50_PASS=true`
- `COBEVT_FULL_VALIDATION_EXECUTED=false`
- `COBEVT_MIXED_INT8_DEPLOYMENT_AVAILABLE=false`
- `PYRAMID_PRODUCTION_PATH_MODIFIED=false`

--- ROUND 008 COMPLETE | 2026-07-18T05:01:09+08:00 ---
