# 4090 CNN / Soft-Fusion Search Generalization

## Current status

- Branch: `feature/heal-compress-4090-cnn-softfusion-family`
- Base commit: `d9e68ed54acb9dd878ea7949114dc3899348abf8`
- H800 ancestor retained: `b862b3d8ad061bd12580776226c75f564918298d`
- Pyramid GA remains in the original worktree and is not modified by this branch.
- First model family: HEAL DAIR-V2X LiDAR DiscoNet.
- Deferred model family: F-Cooper, pending checkpoint upload.

## Approved experiment contract

Both Greedy and GA use the six BOPS retention budgets:

```text
0.05, 0.10, 0.15, 0.20, 0.25, 0.30
```

The DiscoNet GA contract is:

```text
independent_seeds = 1
population_size = 64
offspring_size = 64
generations = 15
per_generation_stage2 = true
topk_stage2 = 5
```

Stage-2 remains a real deployment path:

```text
legal-width genotype
-> deterministic fixed Taylor mask
-> physical pruning replay
-> structure audit
-> fixedK29696 typed ONNX
-> explicit Q/DQ and explicit Cast
-> strongly typed TensorRT 10.9 engine
-> requested/realized precision audit
-> smoke10
-> fixed500 generation selection
-> full-validation generation-winner selection
-> isolated formal latency
```

No candidate outside the active BOPS gate can consume a Stage-2 engine slot.
Each generation builds at most five engines. Fewer than five legal candidates
continue without duplication; zero legal candidates skip the generation with a
recorded reason.

## Artifact retention design

The current Pyramid run writes multiple large representations of the same
logical structure:

```text
physical_plan.json
physical_pruning_plan.json
legalized_plan.json
pruning_request.json
sampling_pruning_request.json
```

The observed five families occupy approximately 69.4 GiB in the active run.
The new policy is content-addressed and completion-gated:

1. Workers may use full temporary artifacts while a candidate is active.
2. A completed candidate stores one deterministic compressed JSON object per
   unique content hash under the run-level audit store.
3. The candidate directory retains a small `candidate_audit_manifest.json`
   containing hashes, logical roles, sizes, and relative store references.
4. A compact `structure_plan_summary.json` retains the fields needed for routine
   review: selected units, domain widths, module shapes, parameter counts,
   physical hash, validation status, and failure reason.
5. Full plans remain recoverable for replay from the content-addressed store.
6. Engine, evaluation, precision realization, BOPS, QDQ/merge audits, deployment
   identity, and failure evidence remain candidate-addressable.
7. Compaction refuses active/incomplete generations and defaults to dry-run.
8. The active Pyramid generation is never compacted by this branch.

This reduces file count and semantic duplication without discarding replay
evidence. Byte-identical aliases from legacy runs can be migrated after their
generation completion markers have been verified.

## Model-family architecture

The existing Pyramid code embeds Pyramid assumptions in the context builder,
export wrapper, evaluator, and CLI. Adding model-name conditionals directly to
those files would make deployment validation fragile. The selected design adds
an explicit model-family boundary:

```text
HEALLidarFamilySpec
|- model/config/checkpoint identity
|- weighted and protected layer policy
|- trace and legal-width inventory policy
|- fixedK export module factory
|- canonical precision capability policy
|- merge and functional-op contracts
|- Stage-2 evaluator factory
`- evaluation input/output contract
```

The shared Greedy, legal-width GA, archive, BOPS gate, process pool, artifact
identity, and reporting code remains common. Pyramid keeps its existing adapter;
DiscoNet gets a separate soft-fusion recipe. F-Cooper will later register a Max
fusion recipe against the same interface.

## DiscoNet structure and deployment contract

The checkpoint and configuration resolve to:

```text
PointPillar encoder
-> BaseBEVBackbone
-> per-modality DownsampleConv shrinker
-> DiscoFusion
   -> affine warp
   -> concat(neighbor, ego)
   -> PixelWeightLayer: 512 -> 128 -> 32 -> 8 -> 1
   -> agent-axis Softmax
   -> weighted sum
-> cls/reg/dir heads
```

The PointPillar scatter remains a non-quantized FP32 plugin boundary. Fusion
weighted layers participate in the canonical precision inventory only after
typed ONNX mapping and strongly typed engine realization pass.

The external HEAL checkout currently lacks
`opencood.models.fuse_modules.disco_fuse`, while the accepted checkpoint contains
all `fusion_net.pixel_weight_layer.*` tensors. The adapter will install an
explicit, local compatibility module only for the DiscoNet family. Its topology
and state-dict keys match the historical `PixelWeightLayer`; checkpoint loading
fails closed on missing or unexpected weighted keys. The external HEAL checkout
is not edited.

## Readiness gates before full search

DiscoNet full search starts only after all of the following pass:

1. Checkpoint identity and state-dict audit.
2. PyTorch synthetic and real-frame forward.
3. Legal-width inventory and deterministic physical replay.
4. FixedK29696 wrapper parity against the original model.
5. ONNX checker and output shape parity.
6. PointPillarScatterTRT plugin load and FP32 boundary audit.
7. Strongly typed strict-FP32 engine build, deserialize, and smoke10.
8. Strongly typed FP16 and explicit-QDQ capability audit.
9. Requested/legalized/realized precision identity.
10. Merge, functional Softmax, and fusion weighted-sum dtype audit.
11. Real BOPS accounting from physical shapes and realized precision.
12. Fixed manifest evaluation with zero skipped frames.

Unsupported INT8 actions are removed from the DiscoNet precision space rather
than silently realized as FP16/FP32.

## Experiment outputs

Large outputs use a new timestamped directory outside Git. Git receives only:

- source code;
- tests;
- lightweight configs;
- compact JSON/CSV summaries;
- build/precision/structure evidence with bounded size;
- this handoff report.

ONNX, PTH, TensorRT plans, calibration engines/caches, and tensor dumps remain
ignored.

## Round 1 findings

- Existing artifact compaction tests: `18 passed`.
- DiscoNet checkpoint:
  `/home/lixingfeng/UniAD_examine/Auto_Search/original_models/dairv2s/LiDAROnly/lidar_disco/net_epoch_bestval_at35.pth`
- Checkpoint SHA256:
  `cc69e872fab245d687260ec4c277fab531b66a0b1e4d89fedfd4605b8f13e96c`
- Checkpoint contains 171 state entries, including the complete four-layer
  PixelWeightLayer.
- Current unmodified model-load result:
  `ModuleNotFoundError: opencood.models.fuse_modules.disco_fuse`.
- F-Cooper checkpoint: pending upload.

--- ROUND 1 | 2026-07-19 18:10:52 +0800 ---

## Round 2: completed candidate artifact compaction

Implemented:

- `search/artifacts/candidate_audit_store.py`
  - deterministic canonical JSON;
  - gzip with fixed timestamp;
  - SHA256 content addressing;
  - atomic writes;
  - reference resolution with path, size, and hash verification;
  - completion-marker ownership checks;
  - fail-closed alias mismatch handling;
  - compact structure-plan summary generation.
- `search/orchestration/stage2_artifact_compaction.py`
  - retained legacy hardlink mode;
  - added explicit content-addressed mode;
  - added dry-run mode;
  - only discovers candidates under completed-generation markers.
- `search/orchestration/legal_width_six_budget_ga.py`
  - added opt-in automatic finalization after Stage-2 retention completes;
  - final engine plans remain preserved;
  - existing Pyramid configuration behavior remains unchanged.
- `search/stage2/candidate_artifacts.py`
  - audit manifests and structure summaries now participate in candidate hashes.

Verification:

```text
17 focused tests passed
real candidate sample before = 13,787,508 bytes
real candidate sample after  =    162,796 bytes
sample reduction             =      98.82%
removed redundant files      =          5
stored canonical blobs       =          2
```

The real sample was copied to `/tmp` before migration. No file in the active
Pyramid output was changed.

--- ROUND 2 | 2026-07-19 18:22:20 +0800 ---

## Round 3: model-family registry and DiscoNet checkpoint loading

Implemented:

- `search/integration/lidar_family.py`
  - immutable HEAL LiDAR family schema;
  - fixedK, plugin boundary, export recipe, fusion kind, output contract.
- `search/integration/lidar_family_registry.py`
  - canonical registrations for Pyramid, DiscoNet, and deferred F-Cooper;
  - strict alias resolution and unknown-family rejection.
- `search/integration/disconet_compat.py`
  - local checkpoint-compatible PixelWeightLayer;
  - explicit installation only when the HEAL native module is absent;
  - native implementation is never overwritten;
  - compatibility source SHA recorded.
- `search/integration/model_provider.py`
  - generalized `HEALLidarModelBundle`;
  - preserved `LidarPyramidModelBundle` and `load_lidar_pyramid_model` APIs;
  - checkpoint state count, missing/unexpected keys, and shape audit;
  - weighted mismatch fails closed before search readiness.

Real checkpoint evidence:

```text
DiscoNet source state entries       = 171
DiscoNet missing parameter keys     = 0
DiscoNet unexpected weighted keys   = 0
DiscoNet strict weighted pass       = true
Pyramid missing parameter keys      = 0
Pyramid unexpected weighted keys    = 0
Pyramid strict weighted pass        = true
```

Tests:

```text
8 family/provider tests passed
real DiscoNet checkpoint loaded on CPU
real Pyramid compatibility checkpoint loaded on CPU
```

--- ROUND 3 | 2026-07-19 18:28:02 +0800 ---

## Round 4: fixedK soft-fusion export wrapper

Implemented:

- `search/integration/softfusion_trt_export.py`
  - static BaseBEVBackbone blocks/deblocks execution;
  - per-modality shrinker before fusion;
  - fixedK PointPillarScatterTRT frontend reuse;
  - exportable affine warp;
  - MaxFusion agent-axis maximum;
  - DiscoFusion PixelWeightLayer, agent-axis Softmax, and weighted sum;
  - fixed cls/reg/dir output contract;
  - explicit family/model mismatch rejection.
- `search/integration/trt_compatible_export.py`
  - family export factory;
  - original Pyramid factory remains unchanged;
  - unsupported recipes fail closed.

Verification:

```text
13 family/provider/export tests passed
real DiscoNet wrapper class = SearchTensorRTCompatibleSoftFusion
real execution contract = pillar_vfe -> scatter_plugin ->
  base_bev_backbone -> modality_shrinker -> disconet_fusion -> heads
fixedK = 29696
outputs = cls_preds, reg_preds, dir_preds
```

GPU 7 was explicitly approved for a bounded real-checkpoint parity run. The
process peak allocation was 654,405,632 bytes. Results were finite:

```text
output      shape             max_abs       mean_abs
cls_preds   [1,2,128,256]     7.1907e-4     7.7321e-7
reg_preds   [1,14,128,256]    4.8721e-4     3.1177e-7
dir_preds   [1,4,128,256]     1.9560e-3     9.7775e-7
```

These errors do not meet a strict `rtol=1e-5, atol=1e-6` elementwise allclose
test, so the result is recorded as bounded finite parity rather than exact
parity. ONNX/TRT tensor parity and real AP remain readiness-gate requirements;
no TensorRT capability claim is made here.

--- ROUND 4 | 2026-07-19 18:38:20 +0800 ---

## Round 5: family-aware context and Stage-2 hooks

Implemented:

- `search/integration/lidar_family_context.py`
  - family-neutral facade over the production context builder.
- `search/integration/lidar_pyramid_context.py`
  - records model family and family specification;
  - loads generic HEAL LiDAR bundles;
  - family-specific protected precision modules;
  - family-specific functional FP16 output boundary;
  - family identity included in ONNX/cache signature;
  - existing Pyramid defaults preserved.
- `search/stage2/lidar_family_real_evaluator.py`
  - family-neutral production evaluator name backed by the validated Stage-2
    core.
- `search/stage2/lidar_pyramid_real_evaluator.py`
  - selects export wrapper through the family factory;
  - retains `/Concat_9` as a Pyramid-only named merge contract;
  - passes family and worktree identity into real evaluation subprocesses.
- `search/stage2/candidate_worker.py`
  - propagates `model.family` and constructs family context/evaluator.
- `search/integration/evaluation_provider.py`
  - records `model_family` and `repository_root` in every evaluation request.
- `search/integration/evaluation_worker.py`
  - imports code from the producing worktree;
  - installs DiscoNet compatibility before model construction;
  - avoids silently importing the active Pyramid worktree implementation.

Real context evidence on GPU 7:

```text
trace atomic units         = 4733
trace coupled units        = 4733
initial safe units         = 64
initial pruning actions    = 64
weighted precision layers = 32
precision groups           = 32
protected precision groups = 1
trace hash = 308184a6ab6605c51fad1ec1049ffdd81e233f313d6ed9127df1d7111fa5fc4d
```

The final PixelWeightLayer output owns an explicit FP16 functional boundary
before `Relu -> Softmax -> Mul -> ReduceSum`; its compute precision remains a
separate gene only if later strongly typed realization proves the action legal.

Tests:

```text
9 new family context/Stage-2 tests passed
51 combined worker/process/QDQ/typed/merge/BOPS tests passed
```

--- ROUND 5 | 2026-07-19 18:47:40 +0800 ---

## Round 6: single-seed DiscoNet six-budget orchestration

Implemented:

- `search/orchestration/legal_width_six_budget_ga.py`
  - accepts one or more independent seeds instead of hard-coding three;
  - keeps the existing Pyramid three-seed configuration valid;
  - enforces population and offspring sizes of at least 64;
  - enforces a Stage-2 build cap in `[1, 5]`;
  - retains one Stage-2 decision per budget and generation.
- `search/orchestration/lidar_pyramid_search.py`
  - routes the formal runner through family-aware context and evaluator hooks;
  - recognizes legacy three-seed, new single-seed, and generic six-budget names;
  - leaves existing Pyramid defaults unchanged.
- `search/configs/lidar_disco_4090_joint_six_budget_ga.yaml`
  - six targets: `0.05, 0.10, 0.15, 0.20, 0.25, 0.30`;
  - one seed, 64 initial/population/offspring, 15 generations;
  - at most five Stage-2 engines per generation;
  - real 500-frame generation evaluation and 1789-frame final validation;
  - completed-candidate content-addressed audit storage;
  - GPU AP IoU with eight DataLoader workers.
- `search/configs/lidar_disco_4090_greedy_six_budget.yaml`
  - identical six budgets;
  - final endpoint deployment only;
  - real strongly typed deployment and full validation protocol.

GPU policy:

```text
Stage-1 GPU = 7
Stage-2 GPUs = [7]
per-process memory fraction <= 0.50
foreign processes are not killed or paused
shared-GPU timing is screening only
formal latency requires a later isolated serial replay
```

Verification:

```text
single-seed/config/Pyramid compatibility tests = 11 passed
Disco GA CLI dry-run                          = passed
Disco Greedy CLI dry-run                      = passed
py_compile                                    = passed
git diff --check                              = passed
```

No search, engine build, or AP result is claimed by this round; strongly typed
DiscoNet readiness remains the fail-closed prerequisite for both searches.

--- ROUND 6 | 2026-07-19 18:55:24 +0800 ---

## Round 7: decouple baseline deployment from Stage-1 scale

The first real `--baseline-only` attempt stopped before ONNX export with:

```text
RuntimeError: joint_taylor_joint_loss_scale_path_required
```

Root cause: the runner loaded the Stage-1 joint-loss scale before dispatching
the baseline-only branch, although baseline export, TensorRT build, and AP
evaluation do not consume that scale.

Minimal fix:

- `search/orchestration/lidar_pyramid_search.py`
  - baseline-only context, GPU gate, and evaluator run first;
  - Stage-1 scale loading remains mandatory immediately after that branch;
  - normal Greedy/GA behavior is unchanged.
- `tests/test_lidar_family_cli.py`
  - reproduces a joint-Taylor config with no scale;
  - proves baseline-only execution succeeds;
  - search paths retain their existing scale validation tests.

Verification:

```text
focused baseline/Stage-1/config tests = 21 passed
py_compile                          = passed
git diff --check                    = passed
ONNX or engine created by failed run = no
```

--- ROUND 7 | 2026-07-19 19:01:13 +0800 ---

## Round 8: make the real DiscoNet PixelWeight graph exportable

All three baseline precisions reproduced one precision-independent export
failure before TensorRT:

```text
TypeError: len() of a 0-d tensor (Occurred when translating size)
```

Root-cause evidence:

- the synthetic DiscoFusion graph exported successfully;
- the real checkpoint JIT graph exposed three negative-dimension
  `aten::size` nodes under `fusion_net.pixel_weight_layer`;
- they came from the historical, redundant
  `view(-1, x.size(-3), x.size(-2), x.size(-1))` on an already 4-D tensor;
- PyTorch 2.0 ONNX translation could not lower that negative-dimension size
  after the custom scatter boundary.

Minimal export-adapter fix:

- `search/integration/softfusion_trt_export.py`
  - executes the existing PixelWeightLayer Conv/BN/ReLU submodules directly;
  - does not replace, resize, or reload any checkpoint parameter;
  - preserves the exact weighted operation order and 4-D tensor semantics;
  - removes only the redundant reshape from the export graph.
- `tests/test_softfusion_trt_export.py`
  - RED test captured the three PixelWeight `aten::size` nodes;
  - GREEN test proves the export graph contains none.

Verification:

```text
focused export/family/Stage-2 tests = 15 passed
real checkpoint ONNX checker       = passed
real weighted origin entries       = 32
real ONNX bytes                     = 45,191,620
py_compile                          = passed
git diff --check                    = passed
```

The earlier failed baseline directory is retained as failure evidence and is
not reused for the next fresh build.

--- ROUND 8 | 2026-07-19 19:08:36 +0800 ---

## Round 9: audit strongly typed fused DiscoNet concat boundaries

Fresh strict FP32 and strict FP16 engines built successfully, but the existing
merge audit rejected `/Concat_7` as `not_yet_verified`.

EngineInspector evidence showed that TensorRT fused:

```text
GridSample -> Slice -> /Concat_7 -> explicit downstream Cast
```

into one kgen layer. The layer Metadata names both `/Concat_7` and the
downstream Cast. The typed graph independently proves that every Concat input
has an explicit FP16 Cast. In strict FP32 the fused layer exposes the tensor
after its downstream FP32 Cast; in strict FP16 it exposes Half. Therefore the
old audit incorrectly treated the final fused-layer output dtype as the Concat
semantic dtype.

Implemented in `search/stage2/lidar_pyramid_real_evaluator.py`:

- match compiler-fused merges through exact EngineInspector Metadata tokens;
- accept an FP16 Concat only when all graph branches explicitly cast to FP16;
- additionally require a named downstream Cast after the merge in Metadata;
- retain fail-closed behavior when either proof is absent.

Real saved-engine replay:

```text
strict_fp32 /Concat   = FP16, compiler_backend_merge_tensor_match
strict_fp32 /Concat_7 = FP16, graph_constrained_fp16_concat_fused_with_downstream_cast
strict_fp16 /Concat   = FP16, compiler_backend_merge_tensor_match
strict_fp16 /Concat_7 = FP16, graph_constrained_fp16_concat_fused_with_downstream_cast
issues = []
```

Verification:

```text
focused merge/family Stage-2 tests = 7 passed
py_compile                         = passed
git diff --check                   = passed
```

--- ROUND 9 | 2026-07-19 19:15:30 +0800 ---

## Round 10: DiscoNet strongly typed readiness accepted

Fresh all-keep strict FP32, strict FP16, and maximal legal INT8 engines completed
the same real 500-frame protocol on GPU 7. Full results and hashes are recorded
in `docs/codex_handoffs/4090-DISCONET-STRONGLY-TYPED-READINESS.md`.

Key result:

```text
strict FP32 mAP / BOPS = 0.603354 / 1.000000
strict FP16 mAP / BOPS = 0.603948 / 0.250000
maximal INT8 mAP / BOPS = 0.512429 / 0.062500
all evaluated = 500
all skipped = 0
requested/realized mismatches = 0
DISCONET_READY_FOR_SEARCH = true
formal isolated latency = pending
```

The readiness output occupies 1.2 GiB under `/var/tmp`; no ONNX, engine,
calibration cache, or checkpoint is added to Git.

--- ROUND 10 | 2026-07-19 19:35:51 +0800 ---

## Round 11: compact Stage-1 proxy cache records

The first DiscoNet Greedy Stage-1 run was stopped after a storage audit found:

```text
proxy rows written = 7,756
incomplete run size = 2.1 GiB
representative row = 216,060 bytes
phenotype field = 199,999 bytes
repeated group_mask entries per row = 4,265
```

The cache embedded the full legal-width phenotype in every JSONL row even
though the evaluator always has the current phenotype in memory. This repeated
the deterministic group mask, domain maps, and group contracts thousands of
times.

Implemented:

- `search/stage1/proxy_evaluator.py`
  - disk cache stores scalar metrics, deployment candidate identity, and a
    canonical phenotype archive hash;
  - full phenotype is restored from the current candidate on cache hits;
  - in-memory evaluator and generation Top-K records remain complete.
- `search/stage2/repaired_topk_manifest.py`
  - reads new compact hash rows;
  - retains backward compatibility with old embedded-phenotype archives;
  - still scans the archive once for multiple Top-K candidates.

Measured on a real DiscoNet row:

```text
old row bytes = 216,060
new row bytes =   1,128
reduction     = 99.48%
```

Verification:

```text
proxy/cache/Top-K/joint-proxy tests = 19 passed
py_compile                          = passed
git diff --check                    = passed
```

The incomplete 2.1-GiB run contains no endpoint or deployment result and is
removed after this audit is committed. The Greedy search restarts fresh.

--- ROUND 11 | 2026-07-19 19:52:53 +0800 ---

## Round 12: size the DiscoNet Greedy state bound from its legal space

The first compact-cache run reached the configured 16,384-state cap for the
0.05 target. Its nearest state remained at BOPS retention 0.244341 with zero
INT8 MAC share, so the resulting `infeasible` status was a search-cap failure,
not a model reachability result.

DiscoNet inventory audit:

```text
legal-width domains = 25
adjacent width-decrease actions on a complete path = 842
maximum precision-decrease actions = 64
maximum monotonic path actions = 906
maximum simultaneous successors = 57
conservative state bound = 51,643
```

Implemented:

- Disco Greedy `max_expansions` raised from 16,384 to 65,536;
- BOPS targets and tolerances remain unchanged;
- action ordering, Taylor score, and legal widths remain unchanged;
- the incomplete run is resumed so its Fisher statistics and compact proxy
  cache are reused;
- the legacy `build_lidar_pyramid_context` runner symbol is retained for
  Pyramid API/test compatibility while execution continues through family
  hooks.

Verification:

```text
Greedy/config/family/Pyramid compatibility tests = 15 passed
py_compile                                    = passed
git diff --check                              = passed
```

--- ROUND 12 | 2026-07-19 20:19:18 +0800 ---

## Round 13: complete DiscoNet six-budget Greedy Stage-1 and compact traces

The resumed legal-width Greedy search completed all six requested BOPS
budgets with the unchanged primary tolerance of +/-0.005. Every endpoint was
admitted by the primary band; the expanded +/-0.0075 band was not needed for
the selected endpoint.

```text
target   endpoint R_BOPS   admission
0.05     0.054996405       primary
0.10     0.102304861       primary
0.15     0.152289152       primary
0.20     0.200173050       primary
0.25     0.252669007       primary
0.30     0.304440916       primary

unique proxy evaluations = 27,590
infeasible budgets        = 0
normal candidate repair   = 0
proxy device              = cuda:7 (physical GPU 7)
```

The lowest endpoint is a proxy result only. It still requires physical
materialization, ONNX/QDQ export, strongly-typed TensorRT construction,
deserialization, smoke evaluation, and full validation before it can be used
as a deployment result.

The Greedy trace writer now defaults to `greedy-compact-v1`: accepted path,
terminal state, BOPS funnel, rejection counts, evaluated-state count, and a
deterministic state-hash digest are retained; repeated full evaluated
phenotypes are omitted. Existing completed traces were compacted from roughly
1.0 GiB to roughly 15 MiB after verifying counts and digests. `trace_detail:
full` remains available for targeted debugging.

--- ROUND 13 | 2026-07-19 21:37:19 +0800 ---

## Round 14: complete DiscoNet Greedy deployment and full validation

The six-budget Greedy endpoint set completed the real DiscoNet deployment
path. The endpoint Stage-2 resume used a unique process-pool namespace
(`attempt_20260719_074233_248406`) after an earlier interrupted controller had
left stale task identifiers. The resume accepted only identity-complete,
frame-complete artifacts:

```text
unique endpoints              = 6
unique deployments            = 6
build tasks                   = 0
validated build artifacts reused = 6
full-validation tasks         = 0
validated full results reused = 6
successful endpoints          = 6
full validation               = 1789/1789, skipped=0 for every endpoint
```

The final full-validation results are recorded in
`docs/codex_handoffs/4090-DISCONET-GREEDY-RESULTS-20260719.md` and the
timestamped run's `disco_greedy_six_budget_full_results.csv/json`. The strict
FP32 reference was mAP `0.6357519111`, forward p50 `7.758738 ms`, and total
pipeline p50 `13.469938 ms`. The six endpoint mAP values, in budget order,
were `0.634540`, `0.636059`, `0.636044`, `0.635947`, `0.635771`, and
`0.635952`; all structure, typed-QDQ, merge, precision-identity, and realized
BOPS audits passed. The lowest endpoint retained `26.97%` physical
parameters and `9.96%` weight storage, with forward speedup `2.412x` and
total-pipeline speedup `1.700x` on the shared GPU7 screening measurement.

No GA generation has started yet. The next controlled action is a fresh
single-seed DiscoNet GA run with population/offspring `64`, `15` generations,
six budgets, and at most five real Stage-2 engine candidates per generation.
F-Cooper remains deferred until its checkpoint is available.

--- ROUND 14 | 2026-07-19 22:58:00 +0800 ---
