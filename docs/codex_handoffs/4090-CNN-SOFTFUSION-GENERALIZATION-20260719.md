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
