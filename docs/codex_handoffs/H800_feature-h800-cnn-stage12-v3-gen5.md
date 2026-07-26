# H800 CNN Strict Stage-1/Stage-2/V1–V3 GA Gen5

This handoff tracks the isolated re-search of Pyramid, DiscoNet and F-Cooper
with the current strict GA framework. Historical two-stage GA artifacts are
read-only and are never used as current-framework search results.

---

## 2026-07-26 02:05:30 -07:00

- Root-caused the new-run budget failures to the specialized CNN Greedy
  adapter, not to an exact-BOPS arithmetic defect. It captured only selected
  trajectory states. DiscoNet jumped from `0.3230040913` to `0.2234232808`,
  and F-Cooper from `0.2774228250` to `0.1904317909`, crossing valid bands in
  one legal FP32→FP16 action.
- Restored two audited capabilities from the generic Greedy engine without
  restoring repair: (1) capture of already-evaluated legal neighbor frontier
  states and (2) bounded target-directed beam recovery with adjacent legal
  actions only. No candidate is projected or rewritten to meet a budget.
- Added explicit iteration accounting. A selected primary action or one beam
  depth expansion counts as one search iteration; parallel neighbor candidate
  evaluations are reported separately. Thus recovery work is included in
  `total_search_iteration_count` rather than hidden.
- Added a formal pre-GA gate: all six exact Greedy anchors must first exist in
  their `[target-0.005,target+0.005]` bands and pass real engine/fixed50
  deployment validation. Formal GA cannot begin for any budget until this
  six-budget gate passes. `--greedy-only` permits an explicit stop after the
  gate; a resumed GA run reuses immutable Stage-2 anchor caches.
- Added deterministic regression cases where the selected path jumps over
  both targets: one target is recovered from the evaluated-neighbor frontier,
  the second at beam depth 1. The audit reports primary steps `1`, recovery
  iterations `1`, total search iterations `2`, and all repair counts `0`.
- Tests: targeted `11 passed`; full suite `1040 passed`; `compileall`,
  `py_compile`, and `git diff --check` passed. GPU2/GPU3 remain occupied by
  the independent V2X-ViT/AttFusion formal runs, so the new three-model
  Greedy gate and GA reruns have not yet been launched at this checkpoint.

### Runtime bug found during the first fresh launch

- The first isolated Pyramid launch failed before model initialization and
  before any candidate build. With `CUDA_VISIBLE_DEVICES=3`, CUDA exposes the
  selected physical GPU as logical device 0, but the CNN context still called
  `torch.cuda.set_device(3)`, causing `invalid device ordinal`.
- Added an explicit physical-to-logical CUDA ordinal resolver. Provenance and
  scheduling retain physical GPU index/UUID, while PyTorch/ModelOpt/TensorRT
  contexts use the process-local ordinal. Regression tests cover single- and
  two-device visibility plus fail-closed handling of a hidden physical GPU.
- The failed output
  `h800_cnn_frontier_beam_greedy_then_ga_gen5_20260726_020852/pyramid`
  contains only startup provenance and a log; it will not be reused.
- The next fresh run proved DiscoNet all six budgets analytically reachable;
  five anchors came from already-evaluated legal neighbors and 0.05 from the
  selected path. It used 726 primary iterations, 38,827 neighbor evaluations,
  zero beam iterations, zero repair, and zero projection.
- DiscoNet then built and evaluated its first exact anchor successfully, but
  the reporting layer failed on a missing explicit import of
  `stage2_payload`. No GA began. Added that import plus a fail-closed resume
  loader which accepts an existing exact Greedy report only after current
  schema, complete phenotype hash, and exact current BOPS hard-gate checks.
  The immutable Stage-2 cache is reused by hash; analytic neighbors and the
  already-built anchor are not needlessly recomputed.

---

## 2026-07-25 15:52:00 -07:00

- First concurrent launch exposed a fail-closed reporting alias defect after
  full Greedy trajectory generation: `SizeProxy` reports
  `R_size_vs_fp32`, while the strict winner selector consumes
  `mixed_weight_retention`. DiscoNet and F-Cooper exited before GA generation
  0; no candidate engine was built and no historical output was overwritten.
- Added the explicit `canonical_size_metrics` adapter and regression coverage.
  The value is unchanged; only the production field name is mapped into the
  strict Stage-1 contract.
- V2X-ViT GPU3 was not stopped: its generation-8 worker exited normally and a
  generation-9 Stage-2 worker subsequently occupied GPU3. The main process on
  GPU2 remained alive throughout.

---

## 2026-07-25 23:40:00 +08:00

- Created branch `feature/h800-cnn-stage12-v3-gen5` and worktree
  `/home/lixingfeng/UniAD_examine/heal_compress_h800_cnn_stage12_v3_gen5`
  from `0a7388a1f88f0ce51d48b96870ee6f7193c7c20c`.
- Confirmed old Pyramid/DiscoNet/F-Cooper configurations call
  `LidarPyramidTwoStageSearch` / `HealLidarBaselineTwoStageSearch`, use the old
  repaired GA objective, and therefore are not current-framework results.
- Added a shared CNN adapter around `StrictStage12V3Runner`; the three models
  now share legal-by-construction width/precision genes, exact BOPS hard gate,
  `J_struct_gate + J_WQ + J_AQ`, Stage-2 accuracy/latency scoring, and V1/V2/V3
  real archives.
- Added explicit `formal_gen5` scheduler contract. Default V2X-ViT
  `formal_gen10` remains exactly 10 generations.
- Added `scripts/run_cnn_formal_ga_gen5.py` with frozen invariants: seed 0,
  population/offspring/survivor 64, generation 0 initialization only,
  generations 1–5, per-generation new Stage-2 quota 5, and no full1789.
- Fixed Fisher `absolute_gradients` aggregation to `mean(abs(g))`, matching the
  existing `mean(g²)` Fisher sample reduction.
- Fixed activation Taylor at raw floating input Q/DQ boundaries by promoting
  non-gradient floating leaves to `requires_grad=True` without altering their
  forward values. Pyramid real preflight then covered 21 structure domains and
  69 activation/precision loci.
- Target GPU mapping for the formal concurrent runs is Pyramid→GPU4,
  DiscoNet→GPU5, F-Cooper→GPU6. The active V2X-ViT process on GPU2/3 remains
  untouched.
- Targeted tests: 31 passed. Full suite: 1036 passed. `compileall` and
  `git diff --check` passed. Real five-generation searches remain pending at
  this checkpoint.

---
