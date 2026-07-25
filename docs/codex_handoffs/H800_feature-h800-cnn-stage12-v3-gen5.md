# H800 CNN Strict Stage-1/Stage-2/V1–V3 GA Gen5

This handoff tracks the isolated re-search of Pyramid, DiscoNet and F-Cooper
with the current strict GA framework. Historical two-stage GA artifacts are
read-only and are never used as current-framework search results.

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
