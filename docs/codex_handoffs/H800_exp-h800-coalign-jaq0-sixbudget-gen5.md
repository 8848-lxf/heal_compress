# H800 CoAlign J_AQ=0 six-budget gen5 search

## 2026-07-30T14:43:00+08:00

- Branch: `exp/h800-coalign-jaq0-sixbudget-gen5`.
- Worktree: `/home/lixingfeng/UniAD_examine/heal_compress_h800_coalign_jaq0_sixbudget_gen5`.
- Base: `295dedca425a2a5d24717a20b2030b8d65eb9b3b`.
- CoAlign is audited as `HeterModelBaselineMs` with three real, independent,
  projection-free `AttFusion` scales at 64/128/256 channels.  It has no
  trainable Q/K/V/O projections and therefore does not receive a synthetic
  attention-d_h domain.
- Added the `heal_lidar_coalign` family, strict checkpoint/config/structure
  validation, real-forward filtering for the intentionally dormant fusion
  backbone layer-0 tensors, fixed-K multi-scale export registration, runtime
  precision coupling, evaluator routing, and formal CNN GA model spec.
- Added a 21-domain physical pruning topology: 16 residual-block hidden
  domains, three residual-stream/attention-feature domains, and two shrinker
  domains.  Deblock outputs and detection outputs remain fixed.
- A real GPU preflight produced 44/44 active weighted-module coverage,
  3,840 dependency-closed atomic units, 21 width domains and 44 precision
  genes.  The six dormant `backbone.resnet.layer0.*` convolution tensors are
  recorded explicitly and excluded rather than becoming unbuildable genes.
- Real physical replay of scale-0 width 64→60 rewrote 13 coupled members,
  including the next residual level and deblock input.  The parameter-free
  attention scale was updated from `sqrt(64)` to `sqrt(60)`; strict replay and
  finite forward passed with zero repair.
- Targeted family/pruning/deployment/GA tests: 54 passed.
- Planned formal run: six budgets 0.30/0.25/0.20/0.15/0.10/0.05, seed 0,
  five generations, activation Taylor disabled in fitness, one isolated GPU.

---

## 2026-07-30T20:18:40+08:00

- Recovered the interrupted session and audited the two post-launch CoAlign
  failures without changing the six exact Greedy phenotypes.  Budgets
  0.30/0.25/0.20/0.15 had already completed Q/DQ, TensorRT and fixed300
  evaluation; only 0.10 and 0.05 failed at residual `conv2` output-scale
  ownership.
- Root cause: TensorRT entropy-cache conversion resolved `Conv -> Add -> Relu`
  to the post-ReLU tensor, while adaptive-merge Q/DQ insertion correctly
  stopped at the raw Conv output for an INT8 producer.  Calibration and
  insertion therefore assigned one scale to different tensors.
- Commit `e23fb2f3` makes calibration, Q/DQ insertion and boundary audit use
  one realized-precision-aware resolver.  The HEAL family evaluator now builds
  one Q/DQ config and passes it to both calibration and insertion.  Stale
  failed Stage-2 cache entries are archived and retried once under a versioned
  cache schema; successful artifacts remain immutable and reusable.
- Regression: the adaptive-merge fixture proves entropy-cache scale ownership,
  Q-node placement and boundary audit all stop before the merge.  Relevant
  tests passed (126), the cache/GA-focused tests passed (46), and the complete
  CPU suite passed (1066).  `compileall`, `py_compile` and `git diff --check`
  passed.
- Resumed run root:
  `/data/lxf/heal_data/outputs/h800_coalign_jaq0_sixbudget_gen5_shape_merge_fix_20260730_181000`.
  The 0.10 candidate now passes with fixed300 mAP `0.7300615456`, p50
  `3.04116495 ms`, requested/realized INT8 `4/4`, and zero skipped frames.
  The 0.05 candidate passes with fixed300 mAP `0.7263172383`, p50
  `2.79724714 ms`, requested/realized INT8 `22/22`, and zero skipped frames.
- All six Greedy deployment gates are now accepted and the formal
  J_AQ=0, seed-0, five-generation GA has started on physical GPU0
  (`GPU-2133c0e5-4e29-a3aa-d7c6-c45046e073f7`).  No other model process or
  output path was modified.

---
