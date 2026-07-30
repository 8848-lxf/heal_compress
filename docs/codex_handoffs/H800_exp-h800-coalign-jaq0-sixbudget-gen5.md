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
