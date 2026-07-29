# H800 DiscoNet/F-Cooper J_AQ=0 GA ablation handoff

## 2026-07-29 04:04:09 CST

- Branch: `exp/h800-disco-fcooper-r005-ga-jaq0-gen5`.
- Worktree: `/data/lxf/heal_data/worktrees/heal_compress_h800_disco_fcooper_r005_ga_jaq0`.
- Base: `051db98f07e04efc8a1c59c29ea64b62a6b38d21`, the accepted Pyramid `J_AQ=0` implementation and physical-GPU binding fixes.
- The first worktree checkout under `/home` failed before registration because the root filesystem had only about 595 MiB free. The isolated worktree was therefore created under `/data`; no historical artifact was deleted.
- Generalized the controlled `J_AQ=0` gate from Pyramid-only to the explicitly approved CNN set `{pyramid, disco, fcooper}`. The contract remains fail-closed for a single `R_BOPS=0.05` target; activation-Taylor deployment remains enabled while its fitness weight is zero.
- Added tests for the approved model set and rejection of unapproved models. Targeted result: `19 passed`; `py_compile` and `git diff --check` passed.
- Code commit: `23bff5d9` (`exp: enable controlled JAQ0 CNN ablations`), pushed to origin.
- DiscoNet run: GPU6 (`GPU-cd2c090b-dc30-681c-1380-886b8bcabee5`), output `/data/lxf/heal_data/outputs/h800_disco_r005_ga_jaq0_gen5_20260729_040409`.
- F-Cooper run: GPU7 (`GPU-fc1ca600-f06e-f791-125e-102e4916449b`), output `/data/lxf/heal_data/outputs/h800_fcooper_r005_ga_jaq0_gen5_20260729_040409`.
- Both runs use seed 0, five formal generations, population/offspring 64/64, Stage-2 quota 5, fixed300+warmup100 screening, fixed500+warmup200 generation-winner validation, and `activation_taylor_fitness_weight=0`.
- Existing `J_AQ=1` runs on GPU3/GPU5 use separate worktrees and output roots and were not modified.

---

