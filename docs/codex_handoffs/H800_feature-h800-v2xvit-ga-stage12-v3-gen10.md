# H800 V2X-ViT GA Stage-1/Stage-2 V3 Gen10 handoff

Branch: `feature/h800-v2xvit-ga-stage12-v3-gen10`  
Worktree: `/home/lixingfeng/UniAD_examine/heal_compress_h800_v2xvit_ga_stage12_v3_gen10`  
Base commit: `6edd7164a3b52e8d00d8cf1c1daef638821478ce`  
Run root: `/data/lxf/heal_data/outputs/h800_v2xvit_ga_stage12_v3_gen10_20260724_231128`

The formal `feature/heal-unified-search-h800` worktree is read-only and remains at
`460764fdecf9ea6298667586c033a51aa4b71c7d`.

## Development progress

- Separated Greedy exact-winner validation from GA Stage-2 terminology. Greedy
  captures one proxy-selected winner per budget and never performs a Top-5
  replacement.
- Added strict GA Stage-1/Stage-2 V1/V2/V3 implementation in
  `search/ga/stage12_v3.py`: fixed mutable schema, adjacent mutation,
  same-locus crossover, exact BOPS hard gate before Taylor evaluation, physical
  hash deduplication, no repair, accuracy-gated real feedback, permanent Greedy
  anchor, and global real anchors.
- Enforced generation 0 as initialization only and generations 1--10 as the
  exact ten evolution generations. Population and offspring are both 64, and
  the new real Stage-2 quota is at most five per generation.
- Fixed the formal proxy aggregation so gate and activation statistics are
  computed independently for each sample and averaged only after elementwise
  absolute contributions. Formal Fisher is `E[g^2]` on the frozen train32
  prefix; joint/cross and legacy coupled-weight scores remain diagnostic only.
- Removed the Greedy 400-step cutoff and completed one 858-step trajectory that
  captured exact winners for 0.30, 0.25, 0.20, 0.15, 0.10 and 0.05 with zero
  structural, precision, or budget repair.
- Taylor 16-to-32 convergence passed: Spearman
  `0.9998579657539651`, top-10 overlap `1.0`, top-20 overlap `1.0`.
- Built B0 plus S32/JMIX engines for all six exact winners. All 13 TensorRT
  engines built successfully with requested/realized precision exact. INT8
  winners used fresh train200 calibration with 200 processed and zero skipped.
- Completed shared-manifest fixed500 on all 13 controls with 500 evaluated and
  zero skipped. Budgets 0.30 through 0.10 have both S32 and JMIX drops below
  0.01; budget 0.05 is catastrophic (B0 mAP `0.6583137730`, S32
  `0.2652507393`, JMIX `0.2574539018`).
- Prepared deterministic natural in-band GA populations for budgets 0.30--0.10:
  one exact Greedy anchor, 16 structure-biased, 16 precision-biased, 16 mixed,
  and 15 deterministic fill candidates per budget. No candidate is projected or
  repaired.
- Added formal GA orchestration with real physical materialization, fresh
  train200 when INT8, strongly typed TensorRT build, exact realization audit,
  fixed50 and screening p50 feedback. Completed Stage-2 artifacts are cached by
  complete phenotype hash without counting historical elites against the next
  generation's new-engine quota.
- Targeted GA/proxy tests currently pass (`14 passed`); full regression is
  pending after experiment completion.

The six-budget matched B0 pre/post formal latency run is in progress on physical
GPU6 (`GPU-cd2c090b-dc30-681c-1380-886b8bcabee5`). The first completed budget,
0.30, reports B0/S32/JMIX p50 `26.5998/25.2886/16.9896 ms`, corresponding to
`1.0519x/1.5657x` S32/JMIX speedup. GA admission will be frozen only after all
six latency batches satisfy the <=1% replay-drift gate.

---
Timestamp: 2026-07-25T16:05:00+08:00

## Development progress

- Completed matched B0 pre/post latency for every Greedy budget. Thermal
  preconditioning was added as an explicitly unmeasured protocol field; the
  acceptance gate remains an uncompromised maximum absolute replay drift of
  1%. All six budgets now pass, including 0.25 (0.259% maximum drift) and 0.05
  (0.188% maximum drift).
- Frozen GA admission to 0.30, 0.25, 0.20, 0.15 and 0.10. Budget 0.05 is
  `GA_UNSAFE` because its exact-winner fixed500 S32/JMIX accuracy is
  catastrophic; its valid 2.717x latency speedup does not override accuracy.
- Added `scripts/run_v2xvit_005_structural_rescue.py` for the requested
  independent root-cause controls: restore FFN to 256/256/256, restore only the
  shrinker, restore Attention to the exact 0.10 widths, restore only backbone
  stage 2, and reconstruct exact 0.08/0.07/0.06 winners from the frozen Greedy
  trace. These are marked `diagnostic_control=true` and cannot alter formal
  Greedy winners or GA admission.
- The diagnostic reconstructs the train32 gate ordering and requires the known
  0.10 and 0.05 complete phenotype hashes to match before building any control.
  Each control is physically materialized, exported as S32 TensorRT, and
  evaluated on the shared fixed500 manifest with a 500/0 evaluated/skipped
  hard gate.

---
Timestamp: 2026-07-25T19:15:00+08:00

## Development progress

- Completed all seven requested structure-only fixed500 controls with 500
  evaluated and zero skipped. Relative to B0 mAP `0.658313773`, the frozen 0.05
  S32 mAP is `0.265250739`. Restoring FFN gives `0.265711425`, restoring
  backbone Stage-2 gives `0.267924153`, restoring Attention to the 0.10 widths
  gives `0.365126036`, and restoring only the shrinker gives `0.477695194`.
- The primary single subsystem is therefore the shrinker, Attention d_h is
  secondary, and FFN/backbone Stage-2 are rejected as primary causes. No single
  restore reaches the safe range, so the residual failure is an extreme
  multi-subsystem structure interaction.
- Exact intermediate Greedy structures produce S32 fixed500 mAP
  `0.614610649` at 0.08, `0.552461098` at 0.07 and `0.463600488` at 0.06.
  Significant structural loss first appears at sampled budget 0.08, severe
  loss at 0.07, and catastrophic loss at 0.05.
- Added an anchor-constrained domain reconstruction. Six serialized exact
  winners bind every observed prune mask, Attention/FFN decoded keep set, and
  ranking hash; newly sampled widths use the fresh train32 order only inside
  unconstrained nested segments. This prevents CUDA near-tie replay from
  changing the frozen physical anchor.
- Full canonical phenotypes for 0.10 and 0.05 match byte-for-byte. Candidate
  identity hashes are explicitly rebased because the hash contract also binds
  the newly generated trace snapshot hash; source and rebased hashes are both
  retained rather than misreporting this as physical drift.

---
Timestamp: 2026-07-25T19:50:00+08:00

## Development progress

- The user superseded the original three-seed request. Formal GA now uses only
  `seed_0`; generation 0 remains initialization and generations 1--10 remain
  exactly ten evolution generations. The initial run was stopped during
  budget-0.30 seed-0 generation 3 before any second seed existed. Completed
  phenotype Stage-2 cache entries remain reusable; no external process was
  signalled.
- Added the 0.10/0.08/0.07/0.06/0.05 compression audit. Parameter pruning
  ratios are respectively 46.8658%, 54.2981%, 58.1139%, 63.8103% and 68.4594%.
  INT8 mutable-locus coverage is respectively 49/80, 50/80, 53/80, 55/80 and
  56/80 (61.25%, 62.50%, 66.25%, 68.75%, 70.00%). Fixed protected loci are
  deliberately excluded from this coverage denominator.

---
Timestamp: 2026-07-25T20:15:00+08:00

## Development progress

- Completed an interim same-protocol final comparison for the only fully
  completed formal-GA budget, 0.30. The GA run uses seed 0, generation 0 for
  initialization, and exactly generations 1--10 for evolution. It performed
  50 real Stage-2 evaluations and selected phenotype
  `fa1c14bdc183ab293afe55295667e220521bf6bcf489e1b694064ac948900ee4`.
- Re-evaluated that GA winner on the frozen fixed500 manifest
  `912b2d367be26af967474f6271f60034004f7ee8d68119d65b5e41c722fd24c1`:
  AP30/AP50/AP70/mAP are
  `0.771701424/0.694040246/0.512465172/0.659402281`, with 500 evaluated and
  zero skipped. The same-manifest B0 mAP is `0.658313773`; the exact Greedy
  winner mAP is `0.659306638`. The GA-minus-Greedy difference is only
  `+0.000095643` and is treated as evaluation noise rather than an accuracy
  improvement.
- Ran a new matched B0 pre/post TensorRT latency batch on physical GPU1
  (`GPU-fecad09b-d28a-8f1a-6da8-a29fd4ddd659`). B0, exact Greedy, and GA-final
  p50 are `15.936368/11.730688/11.700176 ms`, equivalent to
  `1.0000x/1.35852x/1.36206x` versus B0. Maximum absolute B0 replay drift is
  `0.1083%`, below the 1% acceptance threshold. GA is only `0.2608%` faster
  than Greedy in this batch, so the practical latency difference is marginal.
- GPU1 also hosted an external user01 training process at approximately 17%
  utilization and 10.7 GiB memory before the test. No signal was sent to it.
  Matched replay drift passed, but the co-resident load is retained in the
  process snapshots and should be considered when interpreting sub-percent
  Greedy-versus-GA differences.
- Formal GA for budget 0.25 continued independently on the multi-GPU Stage-2
  pool and had reached generation 9 at the end of this comparison. No result
  for an incomplete budget is reported as final.

---
Timestamp: 2026-07-25T20:40:00+08:00
