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
