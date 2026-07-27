# H800 V2X-ViT R=0.05 GA `J_AQ=0` controlled ablation

Branch: `exp/h800-v2xvit-r005-ga-jaq0`  
Worktree: `/home/lixingfeng/UniAD_examine/heal_compress_h800_v2xvit_r005_ga_jaq0`  
Base: `ee1306c3986e6cd328eb0cc8feacb3bac88116ac`  
Implementation commit before final reporting: `30b59a050153cf24e4452fc07504b8fae555c6a4`  
Run root: `/data/lxf/heal_data/outputs/h800_v2xvit_r005_ga_jaq0_stage2_300_genwinner500_20260726_222720`

## Controlled code change

- `search/ga/stage12_v3.py`
  - Added an explicit, finite, non-negative `activation_taylor_fitness_weight`.
  - Preserved raw `J_AQ` as a diagnostic while defining Stage-1 fitness as
    `J_struct + J_WQ + activation_taylor_fitness_weight * J_AQ`.
  - Persisted raw/weighted activation terms and whether activation Taylor was used.
- `search/greedy/weight_only_abs.py`
  - Applied the same coefficient to Greedy and removed a cumulative-score path that
    could otherwise leak activation risk into the `J_AQ=0` objective.
- `scripts/run_v2xvit_six_budget_proxy.py`
  - Added target-subset and activation-coefficient arguments; convergence checks now
    use the exact configured objective.
- `scripts/run_v2xvit_six_budget_formal_ga_gen10.py`
  - Enforced matching Greedy/GA activation coefficients and recorded the controlled
    ablation contract while keeping actual activation quantization enabled.
- `tests/test_v2xvit_activation_taylor_ablation.py`
  - Covers raw-vs-weighted `J_AQ`, zero-weight behavior, invalid weights, deployment
    quantization preservation, and Greedy/GA contract consistency.
- `scripts/finalize_v2xvit_jaq0_ablation.py`
  - Report-only validator for generation count, Stage-2/fixed500 protocols, fresh
    train200, precision realization, resource metrics, and the `J_AQ=1` comparison.

The search space, legal widths, BOPS definition/denominator, structure gate Taylor,
weight quantization Taylor, precision deployment contract, manifests, and Stage-2
selection formula were not changed.

## Search and deployment result

- Taylor 16→32 convergence: Spearman `0.9998062766369625`, top-10 overlap `1.0`,
  top-20 overlap `1.0`.
- Greedy completed 757 selected steps with zero structural/precision/budget repair.
- Formal GA used seed 0, generation 0 initialization, and exactly generations 1–10.
- Each formal generation produced 64 offspring and retained 64 legal, unique,
  in-band survivors.
- Each generation evaluated exactly five new physical candidates: 50/50 engine
  builds succeeded, all requested/realized audits were exact, and all candidates
  completed fixed300 with zero skips.
- The Greedy anchor plus one unique winner from each generation completed fixed500
  with warmup 200: 11/11 evaluated 500, zero skipped.
- Final fresh calibration processed train200 `200/200`, zero skipped; precision
  conflicts, unmapped units, and fallbacks were all zero.

Greedy exact anchor:

- Hash: `6bd022095b96c1fed40974ab7704023ef59d3ca85c06042d74179dc9586d3f28`
- `R_BOPS=0.05473454187919699`
- Parameters: `4,210,277`, pruning rate `68.7043%`
- Mixed-weight compression: `6.7268x`
- Precision genes: 53 INT8, 0 FP16, 0 FP32
- Shrinker: 52; FFN: 256/256/196
- Fixed500 AP30/AP50/AP70/mAP: `0.445166/0.368494/0.030291/0.281317`

Final GA winner (generation 3):

- Hash: `8dd001ed683ec2815bcc4f698fef60e9f3753139cdecf6999a3acf43802aff70`
- Engine SHA256: `3d95c8f7c8a3db81636692224a9cf98c839a0259b93f557972fdda42c421eea4`
- `R_BOPS=0.05496807393450855`, BOPS compression `18.1924x`
- Parameters: `4,176,349`, retention `31.0435%`, pruning rate `68.9565%`
- Mixed-weight compression: `6.6862x`
- Precision genes: 51 INT8, 2 FP16, 0 FP32
- Shrinker: 52; FFN: 256/256/196
- Fixed500 AP30/AP50/AP70/mAP: `0.455326/0.374846/0.030487/0.286887`
- Fixed500 forward p50: `6.42899 ms`; this is not the isolated formal-latency
  protocol and must not be reported as formal speedup.

Against the same-manifest strict B0 mAP `0.658670`, final mAP retention is `43.56%`:
the result remains catastrophic at R=0.05.

Compared with the prior `J_AQ=1` final GA reference:

- mAP changes from `0.255447` to `0.286887` (`+0.031440`).
- Shrinker changes from 28 to 52.
- Parameter pruning falls from `72.7046%` to `68.9565%`.
- INT8/FP16 changes from 44/9 to 51/2.
- FFN changes from 256/240/164 to 256/256/196.

Thus activation Taylor was a material driver of the more extreme structural path:
penalizing activation INT8 actions made structure pruning comparatively cheap. Turning
it off reallocates compression toward INT8 and preserves more structure, but protected
FP32 compute plus the 5% global BOPS target still force destructive pruning.

## Verification and restrictions

- Final targeted tests: 39 passed.
- Final full pytest: 1067 passed, 82 warnings.
- Compileall, py_compile, and `git diff --check` passed after report generation.
- GPU used: physical GPU2, UUID
  `GPU-cc8e76f2-33c7-4dcf-0f77-dbf3354fcd54`; it was idle after completion.
- No full1789, additional budget, multi-seed search, or formal isolated latency was run.

Reports:

- `reports/final_acceptance.json`
- `reports/jaq0_vs_joint_summary.json`
- `reports/jaq0_vs_joint_summary.md`
- `reports/jaq0_generation_summary.csv`
- `reports/jaq0_fixed500_metrics.csv`
- `reports/jaq0_vs_joint_metrics.csv`
- `root_conclusion.md`

---

Timestamp: 2026-07-27 20:18 Asia/Shanghai
