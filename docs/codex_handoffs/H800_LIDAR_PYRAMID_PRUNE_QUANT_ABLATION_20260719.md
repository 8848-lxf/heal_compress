# H800 lidar_pyramid structured-pruning / mixed-quantization ablation handoff

## Round 1 — formal 12-budget three-way full-validation ablation

Objective: for every accepted GA and greedy winner at BOPS budgets
`{0.30, 0.25, 0.20, 0.15, 0.10, 0.05}`, compare:

- `prune_quant`: the searched physical pruning mask plus searched precision profile;
- `prune_only`: exactly the same physical mask with the strict FP32 profile;
- `quant_only`: original all-keep structure with exactly the same precision profile.

The strict FP32 profile has one mapped, protected FP16 functional affine-grid
MatMul entry, so its canonical EngineInspector count is `0 INT8 / 1 FP16 / 69
FP32`. This is the same audited exception used by the formal original FP32
baseline; there are no unmapped weighted layers.

### Production code and tests

Added:

- `search/ablation/lidar_pyramid_prune_quant.py`
  - validates the read-only authoritative GA/greedy source artifacts;
  - creates exact P+Q, P-only, and Q-only phenotypes;
  - strips pruning metadata only for Q-only all-keep deployment;
  - forces every parameterized precision group to FP32 for P-only while
    retaining the original physical mask and frozen grouped maps;
  - exactly replays the saved greedy adjacent-action path and verifies its
    genotype hash.
- `scripts/run_lidar_pyramid_prune_quant_ablation.py`
  - prepares the 36-row ablation manifest;
  - fresh-builds and evaluates the same-run strict FP32 reference;
  - re-evaluates proven historical P+Q engines without overwriting them;
  - builds all missing strongly-typed engines through the production Q/DQ
    path and performs structure/precision/merge/boundary acceptance checks;
  - serializes all evaluation work on one GPU and resumes from strict caches;
  - writes full CSV/JSON/Markdown results and additive interaction metrics.
- `tests/test_search_lidar_pyramid_prune_quant_ablation.py`
  - verifies exact-mask P-only conversion, all-keep Q-only conversion, and
    lowest-Taylor in-band greedy path replay.

### Greedy 0.30 correction

The prior `R_BOPS=0.282141` snapshot remains invalid and is excluded. The saved
formal greedy path contains two legal in-band states:

- step 104: `R_BOPS=0.3019222021`, joint Taylor `1.1943276e-05`;
- step 105: `R_BOPS=0.3006227016`, joint Taylor `1.2144507e-05`.

The formal hard-band objective selects step 104 because it has the lower Taylor
loss. Replay reproduced the logged genotype SHA-256 exactly. Its newly built
P+Q engine passed full validation with `mAP=0.736713448`, `p50=6.582260 ms`.

### Fair evaluation protocol

- physical H800 GPU 7, all engine evaluations serial;
- fixedK=29696;
- train200 TensorRT EntropyCalibration2 for every fresh INT8 topology;
- same 1789-frame validation manifest and ordering;
- warmup=200, iterator reset after warmup;
- DataLoader workers=8 and CUDA postprocess;
- latency rounds=3;
- evaluated=1789, skipped=0 for the baseline and all 36 rows.

Fresh strict FP32 reference:

| AP30 | AP50 | AP70 | mAP | p50 ms | p90 ms | p99 ms |
|---:|---:|---:|---:|---:|---:|---:|
| 0.826155 | 0.781573 | 0.602123 | 0.736617 | 7.032 | 7.239 | 28.111 |

### Principal results

The order in each result triple is `P+Q / P-only / Q-only`.

| method | budget | actual BOPS | mAP triple | p50-ms triple | P+Q speedup |
|---|---:|---:|---|---|---:|
| GA | 0.30 | 0.304223 | 0.736522 / 0.736767 / 0.736731 | 6.639 / 6.031 / 7.186 | 1.059x |
| GA | 0.25 | 0.254517 | 0.736830 / 0.736651 / 0.736846 | 4.951 / 6.080 / 5.047 | 1.420x |
| GA | 0.20 | 0.204300 | 0.736604 / 0.736601 / 0.736713 | 4.883 / 6.117 / 4.971 | 1.440x |
| GA | 0.15 | 0.154526 | 0.736592 / 0.736601 / 0.736733 | 4.564 / 6.117 / 4.677 | 1.541x |
| GA | 0.10 | 0.104957 | 0.736473 / 0.736671 / 0.736749 | 3.492 / 6.045 / 3.562 | 2.013x |
| GA | 0.05 | 0.054807 | 0.707684 / 0.736394 / 0.707059 | 3.244 / 5.730 / 3.337 | 2.167x |
| Greedy | 0.30 | 0.301922 | 0.736713 / 0.736716 / 0.736710 | 6.582 / 6.101 / 7.238 | 1.068x |
| Greedy | 0.25 | 0.249080 | 0.736807 / 0.736716 / 0.736821 | 6.606 / 6.101 / 6.649 | 1.064x |
| Greedy | 0.20 | 0.196774 | 0.736671 / 0.736716 / 0.736656 | 4.724 / 6.101 / 4.812 | 1.488x |
| Greedy | 0.15 | 0.149413 | 0.736802 / 0.736716 / 0.736565 | 4.338 / 6.101 / 4.468 | 1.621x |
| Greedy | 0.10 | 0.098478 | 0.736662 / 0.736623 / 0.736698 | 3.262 / 5.912 / 3.283 | 2.156x |
| Greedy | 0.05 | 0.049994 | 0.699707 / 0.736253 / 0.698748 | 3.157 / 5.687 / 3.228 | 2.227x |

Interpretation:

- At budgets 0.10 through 0.30, both pruning-only and quantization-only retain
  the fresh FP32 accuracy. Mixed quantization supplies most of the speedup at
  budgets 0.10--0.20, with pruning giving a smaller additional gain.
- At budget 0.05, pruning-only still gives about 0.736 mAP, whereas Q-only and
  P+Q both fall to about 0.699--0.708. The low-budget AP cliff is therefore
  caused primarily by the aggressive quantization profile, not by physical
  structured pruning.
- The mAP interaction term is small for every point (`-0.000359` to
  `+0.001323`); pruning does not amplify the 0.05 quantization collapse.
- Latency should be compared only against the fresh same-run FP32 p50 above;
  older GA/greedy FP32 references are not used in this report.

### Artifact and cache policy

Evidence root (not tracked by Git):

```text
outputs/h800_lidar_pyramid_prune_quant_ablation_20260718_124729/
```

Key small reports:

```text
ablation_manifest.json
greedy_030_repair.json
fresh_fp32_baseline.json
ablation_summary.json
ablation_full_results.csv
ablation_comparisons.csv
ablation_report.md
```

- 11 read-only historical P+Q engines were hash-verified and re-evaluated;
- 21 candidate engines were newly built, plus one fresh FP32 baseline engine;
- 4 exact candidate cache hits avoided duplicate engine construction and
  duplicate evaluation;
- no historical engine, ONNX, checkpoint, or experiment directory was
  overwritten.

Git delivery:

- implementation/results commit: `a811ce8` (`feat(search): add lidar pyramid prune quant ablation`);
- branch: `feature/heal-unified-search-h800`;
- HTTPS push was attempted but the server's VS Code Git credential socket was
  stale and GitHub rejected anonymous write access; no credential, remote URL,
  or system configuration was changed.

Completed: 2026-07-19 05:47:26 CST

---
