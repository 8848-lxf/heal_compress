# DiscoNet Greedy Six-Budget Results

This report records the first complete real deployment run for the legal-width
DiscoNet family path. The run used the strongly typed TensorRT 10.9 pipeline,
GPU7, GPU AP-IoU, eight DataLoader workers, fixedK=29696, and one shared
strict-FP32 reference. Candidate artifacts were keyed by physical,
precision, deployment, calibration, and engine identity. Results below use
the fixed 1789-frame validation manifest; no smoke-only metric is promoted to
the final table.

## Baseline

```text
strict FP32 mAP                 0.6357519111
AP30/AP50/AP70                  0.7341804587 / 0.6628115848 / 0.5102636900
physical params                 8,128,789
R_BOPS / R_MAC / param retain   1.000000 / 1.000000 / 1.000000
forward p50/p90/p95 (ms)        7.758738 / 17.115440 / 20.638604
total p50/p90/p95 (ms)          13.469938 / 26.214755 / 33.674918
evaluated/skipped               1789 / 0
```

## Greedy endpoints

| target | realized BOPS | R_MAC | parameter prune | mAP | AP30 | AP50 | AP70 | forward p50 ms | total p50 ms | precision (INT8/FP16/FP32) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 0.05 | 0.054996 | 0.366534 | 0.730311 | 0.634540 | 0.732817 | 0.661583 | 0.509221 | 3.217041 | 7.922273 | 12/18/2 |
| 0.10 | 0.102305 | 0.411364 | 0.572471 | 0.636059 | 0.734395 | 0.662965 | 0.510817 | 3.481150 | 8.807734 | 9/15/8 |
| 0.15 | 0.152289 | 0.419845 | 0.525510 | 0.636044 | 0.734402 | 0.663024 | 0.510707 | 3.593439 | 9.083716 | 2/17/13 |
| 0.20 | 0.200173 | 0.420671 | 0.520936 | 0.635947 | 0.734442 | 0.662856 | 0.510542 | 3.914453 | 8.966593 | 2/14/16 |
| 0.25 | 0.252669 | 0.475557 | 0.431270 | 0.635771 | 0.734212 | 0.662824 | 0.510277 | 4.361611 | 9.606978 | 0/13/19 |
| 0.30 | 0.304441 | 0.428912 | 0.475308 | 0.635952 | 0.734357 | 0.663035 | 0.510466 | 4.446868 | 9.177654 | 1/12/19 |

All six rows are `1789/1789`, `0 skip`, and passed physical structure,
strongly typed, QDQ boundary, merge, precision identity, and realized-BOPS
audits. The full result table with hashes is stored in the run output as
`disco_greedy_six_budget_full_results.csv/json`.

Relative to the strict-FP32 reference, the lowest-BOPS endpoint retains
26.97% of physical parameters and 9.96% of weight storage, with forward
speedup `2.412x` and total-pipeline speedup `1.700x`; its absolute mAP change
is `-0.001212`. These are screening/shared-GPU timings, not isolated formal
latency rankings. The GPU identity, manifest hashes, engine hashes, physical
hashes, and deployment hashes are in the machine-readable result files.

## Search status

```text
GREEDY_STAGE1_COMPLETE=true
GREEDY_STAGE2_BUILD_SUCCESS=6/6
GREEDY_FULL_VALIDATION_SUCCESS=6/6
GREEDY_NORMAL_REPAIR_INVOCATIONS=0
GA_STARTED=false (at report creation)
```

The search used the fixed pure-pruning Taylor ranking and legal keep-width
genotypes. No candidate was backfilled by copying another phenotype. Reused
artifact evidence is recorded with `cache_hit` and `reuse_reason`; a fresh
process-pool attempt namespace prevents stale task IDs from being reused after
an interrupted controller.

--- ROUND 1 | 2026-07-19 22:45:00 +0800 ---
