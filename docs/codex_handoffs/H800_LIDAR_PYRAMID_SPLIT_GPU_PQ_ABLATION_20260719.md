# H800 lidar_pyramid split-GPU P/Q ablation handoff

## Corrected scope

This round supersedes the GPU assignment used by the earlier dual-GPU
cross-card consistency audit. The required assignment is:

- physical GPU 0: fresh strict-FP32 baseline followed only by the six GA
  budgets and their P+Q, P-only, and Q-only engines;
- physical GPU 1: fresh strict-FP32 baseline followed only by the six greedy
  budgets and their P+Q, P-only, and Q-only engines.

Each GPU is strictly serial internally. The independent GPU 0 and GPU 1
sequences run concurrently. A completed item exits its evaluation process,
executes explicit CUDA allocator cleanup, and must pass an nvidia-smi
memory-return check before the next item on that GPU starts.

## Protocol and provenance

- fixedK = 29696;
- fixed 1789-frame manifest and identical frame order;
- warmup = 200 followed by iterator reset;
- latency rounds = 3;
- DataLoader workers = 8;
- CUDA NMS/IoU postprocess required;
- 38/38 evaluations completed 1789/1789 with zero skips;
- 38/38 cache cleanup audits passed;
- all source engine hashes and six deployment acceptance reports passed;
- source engine size, mtime, and SHA256 were unchanged after evaluation;
- evaluation output contains no `.plan`, `.engine`, `.onnx`, or `.pth`;
- engine build count = 0.

The P-only and Q-only engines come from the completed contribution ablation.
P+Q uses the accepted formal GA/greedy winner engine, except corrected greedy
0.30, which uses the already completed `R_BOPS=0.301922` artifact. Some budget
rows share an identical P-only physical engine; they were still independently
loaded and evaluated as requested instead of copying the prior metric.

## Fresh same-method baselines

| Assignment | AP30 | AP50 | AP70 | mAP | p50/p90/p99 ms |
|---|---:|---:|---:|---:|---:|
| GPU 0 / GA FP32 | 0.826229 | 0.781716 | 0.602283 | 0.736743 | 7.3182 / 10.8890 / 41.1455 |
| GPU 1 / Greedy FP32 | 0.826305 | 0.781714 | 0.602245 | 0.736754 | 7.2797 / 10.8448 / 44.8341 |

Every speedup below uses the FP32 reference on the same assigned GPU.

## Pruning/quantization contribution summary

| Method | Budget | Actual BOPS | P+Q mAP | P-only mAP | Q-only mAP | P+Q/P/Q p50 ms | P+Q/P/Q speedup |
|---|---:|---:|---:|---:|---:|---:|---:|
| GA | 0.30 | 0.304223 | 0.736572 | 0.736489 | 0.736575 | 6.7442/6.2376/7.3148 | 1.0851/1.1732/1.0005x |
| GA | 0.25 | 0.254517 | 0.736887 | 0.736740 | 0.736762 | 5.0045/6.2461/5.2075 | 1.4623/1.1716/1.4053x |
| GA | 0.20 | 0.204300 | 0.736696 | 0.736699 | 0.736620 | 4.9437/6.1205/5.0114 | 1.4803/1.1957/1.4603x |
| GA | 0.15 | 0.154526 | 0.736673 | 0.736690 | 0.736608 | 4.6689/6.1841/4.8448 | 1.5674/1.1834/1.5105x |
| GA | 0.10 | 0.104957 | 0.736654 | 0.736774 | 0.736810 | 3.6175/6.2263/3.6670 | 2.0230/1.1754/1.9957x |
| GA | 0.05 | 0.054807 | 0.707625 | 0.736298 | 0.707421 | 3.3380/5.8199/3.4921 | 2.1924/1.2574/2.0957x |
| Greedy | 0.30 | 0.301922 | 0.736824 | 0.736704 | 0.736666 | 6.7319/6.2691/7.3641 | 1.0814/1.1612/0.9885x |
| Greedy | 0.25 | 0.249080 | 0.736700 | 0.736678 | 0.736781 | 6.7153/6.2261/6.8731 | 1.0840/1.1692/1.0591x |
| Greedy | 0.20 | 0.196774 | 0.736652 | 0.736672 | 0.736754 | 4.8704/6.1069/5.0080 | 1.4947/1.1921/1.4536x |
| Greedy | 0.15 | 0.149413 | 0.736880 | 0.736761 | 0.736598 | 4.5397/6.2036/4.6056 | 1.6036/1.1735/1.5806x |
| Greedy | 0.10 | 0.098478 | 0.736670 | 0.736715 | 0.736755 | 3.4261/6.1358/3.4670 | 2.1248/1.1864/2.0997x |
| Greedy | 0.05 | 0.049994 | 0.699694 | 0.736374 | 0.698780 | 3.2948/5.8008/3.3811 | 2.2095/1.2549/2.1530x |

## Interpretation

At budgets 0.10--0.30, all three variants remain effectively FP32-accurate.
P-only consistently gives a modest `1.16--1.26x` speedup. At 0.10, Q-only
already provides nearly all P+Q acceleration (`1.996x` versus `2.023x` for GA,
`2.100x` versus `2.125x` for greedy).

At 0.05, P-only remains near the fresh FP32 reference while Q-only collapses
to approximately the same accuracy as P+Q:

- GA: FP32 `0.736743`, P-only `0.736298`, Q-only `0.707421`, P+Q `0.707625`;
- greedy: FP32 `0.736754`, P-only `0.736374`, Q-only `0.698780`, P+Q `0.699694`.

The formal interaction term `mAP(P+Q)-mAP(P)-mAP(Q)+mAP(FP32)` is small:
`0.000649` for GA 0.05 and `0.001294` for greedy 0.05. This independently
confirms that aggressive mixed quantization is the primary 0.05 AP-cliff
source; the searched structured pruning mask is not.

## Code and tests

- `search/integration/dual_gpu_fair_evaluation.py`
  - added method-specific 19-item inventory construction;
  - exact P+Q source resolution and local P-only/Q-only resolution;
  - same-GPU baseline summary fields;
  - P/Q contribution and interaction calculation.
- `scripts/run_lidar_pyramid_split_gpu_ablation_evaluation.py`
  - fixed assignment `GA -> GPU0`, `greedy -> GPU1`;
  - parallel cross-GPU and serial same-GPU scheduling;
  - immutable engine validation, cache cleanup, full reporting.
- `tests/test_search_dual_gpu_fair_evaluation.py`
  - added source-routing, 19-item inventory, and contribution tests.

Commands:

```bash
python -m pytest -q \
  tests/test_search_dual_gpu_fair_evaluation.py \
  tests/test_search_evaluation_provider.py \
  tests/test_search_lidar_pyramid_prune_quant_ablation.py

python scripts/run_lidar_pyramid_split_gpu_ablation_evaluation.py
git diff --check
```

## Evidence

```text
outputs/h800_lidar_pyramid_split_gpu_ablation_20260718_171525/
```

Small reports suitable for selective Git inclusion:

```text
split_gpu_evaluation_manifest.json
split_gpu_full_results.csv
split_gpu_ablation_comparisons.csv
split_gpu_ablation_report.json
split_gpu_ablation_report.md
```

The two `gpu_*_completed.json` files are about 36 MB each and remain ignored.
All per-evaluation logs and detailed frame data also remain local.

Completed checkpoint: 2026-07-19 08:49:39 +0800 CST

---
