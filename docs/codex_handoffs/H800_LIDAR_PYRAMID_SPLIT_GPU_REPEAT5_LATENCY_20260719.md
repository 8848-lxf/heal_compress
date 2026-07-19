# H800 lidar_pyramid split-GPU five-repeat latency handoff

## Scope and assignment

This round repeats the corrected split-GPU ablation protocol five complete
times. It does not build or modify engines:

- physical GPU 0 evaluates the GA inventory only;
- physical GPU 1 evaluates the greedy inventory only;
- each GPU evaluates its own strict-FP32 reference, followed by the six BOPS
  budgets as P+Q, P-only, and Q-only;
- execution is serial within a GPU and parallel across the two GPUs;
- each item exits its evaluation process and passes CUDA cache/memory-return
  validation before the next item starts.

Each repeat is a fresh full-validation traversal rather than another
TensorRT latency loop over cached results. The evaluation protocol remains
fixedK29696, 1789 formal frames, warmup200 followed by iterator reset,
latency rounds 3, DataLoader workers 8, and CUDA postprocess.

## Latency semantics added in this round

The production report now retains arithmetic mean latency in addition to
p50/p90/p99. For every formal evaluated frame it records:

- `forward_ms`: TensorRT forward latency;
- `postprocess_ms`: CUDA postprocess latency;
- `total_ms`: `forward_ms + postprocess_ms`.

Warmup frames are deliberately excluded from the per-frame CSV. A run-level
`mean` is the arithmetic mean over its 1789 formal frames; the five-repeat
result is the arithmetic mean of the five independent run-level means. The
aggregate output also records across-run standard deviation, minimum, and
maximum.

## Completion and integrity

- repeat count: 5;
- evaluation count: 190/190;
- every evaluation: 1789/1789, zero skipped;
- frame-order hash identical: true;
- GPU cleanup audits passed: 190/190;
- engine builds invoked: 0;
- new model/ONNX/engine files in evaluation output: 0;
- source per-evaluation latency CSV count: 190;
- consolidated formal-frame rows: 339,910;
- every evaluation group contributes exactly 1789 rows;
- consolidated CSV SHA256:
  `a46d8594d9d7a457cc454bb39a9a081992840746ef49fb69534a4e58674fd258`.

## Five-repeat baseline means

| Assignment | AP30/50/70 | mAP mean +/- std | Forward mean/p50/p90/p99 ms | Post mean ms | Total mean ms |
|---|---|---:|---:|---:|---:|
| GPU 0 / GA FP32 | 0.826228/0.781615/0.602286 | 0.736710 +/- 0.000034 | 8.8726/7.1696/7.4770/30.6123 | 5.2975 | 14.1701 |
| GPU 1 / Greedy FP32 | 0.826197/0.781572/0.602154 | 0.736641 +/- 0.000076 | 8.8556/7.1454/7.4390/33.3231 | 5.1320 | 13.9876 |

## Five-repeat ablation means

Each slash-separated cell is ordered as `P+Q / P-only / Q-only`.

| Method | Budget | mAP | Forward mean ms | Post mean ms | Total mean ms |
|---|---:|---:|---:|---:|---:|
| GA | 0.30 | 0.736724/0.736682/0.736722 | 8.2726/7.9145/8.9365 | 5.2708/5.1282/5.2079 | 13.5434/13.0427/14.1444 |
| GA | 0.25 | 0.736732/0.736710/0.736743 | 6.6442/7.8528/6.6626 | 5.2048/5.1423/5.1895 | 11.8489/12.9951/11.8520 |
| GA | 0.20 | 0.736647/0.736662/0.736743 | 6.3668/7.9864/6.5373 | 5.2045/5.3123/5.1904 | 11.5713/13.2987/11.7278 |
| GA | 0.15 | 0.736625/0.736751/0.736671 | 6.1143/7.9799/6.3055 | 5.3302/5.3764/5.5653 | 11.4444/13.3564/11.8708 |
| GA | 0.10 | 0.736676/0.736734/0.736742 | 5.0073/8.1115/5.0712 | 5.6847/5.7549/5.6700 | 10.6920/13.8664/10.7412 |
| GA | 0.05 | 0.707613/0.736349/0.707457 | 4.6706/7.7048/4.7682 | 5.3245/5.4844/5.2051 | 9.9951/13.1892/9.9733 |
| Greedy | 0.30 | 0.736678/0.736671/0.736672 | 8.3639/8.0168/8.9938 | 5.2337/5.3002/5.3105 | 13.5975/13.3170/14.3042 |
| Greedy | 0.25 | 0.736632/0.736687/0.736683 | 8.2239/7.8717/8.3382 | 5.2478/5.0443/5.1163 | 13.4717/12.9160/13.4545 |
| Greedy | 0.20 | 0.736687/0.736653/0.736653 | 6.3148/8.0591/6.5357 | 5.0260/5.1889/5.2847 | 11.3408/13.2480/11.8204 |
| Greedy | 0.15 | 0.736823/0.736652/0.736688 | 5.9340/8.0968/6.2191 | 5.2227/5.3841/5.5232 | 11.1568/13.4809/11.7424 |
| Greedy | 0.10 | 0.736765/0.736650/0.736645 | 4.8670/8.1364/4.8795 | 5.5445/5.6874/5.3266 | 10.4115/13.8238/10.2061 |
| Greedy | 0.05 | 0.699603/0.736337/0.698834 | 4.7284/7.7413/4.9013 | 5.2906/5.3930/5.3874 | 10.0190/13.1343/10.2888 |

The full report additionally contains AP30/AP50/AP70, forward/post/total
p50/p90/p99, speedup, and five-run standard deviation/minimum/maximum for all
38 inventory entries.

## Production code and tests

- `search/integration/dual_gpu_fair_evaluation.py`
  - extracts formal per-frame forward/postprocess/total latency rows;
  - compacts completed evaluation state while preserving frame-order proof;
  - adds forward/postprocess/total mean and percentiles;
  - aggregates AP and latency over independent repeats.
- `scripts/run_lidar_pyramid_split_gpu_ablation_evaluation.py`
  - adds `--repeat-count`;
  - performs full repeated split-GPU scheduling;
  - writes local and consolidated per-frame CSVs;
  - validates exact row counts and writes a SHA256 manifest;
  - emits repeat-level and five-repeat mean CSV/JSON/Markdown reports.
- `tests/test_search_dual_gpu_fair_evaluation.py`
  - covers warmup exclusion/per-frame latency extraction;
  - covers repeated-result mean and same-repeat baseline association.

Validation command and result:

```bash
PYTHONPATH=. pytest -q \
  tests/test_search_dual_gpu_fair_evaluation.py \
  tests/test_search_evaluation_provider.py \
  tests/test_search_lidar_pyramid_prune_quant_ablation.py
# 14 passed

git diff --check
```

Reproduction command:

```bash
python scripts/run_lidar_pyramid_split_gpu_ablation_evaluation.py \
  --repeat-count 5
```

## Evidence

```text
outputs/h800_lidar_pyramid_split_gpu_ablation_repeat5_20260718_181347/
```

Important files:

```text
per_frame_latency.csv
per_frame_latency_manifest.json
split_gpu_repeat_results.csv
split_gpu_five_repeat_mean_results.csv
split_gpu_five_repeat_mean_comparisons.csv
split_gpu_five_repeat_report.json
split_gpu_five_repeat_report.md
```

The 339,910-row consolidated CSV, local per-evaluation CSVs, logs, and detailed
evaluation JSONs remain local and must not be committed. Only the small
manifest and aggregate reports are suitable for selective Git inclusion.

Completed checkpoint: 2026-07-19 11:53:13 +0800 CST

---
