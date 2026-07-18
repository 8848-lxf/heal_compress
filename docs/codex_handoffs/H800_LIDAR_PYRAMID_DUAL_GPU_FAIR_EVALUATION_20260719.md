# H800 lidar_pyramid dual-GPU fair evaluation handoff

## Outcome

The existing strict-FP32 engine and the twelve accepted P+Q winner engines
(six GA and six greedy budgets) were re-evaluated on physical H800 GPU 0 and
GPU 1. The execution was globally serial: GPU 0 completed all thirteen items
before GPU 1 started. No ONNX export, calibration, Q/DQ rewrite, plugin build,
or TensorRT engine build was invoked.

All 26 evaluations passed the fixed protocol:

- fixedK = 29696;
- the same 1789-frame manifest and frame order;
- warmup = 200, followed by iterator reset;
- latency rounds = 3;
- DataLoader workers = 8;
- CUDA NMS/IoU postprocess required and verified;
- 1789 evaluated, 0 skipped;
- one fresh evaluation worker process per engine;
- worker exit followed by explicit CUDA allocator cleanup;
- nvidia-smi memory returned within 128 MiB of the pre-evaluation level after
  every item (26/26 passed).

The evaluation-only manifest verifies all source engine SHA256 values against
their deployment manifests and requires six acceptance reports to pass before
loading an engine. The engine size, mtime, and SHA256 are checked again after
evaluation. The new output directories contain no `.plan`, `.engine`, `.onnx`,
or `.pth` file. Reported engine build count is exactly zero.

## Branch and Git state

Work is on `feature/heal-unified-search-h800`. The previous local HEAD before
this round was `6927ce3`; it was three commits ahead of
`origin/feature/heal-unified-search-h800`. Earlier HTTPS push attempts failed
because the server's VS Code Git credential-helper socket was stale and GitHub
received no valid write credential. This was an authentication failure, not a
branch selection or source-code failure. No remote URL, credential store, or
system Git configuration was modified.

The earlier prune/quant contribution ablation visibly built engines because it
had to create missing P-only and Q-only variants and the corrected greedy 0.30
P+Q artifact. That was intentional for the ablation. This round uses only the
already accepted P+Q engines and has no build path.

## Full-validation result summary

Speedup on each row uses the fresh strict-FP32 p50 measured on the same GPU.
AP30/AP50/AP70 and p90/p99 are in `fair_evaluation_report.md` and the CSV.

| Method | Budget | Actual BOPS | GPU0 mAP | GPU0 p50 | GPU0 speedup | GPU1 mAP | GPU1 p50 | GPU1 speedup |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| FP32 | - | 1.000000 | 0.736783 | 6.9771 | 1.0000x | 0.736601 | 6.9403 | 1.0000x |
| GA | 0.30 | 0.304223 | 0.736648 | 6.5642 | 1.0629x | 0.736707 | 6.5602 | 1.0579x |
| GA | 0.25 | 0.254517 | 0.736722 | 4.8387 | 1.4419x | 0.736852 | 4.8784 | 1.4226x |
| GA | 0.20 | 0.204300 | 0.736670 | 4.7608 | 1.4655x | 0.736737 | 4.7430 | 1.4633x |
| GA | 0.15 | 0.154526 | 0.736627 | 4.4680 | 1.5616x | 0.736689 | 4.4808 | 1.5489x |
| GA | 0.10 | 0.104957 | 0.736549 | 3.4021 | 2.0508x | 0.736646 | 3.4119 | 2.0341x |
| GA | 0.05 | 0.054807 | 0.707517 | 3.1475 | 2.2167x | 0.707856 | 3.1202 | 2.2243x |
| Greedy | 0.30 | 0.301922 | 0.736691 | 6.5240 | 1.0695x | 0.736704 | 6.5338 | 1.0622x |
| Greedy | 0.25 | 0.249080 | 0.736699 | 6.5165 | 1.0707x | 0.736662 | 6.4765 | 1.0716x |
| Greedy | 0.20 | 0.196774 | 0.736609 | 4.6894 | 1.4878x | 0.736619 | 4.7596 | 1.4582x |
| Greedy | 0.15 | 0.149413 | 0.736845 | 4.3365 | 1.6089x | 0.736770 | 4.3640 | 1.5903x |
| Greedy | 0.10 | 0.098478 | 0.736833 | 3.2479 | 2.1482x | 0.736832 | 3.2596 | 2.1292x |
| Greedy | 0.05 | 0.049994 | 0.699676 | 3.1540 | 2.2122x | 0.699905 | 3.2339 | 2.1461x |

Cross-GPU consistency is good: the largest absolute mAP difference is
`0.000339514` (GA 0.05), and the largest absolute p50 difference is
`0.079973 ms` (greedy 0.05). Both GPUs independently reproduce the 0.05 BOPS
accuracy cliff. Budgets 0.10 through 0.30 remain effectively FP32-accurate.

## Code and tests

New production support:

- `search/integration/dual_gpu_fair_evaluation.py`
  - immutable source-engine and acceptance validation;
  - fixed FP32 -> GA -> greedy inventory/order;
  - one fresh evaluation subprocess per item;
  - post-evaluation cache cleanup and nvidia-smi memory audit;
  - same-GPU FP32 speedup calculation;
  - cross-GPU consistency calculation;
  - explicit guard against build artifacts in evaluation outputs.
- `scripts/run_lidar_pyramid_dual_gpu_fair_evaluation.py`
  - requires GPU order `0 1`;
  - rejects a requested GPU if preflight memory/utilization indicates it is
    occupied;
  - runs the two complete sequences globally serially;
  - emits JSON, CSV, and Markdown reports.
- `tests/test_search_dual_gpu_fair_evaluation.py`
  - inventory/order, source hash/acceptance, protocol/build-output guard,
    same-GPU speedup, and cross-GPU comparison tests.

Validation command:

```bash
python -m pytest -q tests/test_search_dual_gpu_fair_evaluation.py
python scripts/run_lidar_pyramid_dual_gpu_fair_evaluation.py
git diff --check
```

The focused test suite passes: `4 passed`.

## Artifacts

Machine-readable evidence root:

```text
outputs/h800_lidar_pyramid_dual_gpu_eval_only_20260718_154151/
```

Important small reports:

```text
evaluation_only_manifest.json
fair_evaluation_results.csv
cross_gpu_consistency.csv
fair_evaluation_report.json
fair_evaluation_report.md
```

Per-evaluation JSON/log files and the 50 MB rolling `completed_results.json`
remain ignored and must not be committed. Source engines remain read-only in
their original GA, greedy, and ablation artifact directories.

Completed checkpoint: 2026-07-19 07:17:50 +0800 CST

---

## Git delivery

Implementation, tests, small machine-readable reports, and this handoff were
committed on `feature/heal-unified-search-h800`:

```text
f445476 feat(search): add dual-GPU fair engine evaluation
```

The branch is locally four commits ahead of the remote immediately after this
implementation commit. A push is attempted separately; if authentication is
still unavailable, `f445476` is the commit that must be fetched or transferred.

Completed checkpoint: 2026-07-19 07:21:25 +0800 CST

---
