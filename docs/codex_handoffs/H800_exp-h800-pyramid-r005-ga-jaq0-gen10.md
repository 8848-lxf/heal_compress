# H800 Pyramid R_BOPS=0.05 J_AQ=0 controlled GA ablation

Branch: `exp/h800-pyramid-r005-ga-jaq0-gen10`  
Worktree: `/home/lixingfeng/UniAD_examine/heal_compress_h800_pyramid_r005_ga_jaq0`

---

Timestamp: 2026-07-28 05:10 CST

## Development progress

- Added a strict `activation_taylor_fitness_weight` switch to the shared
  Stage-1 evaluator and CNN Greedy action scoring. Only values `0` and `1` are
  accepted.
- The `J_AQ=0` mode retains raw activation Taylor statistics as diagnostics,
  but uses `J_total = J_struct_gate + J_WQ + 0 * J_AQ` for Greedy and GA.
- Physical W8A8 activation quantization, calibration, ONNX/QDQ, TensorRT
  realization checks, fixed300 Stage-2 screening and fixed500 generation-winner
  validation remain unchanged.
- Restricted direct ten-generation ablation execution to Pyramid, seed 0 and
  the single R_BOPS=0.05 budget. This is the strongest matched diagnostic for
  the observed low-budget pruning bias while bounding disk and engine cost.
- The experiment must start only after the main Pyramid J_AQ=1 continuation
  finishes. It writes to a new output root and cannot reuse the main run's
  candidate fitness or final winner.
- Targeted regression: 33 passed; compileall, py_compile and git diff check
  passed.

---

Timestamp: 2026-07-29 08:59:40 CST

## Five-generation launch and continuation closure

- Changed the controlled Pyramid `J_AQ=0` experiment to support a fresh,
  single-budget five-generation run while retaining the exact same
  `StrictStage12V3Runner`, population 64, offspring 64, Stage-2 quota 5 and
  seed 0 contracts.
- Generalized the generation-5 freeze/verification code from the historical
  hard-coded six-budget set to the actual requested budget labels.  This lets
  the single `R_BOPS=0.05` ablation be resumed safely.
- Bound continuation to the original activation-Taylor fitness weight.  A
  request that changes `0.0` to `1.0` fails closed instead of mixing proxy
  contracts.
- Added `reports/continuation_ready.json` after a successful generation-5
  run.  It records generation 0--5 summary hashes and the supported resume
  arguments.  The supported extension is `--resume --generations 10`: it
  deterministically replays generations 1--5 using immutable Stage-2 caches,
  verifies every summary hash, then executes new generations 6--10.
- Regression after the change: 28 targeted tests passed; compileall and
  `git diff --check` passed.

---

Timestamp: 2026-07-28 18:24 CST

## Physical/logical GPU isolation correction before restart

- The first five-generation launch was stopped during the Greedy-anchor
  TensorRT calibration phase after read-only process inspection showed that
  the main PyTorch process was correctly isolated on physical GPU6, but its
  ModelOpt calibration worker had been launched on physical GPU0.
- Root cause: the formal launcher correctly converted physical GPU6 to the
  process-local CUDA ordinal 0, while `select_gpu()` subsequently stored that
  local ordinal as `physical_gpu_id=0`.  ModelOpt subprocess construction then
  reused the incorrect physical identifier in `CUDA_VISIBLE_DEVICES`.
- Fixed the shared runtime GPU selection helper so an isolated local ordinal
  is mapped back through the existing `CUDA_VISIBLE_DEVICES` list.  PyTorch
  retains `runtime_device=cuda:0`, while calibration, TensorRT and evaluation
  subprocesses retain `physical_gpu_id=6`.
- Added single-visible-device and multiple-visible-device regression tests.
  A live dry check under `CUDA_VISIBLE_DEVICES=6` now resolves to
  `physical_gpu_id=6` and `runtime_device=cuda:0`.
- The invalid run root
  `/data/lxf/heal_data/outputs/h800_pyramid_r005_ga_jaq0_gen5_20260729_090028`
  is retained as failed provenance and is not resumed or reused.  Restart must
  use a new run root and verify the first ModelOpt worker environment before
  allowing the five-generation search to continue.
- Regression after the isolation fix: 30 targeted tests passed; compileall and
  `git diff --check` passed.

---

Timestamp: 2026-07-28 18:39 CST

## Stage-2 evaluation subprocess physical-GPU closure

- The first restart proved the context fix (`physical_gpu_id=6`,
  `runtime_device=cuda:0`) and completed the proxy/Greedy phase, but the manual
  first-child gate found a second, evaluation-specific path that translated
  `runtime_device=cuda:0` back into `CUDA_VISIBLE_DEVICES=0`.
- Stopped only this run's main/evaluation PIDs before the baseline evaluation
  processed a valid result.  The restart root
  `/data/lxf/heal_data/outputs/h800_pyramid_r005_ga_jaq0_gen5_20260728_182628`
  is retained as failed provenance and must not be resumed.
- Extended `evaluate_engine_modelopt()` with an explicit physical GPU argument
  and wired `LidarPyramidRealEvaluator` to pass the context's physical GPU.
  The worker request now records physical `cuda:6`, while its process-local
  execution device remains `cuda:0`.
- Added a provider regression test that asserts both the worker request and
  ModelOpt subprocess environment use physical GPU6 when the parent runtime
  device is logical `cuda:0`.
- Regression after closing both context and evaluation paths: 34 targeted
  tests passed; compileall and `git diff --check` passed.
