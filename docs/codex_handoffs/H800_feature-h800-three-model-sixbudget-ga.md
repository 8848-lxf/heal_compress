# H800 three-model six-budget formal GA handoff

## 2026-07-25 19:18:00 +0800

Branch: `feature/h800-three-model-sixbudget-ga`

Worktree: `/home/lixingfeng/UniAD_examine/heal_compress_h800_three_model_sixbudget_ga`

Start commit: `0a7388a1f88f0ce51d48b96870ee6f7193c7c20c`

Run root: `/data/lxf/heal_data/outputs/h800_three_model_sixbudget_ga_20260725_184733`

This branch is isolated from `feature/heal-unified-search-h800`.  Only physical
GPU 2 and 3 are used; GPU 4--7 processes are external and have not received any
signal.

Implemented code and functions:

- Ported the common `StrictStage12V3Runner` adapter for CNN-family models and
  added an exact `formal_gen10` HEAL runner.  Configuration is one seed (0),
  population/offspring/survivor 64, ten evolution generations after generation
  0, and at most five new real Stage-2 candidates per generation.
- Added a real AttFusion family provider.  Runtime inventory proves its
  `ScaledDotProductAttention` has no trainable Q/K/V/O projection, so the model
  uses 21 dependency-closed CNN width domains and no artificial attention-dh or
  FFN locus.
- Added physical-GPU to logical-CUDA mapping for isolated worktrees whose child
  processes receive `CUDA_VISIBLE_DEVICES`.
- Added a six-budget Transformer Taylor/Greedy runner for V2X-ViT and CoBEVT.
  It collects sample-first Fisher/gate/AQ statistics, checks 8/16/32 prefix
  convergence and captures exact winners at 0.30/0.25/0.20/0.15/0.10/0.05.
- Fixed model-specific train-prefix provenance: fixed dataset indices and RNG
  seeds are shared, but CoBEVT records its own voxel counts instead of silently
  reusing V2X-ViT preprocessing shapes.
- Added a V2X-ViT six-budget formal GA entrypoint that reuses the validated
  V2X-ViT physical pruning, fresh train200, ONNX/QDQ, functional precision,
  TensorRT and fixed50 implementation.
- Added a CoBEVT provider and formal GA adapter.  It uses the unified
  CNN/attention-dh/FFN materializer and the common HEAL fixed-K deployment
  evaluator rather than a parallel GA implementation.
- Fixed the Greedy selector's size-report field bridge:
  `SizeProxy.R_size_vs_fp32` is exposed as the semantic
  `mixed_weight_retention` alias with one calculation; Taylor/BOPS and
  tie-break semantics are unchanged.

Completed evidence:

- CoBEVT inventory: 20 CNN + 6 independent attention-dh + 6 FFN domains.
- AttFusion inventory: 21 CNN domains, 0 trainable attention-dh, 0 FFN domains.
- AttFusion all-keep FP32 Stage-2 preflight: engine success, precision/QDQ
  accepted, fixed50 50 evaluated / 0 skipped, mAP `0.4810724280`.
- Full repository tests after changes: `1037 passed`, no failures.
- `compileall`, `py_compile`, and `git diff --check` pass.

Active tasks at this timestamp:

- GPU2: V2X-ViT train32 cache and complete six-budget Greedy trajectory.
- GPU2 queued: corrected CoBEVT train32 cache and six-budget Greedy trajectory.
- GPU3: AttFusion six-budget formal GA, resumed after the size-field alias fix.

No full1789 has been run.  No second seed has been started.

---

Timestamp separator: 2026-07-25T19:18:00+08:00

---

## 2026-07-25 19:25:00 +0800

- Commits `79b9901a` and `36f785dc` were pushed; remote divergence was verified
  as `0 0` after each push.
- Full pytest completed with `1037 passed`; the focused CoBEVT provider test
  now additionally verifies model-family detection, Transformer merge policy,
  and unified pruning strategy.
- CoBEVT Stage-2 now explicitly audits the original ONNX QK/Softmax/AV FP32
  contract, inserts typed FP32 protection casts after weighted Q/DQ insertion,
  and checks the realized TensorRT attention contract.  Missing mappings or a
  non-FP32 protected path fail the candidate without fallback.
- GPU2 queue is frozen as: V2X-ViT six-budget Greedy -> corrected CoBEVT
  six-budget Greedy -> V2X-ViT six-budget formal GA -> CoBEVT six-budget formal
  GA.  GPU3 independently runs AttFusion formal GA.  All formal searches use
  seed 0 and generations 1--10 exactly.

---

Timestamp separator: 2026-07-25T19:25:00+08:00

---
