# H800 CoBEVT GA precision/manifest closure

Branch: `fix/h800-cobevt-ga-precision-manifest`  
Worktree: `/home/lixingfeng/UniAD_examine/heal_compress_h800_cobevt_ga_precision_fix`  
Base: `ee1306c3986e6cd328eb0cc8feacb3bac88116ac`

--- 2026-07-28 00:30:31 CST ---

## Scope and isolation

- Prioritize CoBEVT under the latest `StrictStage12V3Runner` (single seed,
  ten generations, 300-frame Stage-2 screening and 500-frame generation-winner
  validation).
- Do not touch the active Pyramid worktree, PID, output tree, engine, manifest,
  or logs.
- DiscoNet/F-Cooper are deferred until CoBEVT completes; their shared 500/200
  evaluation-manifest sizing fix is already present in this base.

## Blocker distinction

- DiscoNet/F-Cooper previously failed before GA because their evaluation
  manifest had been created for 50/20 while the updated protocol requested
  500/200. This was an evaluation-protocol wiring failure.
- CoBEVT failed earlier in Stage-2 precision mapping with
  `heal_lidar_precision_profile_origin_mismatch`. Its unified Transformer space
  omitted nine non-prunable weighted runtime modules (PFN/deblocks/heads and
  followers) and mixed parameter-free Transformer boundaries into the weighted
  ONNX-origin equality check. This is a separate precision-schema bug.

## Code changes in progress

- `_formal_space` can now receive the complete runtime quantization inventory;
  CoBEVT passes all runtime groups instead of reconstructing precision groups
  only from prunable CNN domains.
- The weighted ONNX profile and functional Transformer profile are partitioned
  exhaustively. Missing weighted paths, missing declared functional paths,
  overlaps, and arbitrary unknown paths all fail closed.
- CoBEVT context now publishes precision units, Attention/FFN instances, and
  functional boundary paths to Stage-2.
- Functional ONNX mapping now understands CoBEVT graph names for QK, Softmax,
  AV, LayerNorm, Attention residual, FFN residual, and FFN2 activation input.
- Stage-2 now requires both Attention FP32-island acceptance and complete
  functional requested/realized acceptance.

## Verification

- Targeted tests: `63 passed`.
- `python -m compileall -q search scripts tests`: passed.
- `git diff --check`: passed.
- Next gate: build a fresh CoBEVT search context, prove complete precision
  ownership, then build/evaluate one exact Greedy anchor before formal GA.

--- 2026-07-28 00:36:00 CST ---

## Search-space preflight accepted

- Preflight root:
  `/data/lxf/heal_data/outputs/h800_cobevt_precision_closure_preflight_20260728_004000`.
- Runtime weighted modules: 53; unified weighted profile owners: 53;
  missing=0, unexpected=0.
- Unified domains: 32 total = 20 CNN + 6 independent Attention + 6 FFN.
- Precision groups: 93 total, including 53 mutable genes and 40 protected
  functional paths.
- Added a cached counterfactual Greedy replay with `J_AQ=0`. The formal path
  continues to use `J_struct + J_WQ + J_AQ`; the counterfactual performs zero
  forward/backward/export/build calls and is report-only. It compares action
  composition, parameter retention and precision counts at every budget to
  determine whether activation Taylor actually shifts selection toward
  structural pruning.
- Regression after this addition: 31 passed; compileall and diff check passed.
