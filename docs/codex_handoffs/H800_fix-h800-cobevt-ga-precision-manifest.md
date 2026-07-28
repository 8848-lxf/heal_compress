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

--- 2026-07-28 02:49:52 CST ---

## First formal run audit and merge traversal fix

- Superseded run root:
  `/data/lxf/heal_data/outputs/h800_cobevt_latest_ga_precision_closure_20260728_004814`.
  Its 32-sample cache and deterministic Greedy reports completed, but the run
  was stopped before the first engine build because the adaptive merge ONNX
  traversal was computationally impractical. No external process was signaled.
- Main Greedy trace: 1,088 selected actions. The `J_AQ=0` counterfactual trace:
  973 selected actions. The selected path captured 0.30, 0.20, 0.15, 0.10 and
  0.05; 0.25 was not reached by this single path.
- Activation Taylor shifted all five comparable anchors toward more structural
  pruning. At 0.10, parameter retention changed from 0.380924 (`J_AQ=0`) to
  0.296593 (formal `J_AQ` enabled), while structure actions increased from 664
  to 886. At 0.05, retention changed from 0.324806 to 0.222411 and structure
  actions increased from 807 to 978.
- The first 0.30 physical checkpoint and FP32 ONNX exported successfully, so
  the earlier weighted/functional precision-origin mismatch was crossed.
- Runtime diagnosis on the 3,974-node CoBEVT ONNX found 384 merge/matmul
  candidates. The nearest-weighted-producer traversal recomputed shared
  upstream subgraphs without memoization. A read-only memoized replay took
  0.0164 seconds (5,742 recursive calls, 3,920 cached tensors); the production
  pass remained active for several minutes.
- Added deterministic memoization and explicit graph-cycle failure to both the
  adaptive and FP16 merge traversals. This changes traversal complexity only;
  it does not change BOPS, precision contracts, hashes, or search scoring.
- Regression: 13 targeted merge/CoBEVT precision tests passed; `git diff
  --check` passed.

--- 2026-07-28 03:02:43 CST ---

## Restore formal Greedy frontier and finite beam recovery

- The completed CoBEVT diagnostic path exposed a second integration
  regression: target 0.25 was not hit by the selected single path even though
  the audited generic Greedy engine still contained legal-neighbor frontier
  capture and finite beam recovery.
- Root cause: the CoBEVT `J_AQ=0` audit had wrapped the custom formal
  `cnn_stage12_v3.greedy_anchors` around two selected-only paths, bypassing the
  previously implemented frontier/recovery behavior.
- Restored, for both formal `J_AQ` and report-only `J_AQ=0` paths:
  - capture of already evaluated legal neighbors in a budget band;
  - bounded adjacent-action beam recovery from above-budget legal states;
  - physical phenotype deduplication through canonical hashes;
  - explicit recovery depth/evaluation/iteration accounting;
  - zero structure, precision and budget repair;
  - deterministic winner selection and separate capture-source provenance.
- Added resource-metric memoization; it does not change resource values or
  action ordering and prevents repeated exact BOPS/size evaluation of identical
  phenotypes across the primary/frontier/recovery paths.
- Added executable frontier/beam and same-seed determinism tests. Targeted
  verification: 20 passed; compileall and `git diff --check` passed.

--- 2026-07-28 13:09:15 CST ---

## CoBEVT Stage-2 precision closure repaired

- The superseded six-budget run
  `/data/lxf/heal_data/outputs/h800_cobevt_latest_ga_precision_closure_v4_20260728_030437`
  produced exact Greedy anchors but correctly failed closed before generation
  0. The three independent failures were: residual Add fixed-A16 ownership was
  not transferred to the ONNX node, Softmax inspector treated a boolean mask as
  a numeric precision input, and FFN bias Add/QDQ boundary ownership differed
  between calibration and export.
- Activation boundary resolution now treats a unique Linear/Gemm bias Add as
  the weighted output boundary, and train200 scale extraction uses the exact
  same stop-before-merge semantics as Q/DQ insertion. The entropy worker also
  asserts the consumed sample count and records requested/processed/skipped
  counts explicitly.
- Fixed residual functional precision is now mapped onto exact ONNX Add nodes.
  Q/K (or fused QKV) projection compute remains searchable at FP16/INT8, while
  its realized output is explicitly promoted to FP32 before the protected QK
  matmul. This prevents TensorRT from folding DQ into an INT8 QK tactic.
- TensorRT Softmax auditing now excludes boolean/integer control inputs but
  still requires every numeric Softmax input and output to be FP32.
- A new versioned CoBEVT pre-search cache freezes the complete search space,
  domain-local retained-coordinate rankings, gate scores, WQ/AQ/Fisher state,
  and parameter slices. Cache lookup binds source manifests and domain
  membership but deliberately does not require recomputing floating ranking
  order before loading it. Resume mismatches fail closed with field-level
  provenance.
- Added `scripts/validate_cobevt_precision_closure.py` for a build-only,
  provenance-bound closure check. Historical width/precision genes may be
  rebound to a newly frozen coordinate ranking only as an explicitly labelled
  diagnostic; old coordinate archives are never reused by the new formal run.

## Representative real-engine acceptance

- Validation root:
  `/data/lxf/heal_data/outputs/h800_cobevt_precision_closure_repair_20260727_214351`.
- Representative 0.10 phenotype kept the historical width state and precision
  map, materialized the exact expected parameter shapes, then completed fresh
  train200 calibration (200 requested, 200 processed, 0 skipped), strongly
  typed ONNX/QDQ export, TensorRT build, and engine inspection.
- Candidate hash:
  `d9884dc94bc36b86241e2762bda0306cdd9d66a8a101aaa9c1b908f577d6cc80`.
  Engine SHA256:
  `fc247b712afc12c28c7257683455afae49ff7f5982f05c060e591eb318f45c2c`.
- Requested/realized weighted precision is exact: 7 requested INT8 loci and 7
  realized INT8 loci. All six QK and all six Softmax numeric paths are realized
  FP32; functional precision audit passed with zero failed rows and no silent
  fallback.
- The original inspector failure was a real violation: before the QKV-output
  boundary fix, TensorRT selected an INT8-input QK GEMM tactic. The accepted
  engine preserves INT8 projection compute but presents true FP32 operands to
  QK.

## Verification and isolation

- Targeted precision/Stage-2/formal-package regression: 126 passed.
- Full repository regression: 1077 passed, 0 failed.
- `compileall`, explicit `py_compile`, and `git diff --check`: passed.
- CoBEVT validation used GPU7 only. The active Pyramid process on GPU2 (Python
  PID 713036) and its output tree were not touched or signalled.
- Next action: commit and push this closure, then start a completely new
  CoBEVT formal run so the v2 ranking/proxy cache and all Stage-2 artifacts are
  generated under the repaired contract. The v4 run remains read-only failure
  evidence and will not be resumed.
