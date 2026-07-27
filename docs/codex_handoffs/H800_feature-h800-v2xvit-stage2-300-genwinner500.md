# H800 V2X-ViT GA two-tier evaluation handoff

Branch: `feature/h800-v2xvit-stage2-300-genwinner500`

Base: `836a49a6e23156c55ef6e9fe2028670b810b38de`

## 2026-07-26 12:05 PDT

- Added the two-tier real-evaluation contract without changing the 10-generation GA scheduler:
  - every generation's Stage-2 Top-5 uses 300 validation frames after a 100-frame warmup;
  - the unique winner selected in each generation is re-evaluated on 500 validation frames after a 200-frame warmup;
  - the final budget winner is selected only from the fixed500 generation winners and the fixed500 Greedy anchor.
- Generation-winner validation reuses the exact screened TensorRT engine after phenotype hash, engine SHA256, and requested/realized precision checks. It does not repeat physical pruning, calibration, ONNX export, Q/DQ insertion, or engine build.
- Screening and winner-validation results use separate cache namespaces and carry explicit frame/warmup/protocol metadata.
- Audited the formal Taylor call chain. Structure gate, weight quantization, and activation quantization all use first plus empirical-Fisher second order:
  `abs(g * delta) + 0.5 * abs(E[g^2] * delta^2)`.
- Corrected the functional-gate diagnostic decomposition so first- and second-order totals are reported separately; the fitness formula itself was already correct.
- Tests: 46 targeted tests passed; compileall, py_compile, and `git diff --check` passed.

--- 2026-07-26T12:05:00-07:00 ---

## 2026-07-26 12:15 PDT

- Full regression suite passed: 1057 passed, 0 failed (82 warnings).
- Branch pushed and verified 0 ahead / 0 behind before the documentation-only follow-up.

--- 2026-07-26T12:15:00-07:00 ---

## 2026-07-27 09:30 CST

- Completed the V2X-ViT strict-FP32, six-budget Greedy/GA, and P/Q-only repeated full-validation evaluation requested for the final comparison.
  - Main evaluation root: `/data/lxf/heal_data/outputs/h800_v2xvit_sixbudget_greedy_ga_full1789_repeat3_20260726_170759`.
  - P/Q-only root: `/data/lxf/heal_data/outputs/h800_v2xvit_sixbudget_pq_only_full1789_repeat3_20260726_171505`.
  - Unified accounting: `/data/lxf/heal_data/outputs/h800_v2xvit_repeat3_complete_metrics_20260727_092230/v2xvit_complete_metrics.{csv,json}`.
  - Every control used 3 repeats of all 1,789 validation frames; the completed matrix contains 139,542 frame-evaluations and zero skipped frames.
- Main strict FP32 baseline: AP30/AP50/AP70/mAP = `0.781001/0.700153/0.519452/0.666869`; repeated full-validation forward p50 = `16.5951 ms` on GPU2.
- Greedy/GA findings:
  - budgets 0.30, 0.25, 0.20, and 0.15 retain baseline-equivalent mAP (`0.66737` to `0.66994`) while reaching `1.33x` to `1.74x` repeated full-validation forward speedup;
  - budget 0.10 loses about `0.033 mAP` while reaching about `2.05x` speedup;
  - budget 0.05 suffers severe structural collapse: Greedy/GA mAP `0.27264/0.28350`, despite `2.65x/2.70x` speedup.
- P/Q-only causal controls used each GA-final phenotype:
  - P-only reproduces the 0.10 degradation and 0.05 collapse (`mAP 0.63495/0.28758`);
  - Q-only remains close to baseline even at 0.05 (`mAP 0.66262`, retention `0.99351`), identifying extreme physical pruning as the dominant failure source.
- Built all 12 P/Q-only engines from fresh phenotype-specific artifacts. Requested/realized precision was exact for every engine with no conflict, unmapped major unit, or fallback.
- Fixed CoBEVT Stage-1 activation-Taylor mapping so fixed/protected Softmax groups are excluded from the mutable precision cache. Masked pre-Softmax `-inf` logits are no longer incorrectly treated as a non-finite post-Softmax activation perturbation. Targeted regression suite: 50 passed.
- Background formal GA status at this checkpoint:
  - AttFusion budget 0.30 completed generation 10 and entered generation-winner validation;
  - Pyramid budget 0.30 completed generation 2 of 5;
  - CoBEVT restarted after the Softmax mapping correction and passed the former activation-proxy blocker.

--- 2026-07-27T09:30:00+08:00 ---
