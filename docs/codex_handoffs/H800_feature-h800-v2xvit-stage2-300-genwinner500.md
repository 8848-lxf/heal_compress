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

## 2026-07-27 10:20 CST

- Audited the apparent V2X-ViT parameter-pruning jump between the final GA winners at `R_BOPS=0.10` and `0.05` using the exact frozen genotypes and the production `unified-bops-v2-hgt-relation-closure` evaluator.
- Resource change:
  - `R_BOPS`: `0.104583 -> 0.054525`, a further `47.86%` reduction relative to the 0.10 candidate;
  - physical parameters: `5,919,081 -> 3,486,757`, removing another `2,432,324` parameters and moving the baseline pruning rate from `56.00%` to `74.08%`;
  - mutable precision counts only change from `FP32/FP16/INT8=1/12/40` to `0/9/44`, so precision changes alone cannot explain the halving of BOPS.
- The 0.10 and 0.05 winners are independent GA phenotypes rather than nested subnet points. Their net width differences are nevertheless strongly concentrated in:
  - shrinker `80 -> 28`;
  - stage-2 CNN `[72,92,132,84,116,108,156,188,28] -> [52,68,92,52,88,72,108,124,28]`;
  - Attention layer 0 `[16,16,32,16] -> [4,8,8,4]`, layer 1 `[20,16,20,12] -> [12,8,4,4]`, and layer 2 `[16,12,20,8] -> [4,4,4,4]` for `[agent,w4,w8,w16]`;
  - FFN changes only modestly overall: `[252,256,248] -> [256,240,164]`.
- Production BOPS reconciliation attributes the 0.10-to-0.05 decrease as follows:
  - Transformer `63.54%` of the total decrease; CNN `36.46%`;
  - Attention excluding FFN/Softmax about `62.7%`; shrinker alone `22.18%`;
  - fixed-FP32 QK and AV each contribute `16.13%` of the total decrease because their cost can only be reduced through `d_h`, not projection INT8.
- Structural pruning is unavoidable at 0.05 under the current deployment contract: the original-width maximal-buildable INT8 floor is `R_BOPS=0.293668`. Reaching the upper budget edge `0.055` therefore requires at least an additional `81.27%` reduction in BOPS relative to that unpruned precision floor. This does not mathematically prove that exactly `74.08%` parameter pruning is minimal; BOPS and parameter count are not one-to-one.
- Cross-model current Greedy-anchor comparison at 0.05, using each model's current resource proxy:
  - V2X-ViT `74.08%` physical parameter pruning;
  - F-Cooper `78.42%` proxy parameter pruning;
  - Pyramid `78.01%` proxy parameter pruning;
  - DiscoNet `87.79%` proxy parameter pruning.
  The previously cited Pyramid value near `44%` matches its current 0.20 point (`44.53%`), not its 0.05 anchor.
- Search status at this checkpoint:
  - V2X-ViT six-budget Greedy/GA and repeat-3 full1789/P-Q controls are complete;
  - AttFusion completed budget 0.30 generation 10 and is at budget 0.10 generation 3; 0.25/0.20/0.15 were not captured by its selected Greedy trajectory;
  - Pyramid completed budget 0.30 generation 5 and is at budget 0.25 generation 1;
  - CoBEVT completed the pre-search/Greedy phase but all six formal budgets failed closed before GA because the Stage-2 deployment mapper reported precision-profile origin mismatch between generic CNN origins and Transformer functional groups. No formal CoBEVT generation was executed.

--- 2026-07-27T10:20:00+08:00 ---
