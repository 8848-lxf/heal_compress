# H800 CNN J_AQ=0 six-budget gen5 campaign

## 2026-07-29T11:20:00+08:00

- Branch: `exp/h800-cnn-jaq0-sixbudget-gen5-campaign`.
- Base: `cefe210243ed87955f094503e06d5ce3f30f9af5`.
- Scope: Pyramid, DiscoNet, F-Cooper, and (after user name confirmation) AttFusion.
- Frozen GA protocol: one seed (`0`), five evolution generations, population/offspring/survivor 64, Stage-2 new quota 5.
- Frozen evaluation protocol: Stage-2 Top-5 warmup 100 + fixed300; each generation winner warmup 200 + fixed500.
- Search fitness defaults to `J_struct + J_WQ + 0 * J_AQ`. Activation quantization remains enabled in deployed mixed-precision engines.
- The existing runner previously defaulted to `J_AQ=1` and the experiment branch restricted `J_AQ=0` to a single Pyramid `.05` run. This campaign removes that experiment-only restriction while retaining an explicit audit field in every output.
- No GPU search was launched before the default and protocol tests were updated.

---

## 2026-07-29T19:10:00+08:00

- DiscoNet `.05` Q-only produced a valid TensorRT engine with all 11 requested
  INT8 weighted loci realized exactly, but the auxiliary fusion-island audit
  could not map ONNX `/Expand_2`: Myelin had eliminated that non-arithmetic
  broadcast node. This was an inspector proof gap, not a precision fallback.
- Added a fail-closed elided-Expand proof. It accepts the missing TensorRT layer
  only when the Q/DQ ONNX graph proves that every Expand consumer is an explicit
  Cast to the required auxiliary dtype and every downstream canonical consumer
  is inspector-mapped at that same dtype without INT8 realization.
- The failed DiscoNet artifact now independently verifies with zero issues:
  `/Expand_2 -> Cast(FP16) -> /Mul_9`, with `/Mul_9` realized FP16. No other
  missing operator receives an exemption.
- Focused validation: `34 passed`; direct validation of the produced engine
  returned `passed=true`, `issues=[]`; `py_compile` and `git diff --check`
  passed. The successful P-only artifact remains untouched; the campaign will
  resume in a new output root rather than overwrite the failed diagnostic root.

---

## 2026-07-29T15:35:00+08:00

- User confirmed the fifth target model is `AttFusion`; its permanent campaign assignment is physical GPU6.
- The final full-validation contract changed from five to three serial repetitions. Search remains five GA generations with Stage-2 warmup100/fixed300 and generation-winner warmup200/fixed500.
- Generalized the Pyramid and generic CNN joint/P-only/Q-only full1789 runners to record an explicit repeat count; their campaign default is now three.
- Existing evaluations that had already passed three repetitions were not interrupted. Any fourth/fifth repetitions already produced remain diagnostic only and will not enter the formal three-repeat aggregation.
- Focused validation: `7 passed`; `py_compile` and `git diff --check` passed.
- Added provenance-locked prefix reaggregation so completed five-repeat P+Q runs can formally use identical repeat indices 0/1/2 without rerunning or overwriting the source. Pyramid, DiscoNet, and F-Cooper `.05` repeat3 reports passed.
- Added one fail-closed follow-on orchestrator for DiscoNet/GPU1, F-Cooper/GPU2, and AttFusion/GPU6. It explicitly requests repeat3 for all new joint and P/Q evaluations and rejects any model/GPU mismatch.
- The revised focused suite now reports `8 passed`.
- The first DiscoNet/F-Cooper repeat3 P/Q build attempts stopped before engine construction because their YAML plugin path was relative to the isolated worktree, where the compiled plugin is intentionally absent. Added an explicit, file-validated `--plugin` override to the P/Q builder and wired the campaign to the same accepted absolute plugin used by formal search. No fallback or copied binary was introduced.
- Focused validation after the plugin fix: `9 passed`.

---
