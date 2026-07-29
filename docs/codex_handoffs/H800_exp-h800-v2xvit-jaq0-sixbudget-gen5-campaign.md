# H800 V2X-ViT J_AQ=0 six-budget gen5 campaign

## 2026-07-29T11:35:00+08:00

- Branch: `exp/h800-v2xvit-jaq0-sixbudget-gen5-campaign`.
- Base: `5826b594ba868afa8f607e4a78ca59c2046c1938`, including HGT/AV/merge precision closure and the two-tier 300/500 evaluation protocol.
- The formal contract is exactly five evolution generations; generation zero remains initialization only.
- `J_AQ` defaults to zero in proxy, Greedy, and GA fitness, while mixed deployment keeps real activation quantization and calibration.
- Cross-GPU Stage-2 is fail-closed. The physical GPU argument must exactly match `CUDA_VISIBLE_DEVICES`, and every candidate lifecycle stays on the assigned card.
- The prior `.05` result used ten generations and is provenance-only; this campaign rebuilds `.05` under the five-generation contract.

---

## 2026-07-29T15:10:00+08:00

- The confirmed fifth CNN-family target is `AttFusion`; it remains permanently bound to physical GPU6. V2X-ViT remains on physical GPU3.
- Generalized the V2X-ViT full-validation runners from the historical repeat-3/all-budget-only protocol to an explicit budget subset and repeat count while preserving the old defaults.
- The campaign invocation is now able to enforce five repetitions and five completed GA evolution generations for the `.05` result before the remaining five budgets are launched.
- Both Greedy and GA-final candidates receive P-only and Q-only controls. Full-validation summaries include AP@0.3/AP@0.5/AP@0.7/mAP plus p50/p90/p99.
- Added fail-closed `CUDA_VISIBLE_DEVICES == physical_gpu` validation so these follow-on evaluations cannot silently migrate to another card.
- Focused validation: `6 passed`; `py_compile` and `git diff --check` passed.

---
