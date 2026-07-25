# H800 V2X-ViT Transformer BOPS / precision audit

Branch: `audit/h800-v2xvit-transformer-bops-precision`
Worktree: `/home/lixingfeng/UniAD_examine/heal_compress_h800_v2xvit_bops_precision_audit`
Base: `0d58d34642a9fb7d20898e1e799ce1cfeb462344`

## 2026-07-25 22:23–23:55 CST

This audit paused the six-budget Greedy, formal GA and bulk TensorRT build. It
did not change the production BOPS definition, denominator, budgets, legal
search states, precision policy, Taylor proxy or historical labels.

Added `scripts/audit_v2xvit_transformer_bops_precision.py`. It loads the strict
V2X-ViT checkpoint, traces the first three fixed50 manifest frames, captures all
12 Attention and 3 FFN calls plus the actual einsum operand shapes, independently
recomputes standard and HGT attention MACs, enumerates the formal precision
groups, and reconciles existing B0/S32/JMIX/P8 ONNX/TensorRT evidence. Added
`tests/test_v2xvit_transformer_bops_precision_audit.py` for independent QKV,
QK, AV, O, FFN, relation-attention and operand-bit formulas.

Principal BOPS finding: production Greedy and GA share the same
`UnifiedBOPSProxy`, and the Transformer proxy is active. Standard window
attention, FFN and Shrinker marginal arithmetic reconcile. The HGT agent
relation path is not standard attention: its `relation_att` and `relation_msg`
einsums each include a learned `d_h × d_h` contraction. Production counts only
the standard linear-in-`d_h` QK/AV terms. On the representative shape this
omits 1,649,267,441,664 FP32 BOPS, a 1.3248% total-baseline under-count. HGT
attention marginal deltas are consequently low by about 18.9% for 32→28, 9.8%
for 16→12 and 4.5% for 8→4. No production fix was applied.

Principal precision finding: the formal space contains 104 groups (80 mutable,
12 fixed-FP32 QK, 12 protected residual/default-FP16). It discovers 12
LayerNorm modules without active filtering, but the production active weighted
module filter emits zero LayerNorm groups. In the maximal-P8 evidence, 12
Softmax INT8 genes realize FP32 typed ONNX, 12 AV INT8 genes realize FP16, and
3 FFN activation INT8 genes have no canonical inspector mapping. Weighted
canonical nodes have no fallback, but full-phenotype requested/realized is not
exact. No production fix was applied.

Audit gates: `bops_audit_passed=false`,
`precision_protection_audit_passed=false`, `formal_search_allowed=false`.
Reports are under
`/data/lxf/heal_data/outputs/h800_v2xvit_transformer_bops_precision_audit_20260725_222350`.
Targeted tests: 35 passed. Full pytest: 1007 passed. `compileall`, `py_compile`
and `git diff --check` passed. No Greedy, GA, full1789 or new TensorRT engine was
run/built.

--------------------------------------------------------------------------------
Timestamp: 2026-07-25T23:55:00+08:00
