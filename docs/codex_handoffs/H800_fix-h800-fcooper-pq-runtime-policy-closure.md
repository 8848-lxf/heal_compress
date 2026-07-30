# H800 F-Cooper P/Q runtime-policy closure

Branch: `fix/h800-fcooper-pq-runtime-policy-closure`

Worktree: `/home/lixingfeng/UniAD_examine/heal_compress_h800_fcooper_pq_runtime_policy_fix`

Base commit: `295dedca425a2a5d24717a20b2030b8d65eb9b3b`

## 2026-07-30 — P/Q decomposition deployment-policy repair

### Symptom

The formal P+Q engines were accepted, while the derived F-Cooper Q-only engine failed the legacy FP16 fusion-island audit at `/GridSample`, `/Mul_6`, `/Where_2`, and `/ReduceMax`. The failed run stopped after one of eighteen unique derived deployments:

`/data/lxf/heal_data/outputs/h800_fcooper_remaining5_jaq0_gen5_pq_controls_repeat3_20260729_161500`

### Root cause

The formal search serialized `search_space_policy=heal_runtime_graph_v1`, which creates the runtime precision relation and uses the adaptive runtime merge precision contract. The P/Q decomposition context builder omitted this field, so `build_heal_lidar_baseline_context` silently selected `legacy_family_static_dependency_closure_v1`. The derived Q-only engine was therefore validated with a different ONNX/QDQ and fusion-island policy than its source P+Q engine.

This was a provenance/context propagation defect, not evidence that Q-only quantization was unsupported.

### Code changes

- `search/ablation/heal_lidar_prune_quant.py`
  - validates and propagates the source search-space policy for every formal/legacy candidate;
  - requires GA and Greedy sources to use the same policy;
  - includes the policy in the deployment configuration signature so incompatible engines cannot deduplicate or reuse each other.
- `scripts/run_heal_lidar_prune_quant_ablation.py`
  - writes manifest schema `heal-lidar-family-prune-quant-ablation-v2` with an explicit policy;
  - recreates the P/Q build context with that exact policy;
  - permits old-manifest inference only from an unambiguous serialized precision-policy version and otherwise fails closed.
- `tests/test_search_heal_lidar_prune_quant_ablation.py`
  - covers formal and legacy policy propagation, build-all context binding, old-manifest inference, and mixed-policy rejection.

### Verification

- Targeted regression: `53 passed`.
- Full pytest: `1063 passed`.
- `compileall`, `py_compile`, and `git diff --check`: passed.
- Representative GA 0.30 Q-only FP16 engine:
  - runtime policy `heal_runtime_graph_v1`;
  - 28/28 canonical mappings;
  - conflict/unmapped/fallback = 0/0/0;
  - engine SHA256 `ae905134c2ca778ff164587b8fedcd46ddbd8e32c4c29090aee77cca53406f99`.
- Representative GA 0.10 Q-only INT8 engine:
  - requested/realized INT8 = 4/4;
  - train200 = 200/200, zero skipped;
  - conflict/unmapped/fallback = 0/0/0;
  - engine SHA256 `38adc177a708faf13366f91a4eabd72b44c1ccd16488ba89c2d290629534b712`.
- Full derived deployment build:
  - 30 logical configurations;
  - 18 unique deployments;
  - 18/18 accepted, zero failures;
  - inventory root `/data/lxf/heal_data/outputs/h800_fcooper_remaining5_jaq0_gen5_pq_runtime_policy_fix_v2_20260730_102354`.

The 1789-frame, repeat-3, same-GPU serial decomposition evaluation is running under:

`/data/lxf/heal_data/outputs/h800_fcooper_remaining5_jaq0_gen5_pq_runtime_policy_fix_v2_full1789_repeat3_20260730_102354`

---

Timestamp: `2026-07-30T10:58:14+08:00`

## 2026-07-30 — Full 1789-frame repeat-3 acceptance

The fixed decomposition completed 96/96 same-GPU serial evaluation items: 2 methods × 16 engines × 3 repeats. Every item evaluated 1789 frames, skipped zero frames, and preserved its source engine hash.

Evaluation root:

`/data/lxf/heal_data/outputs/h800_fcooper_remaining5_jaq0_gen5_pq_runtime_policy_fix_v2_full1789_repeat3_20260730_102354`

| method | R_BOPS | FP32 mAP | P+Q mAP | P-only mAP | Q-only mAP | P+Q p50 ms | P+Q speedup |
|---|---:|---:|---:|---:|---:|---:|---:|
| GA | 0.30 | 0.559168 | 0.558792 | 0.558871 | 0.559193 | 4.2109 | 1.481x |
| GA | 0.25 | 0.559168 | 0.558794 | 0.558912 | 0.559048 | 3.2767 | 1.903x |
| GA | 0.20 | 0.559168 | 0.558798 | 0.558813 | 0.559098 | 2.9479 | 2.115x |
| GA | 0.15 | 0.559168 | 0.558428 | 0.558984 | 0.558823 | 2.5626 | 2.433x |
| GA | 0.10 | 0.559168 | 0.553388 | 0.558996 | 0.465151 | 2.3505 | 2.653x |
| Greedy | 0.30 | 0.559133 | 0.558805 | 0.558860 | 0.559226 | 4.2471 | 1.465x |
| Greedy | 0.25 | 0.559133 | 0.558913 | 0.558906 | 0.559262 | 3.7006 | 1.681x |
| Greedy | 0.20 | 0.559133 | 0.558820 | 0.558895 | 0.559230 | 3.5114 | 1.772x |
| Greedy | 0.15 | 0.559133 | 0.558576 | 0.558851 | 0.558808 | 3.1617 | 1.968x |
| Greedy | 0.10 | 0.559133 | 0.558461 | 0.558890 | 0.467501 | 2.3197 | 2.682x |

The 0.10 Q-only control is now a valid deployment result, not a build failure. Its accuracy collapse is reproducible and arises when the jointly selected precision map is applied to the original unpruned structure. The corresponding P+Q candidates retain substantially higher accuracy, demonstrating a strong structure–quantization/calibration interaction at this budget.

Build acceptance totals:

- 18/18 unique derived engines accepted;
- requested/realized INT8 totals 10/10;
- precision mismatches 0;
- unresolved canonical layers 0;
- 30 logical P-only/Q-only/P+Q rows finalized.

---

Timestamp: `2026-07-30T12:45:29+08:00`
