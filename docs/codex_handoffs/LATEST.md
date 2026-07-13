# Joint Search Handoff - 2026-07-12 New Contract Validated

This handoff is based on current source and real artifacts. The active validated run is:

```text
/home/lixingfeng/UniAD_examine/heal_compress/tests/outputs/lidar_pyramid_joint_search_final_20260712_130033
```

The older run below is reference-only under the new contract:

```text
/home/lixingfeng/UniAD_examine/heal_compress/tests/outputs/lidar_pyramid_joint_search_final_20260712_060922
```

That older run used old BOPS targets `0.25, 0.226666..., 0.203333..., 0.18`; all old repaired Top-5 entries violate the new `T_BOPS + 0.005` gates, and its old final winner was only 300-frame selected. Do not reuse its rankings, round winners, or final winner as conclusions.

## Inputs And Environment

- Checkpoint: `/home/lixingfeng/UniAD_examine/Auto_Search/original_models/dairv2s/LiDAROnly/lidar_pyramid/net_epoch_bestval_at17.pth`
- Model config: `/home/lixingfeng/UniAD_examine/Auto_Search/original_models/dairv2s/LiDAROnly/lidar_pyramid/config.yaml`
- TensorRT root: `/home/lixingfeng/UniAD_examine/HEAL/prune_model/TensorRT-10.9_x86_cu118`
- Model/search env: `univ2x-opt`
- TensorRT/build/eval env: `modelopt`
- Stage-1/Stage-2 GPU used by current run: physical GPU `0`

## Source Changes In This Continuation

- Updated new contract BOPS schedule in `search/configs/lidar_pyramid_joint_search_final.yaml`:
  - round 0: `0.230`
  - round 1: `0.213`
  - round 2: `0.197`
  - round 3: `0.180`
- Stage-1 CPU/GPU proxy now defaults to raw Fisher/SQNR terms for F1; normalization is recorded but not applied unless explicitly enabled.
- Added Stage-2 admission/signature helpers:
  - `search/stage2/admission.py`
  - `search/stage2/realized_bops.py`
- Repaired Top-K selection now filters by repaired BOPS budget and requires pruning or legalized INT8.
- Stage-2 score now records:
  - `R_BOPS_realized`
  - `realized_int8_layer_count`
  - `control_only`
  - `realized_precision_profile_hash`
  - `deployment_signature`
- Round winner selection excludes:
  - control-only candidates
  - realized BOPS over budget
  - missing/duplicate deployment signatures
- Orchestration no longer lets the legacy lowest-F2 fallback overwrite a gated Stage-2 round winner after `stage2_top5_results.json` has been written.
- `stage2_only` output round is inferred from the candidate config path.
- Full-validation evaluation path supports `warmup=200`, reset-after-warmup, `1789` measured frames, and fail-closed skips.
- Final selection now has a hard gate for `R_BOPS_realized <= 0.185`, compression/non-control, full 1789 frames, zero skips, and unique deployment signature.

No `tracer/**` or `quantization/**` files were modified.

## Original Precision Audit

Precision audit artifacts remain under the old audit directory:

```text
tests/outputs/lidar_pyramid_joint_search_final_20260712_060922/precision_audit/
```

Audited facts:

- Checkpoint model tensors: FP32.
- Loaded model parameters: FP32.
- Loaded model buffers: FP32 plus INT64 counters/metadata.
- Training compute precision: `unverified`; do not claim full-FP32 training.
- PyTorch eval: FP32 eager for sampled weighted modules, autocast disabled.
- ONNX initializers: FLOAT.
- TensorRT strict FP32/FP16/max legal INT8 baselines were audited.

## New Stage-1 Results

Run:

```text
tests/outputs/lidar_pyramid_joint_search_final_20260712_130033
```

Stage-1 was rerun with the new BOPS targets. Each round has 5 generation CSVs and repaired Top-5 manifests:

```text
round_000/repaired_top5_manifest.json
round_001/repaired_top5_manifest.json
round_002/repaired_top5_manifest.json
round_003/repaired_top5_manifest.json
```

All 20 repaired Top-5 candidates satisfy the new repaired BOPS gate and have pruning or legalized INT8. The old 060922 Top-5 candidates do not.

## Stage-2 Results

All 20 candidates completed the real Stage-2 chain:

```text
repaired phenotype -> SamplingPruningRequest -> physical plan -> materialize
-> pruned ONNX -> explicit Q/DQ -> TensorRT engine -> 300-frame AP/latency
```

Per-candidate artifacts exist under:

```text
round_XXX/stage2/<candidate_hash>/
```

Required artifacts include:

- `sampling_pruning_request.json`
- `physical_pruning_plan.json`
- `physical_plan_validation.json`
- `physical_structure_snapshot.json`
- `physical_validation.json`
- `physical_widths.csv`
- `pruned_checkpoint.pth`
- `pruned_fp32.onnx`
- `pruned_qdq.onnx`
- `engine.plan`
- `evaluation_300.json`
- `realized_bops_report.json`
- `deployment_manifest.json`
- `stage2_score.json`
- `artifact_hashes.json`

All 20 deployment signatures are globally unique.

Round winners by 300-frame Stage-2 F2:

```text
round 0: a411caf1789751da5ac21cfab78ae4ad14ab78120bb6e02f512b2df874034126
  F2 10.270941474292373
  mAP_300 0.4780304255635659
  R_BOPS_realized 0.13959132454818288

round 1: bda30b4342fbb0d1b0d26aa5fc2fae4ebaa28ec477e2f9f1334b69783dd3951a
  F2 25.431658209255968
  mAP_300 0.09970952244454347
  R_BOPS_realized 0.1621464487696243

round 2: 29476bd3a0f7fbd5fcaf698b6a3683559140526c91da5974e337336ab8fda201
  F2 23.04712403150854
  mAP_300 0.1591603314171903
  R_BOPS_realized 0.14617908357690051

round 3: 955f42b08de8df1421356f348f47726d3262cd021aeacb39a1b177a611db2e9e
  F2 16.794411478585566
  mAP_300 0.3155633261627735
  R_BOPS_realized 0.14229095302157307
```

## Full Validation

Full validation was run serially for:

- original strict FP32
- original strict FP16
- the four round winners

Protocol:

- warmup executions: `200`
- reset iterator after warmup: `true`
- evaluated frames: `1789`
- latency measured frames: `1789`
- skipped frames: `0`

Baseline full-validation references:

```text
original_strict_fp32:
  mAP 0.7368049474621197
  forward_p50_ms 5.856374278664589

original_strict_fp16:
  mAP 0.7363040973845818
  forward_p50_ms 2.8072725981473923
```

Final full-validation winner:

```text
candidate: a411caf1789751da5ac21cfab78ae4ad14ab78120bb6e02f512b2df874034126
round: 0
F2_full: 9.96913954537204
mAP_full: 0.4924083305657139
forward_p50_ms_full: 2.712876225511233
R_BOPS_realized: 0.13959132454818288
deployment_signature: 0fa4b6c338f3187c71bcad0ce1caee6cbb0353ee9b3a35797832768706017eab
```

Final artifacts:

```text
final_selection/round_winners.csv
final_selection/round_winners.json
final_selection/full_validation_results.csv
final_selection/full_validation_results.json
final_selection/final_best_manifest.json
final_selection/final_best_pruned_model.pth
final_selection/final_best_pruned.onnx
final_selection/final_best_qdq.onnx
final_selection/final_best.engine.plan
final_selection/final_best_full_validation.json
```

## Resume Verification

Resume was run after final selection with `--skip-baselines`.

Tracked artifacts:

```text
132
```

Resume result:

```text
status: ok
changed_count: 0
missing_count: 0
misses: 0
resume_without_rebuild_verified: true
```

Report:

```text
tests/outputs/lidar_pyramid_joint_search_final_20260712_130033/resume_cache_report.json
```

## Verification

Commands completed successfully:

```bash
pytest -q tests/test_search_candidate_codec.py tests/test_search_cache_reuse.py tests/test_search_topk_diversity.py tests/test_search_tool_adapters.py tests/test_two_stage_joint_search.py tests/test_search_baseline_engines.py tests/test_search_baseline_precision_validation.py tests/test_search_full_val_manifest.py tests/test_search_second_order_fisher.py tests/test_search_local_domain_ranking.py tests/test_search_channel_alignment.py tests/test_search_grouped_conv_actions.py tests/test_search_independent_group_topk.py tests/test_search_quantization_groups.py tests/test_search_runtime_shapes.py tests/test_search_bops_proxy.py tests/test_search_bops_budget.py tests/test_search_gpu_batch_proxy.py tests/test_search_gpu_batch_integration.py tests/test_search_large_population.py tests/test_search_int8_realization.py tests/test_search_final_contract.py tests/test_search_stage2_final_selection.py tests/test_search_contract_v2.py tests/test_search_stage2_round_results.py tests/test_search_stage2_candidate_artifacts.py tests/test_search_stage2_physical_validation.py
```

Result after final handoff update and the orchestration winner overwrite regression test:

```text
107 passed
```

Also completed:

```bash
python -m py_compile $(find search -name '*.py') pruning/importance/second_order_fisher.py pruning/api.py pruning/importance/__init__.py
git diff --check
```

## Completion Flags

Only `true` where proven by current code and artifacts:

```text
original_precision_audit_complete: true
checkpoint_precision_verified: true
training_compute_precision_verified: false
pytorch_eval_precision_verified: true
trt_baseline_precision_verified: true

repaired_top5_manifest_complete: true
repaired_mask_to_request_verified: true
repaired_mask_to_physical_plan_verified: true
group_keep_map_frozen_verified: true
candidate0_materialization_complete: true
candidate0_physical_validation_complete: true
candidate0_pruned_onnx_complete: true
candidate0_explicit_qdq_complete: true
candidate0_trt_engine_complete: true
candidate0_300_frame_complete: true

round0_top5_stage2_complete: true
round0_winner_complete: true
round1_complete: true
round2_complete: true
round3_complete: true
all_round_winners_complete: true
final_full_validation_complete: true
final_winner_complete: true

cross_round_global_dedup_verified: true
physical_cache_verified: true
onnx_cache_verified: true
deployment_cache_verified: true
evaluation_cache_verified: true
resume_without_rebuild_verified: true
end_to_end_joint_search_validated: true
```

## Cautions

- Training compute precision is still `unverified`.
- Final winner is valid under the requested hard gates, but its full-validation mAP drop is large relative to original FP32.
- The new run uses strict new BOPS targets and realized BOPS gates; do not mix old 060922 rankings into final claims.
