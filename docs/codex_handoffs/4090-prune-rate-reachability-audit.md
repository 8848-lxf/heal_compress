# 4090 lidar_pyramid prune-rate reachability audit

历史约 0.6 是否被物理模型参数量证实：YES

历史约 0.7 是否能够物理导出：YES

历史约 0.7 的主要失败类型：ACCURACY_COLLAPSE

当前 0.217361 是否是模型固有结构上限：NO

当前 0.217361 的直接成因：global importance walk 逐 unit 前进时，只有所有 grouped physical groups 同时落入同一合法宽度才记录候选；排序偏斜导致大量合法宽度组合未被枚举。旧 manifest 的 `0.217361` 是 planner 投影/枚举上限，不是 inventory 或模型结构上限。

最大统一口径物理参数剪枝率：`0.892599918276838`（当前 24-domain inventory、当前保护、当前 `domain_cap=0.8`、当前 minimum width/alignment/grouped 规则；物理导出及真实数据 10/10 smoke 通过，但精度崩塌）

## Verdict

This audit selects conclusion **A: current planner coverage/implementation
error**. Historical plans at approximately 0.6 and 0.7 were deterministically
replayed from their original `global_physical_prune_plan.json` files without
current repair. Their replayed parameter counts exactly match the saved
physical model objects. Independently enumerating each current domain's legal
retained widths produces a current-rule mask with 586,919 parameters, or
`0.892599918276838` physical full-model pruning. It materializes and runs both
synthetic and real validation inputs.

The current and historical constraint sets are not identical. In particular,
the historical layer-2 grouped convolutions retained 12 channels per physical
group, while the current allowlist is `[4, 8, 16, ...]`. That difference makes
the old masks illegal under current rules, but it does not explain the
`0.217361` ceiling: the stricter current rules themselves still permit and
physically realize `0.892600`.

## Entry audit

- Branch: `feature/heal-compress-h800-sync-4090`.
- Entry local and remote HEAD: `3f33afb680b2c90e202b534bd99022534f8c94d8`.
- Entry remote delta: `0 0`.
- Required H800 ancestor `b862b3d8ad061bd12580776226c75f564918298d`: present.
- Checkpoint SHA256: `d20a01079cc09b1f313a37bd5ccd5174390a93f5c2b02dc9a942929e1227caca`.
- Model-config SHA256: `52d798abf67cf0a91cfed4738740f9695130f0e1d7869f008a82314ffb87a5f8`.
- GPUs 4/5/6/7 were idle at entry. Only GPU4 was used for the final PyTorch
  10-frame smoke. No process remained afterward.
- Pre-existing joint-Taylor Stage-1 WIP was preserved, not overwritten, in
  `stash@{0}: codex-wip-joint-taylor-stage1-before-reachability-audit-20260715`.
- No GA, Stage-A, Stage-B, QDQ, TensorRT, calibration, or engine build ran.

Full command/process/GPU evidence is in
`outputs/20260715_150636_prune_rate_reachability_audit/entry_audit.json`.

## Metric definitions

The audit keeps requested, predicted, physical-full, physical-prunable,
atomic-unit, and channel ratios separate:

```text
R_physical_full = (P0 - PC) / P0
R_physical_prunable = (P_prunable,0 - P_prunable,C) / P_prunable,0
R_predicted = (P0_predicted - PC_predicted) / P0_predicted
R_atomic = pruned atomic units / searchable atomic units
R_channel = pruned output channels / searchable output channels
```

`R_physical_full` is the primary reachability metric and always comes from
`sum(p.numel())` on a materialized model. The historical dependency-slice
union happens to cover all 5,464,791 model parameters, so the recomputed
historical physical-prunable rates equal their full-model rates in this model;
that equality is observed, not assumed.

## Historical evidence

All three saved model objects were loaded directly. In addition, their exact
historical physical plans were reapplied one-shot to a fresh original model.
Each replay count matched its saved model count and finite forward passed.

| requested | saved/replayed params | physical full rate | atomic/channel ratio | AP30 | AP50 | AP70 | mAP | 500-frame result |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 0.50 | 2,723,599 | 0.501609668 | 0.205507604 | 0.810266 | 0.765233 | 0.584357 | 0.719952 | export/forward/eval pass |
| 0.60 | 2,181,995 | 0.600717575 | 0.227428415 | 0.807678 | 0.759965 | 0.574054 | 0.713899 | export/forward/eval pass |
| 0.70 | 1,632,503 | 0.701268905 | 0.334292369 | 0.084481 | 0.079492 | 0.039484 | 0.067819 | accuracy collapse |

The historical baseline mAP was `0.718853`. At 0.70 the model was exported,
reloaded, forwarded, and evaluated for 500 frames; its mAP drop was `0.651034`.
Shape-invariant reports passed and the failure was accuracy, not structure.

Historical lineage limitations are explicit:

- the historical run did not persist its source git commit;
- it used the deterministic first 500 validation frames but did not persist a
  frame-ID manifest hash;
- historical AP03 was not saved. AP70 above is recovered from the historical
  definition `mAP=(AP30+AP50+AP70)/3`.

## Current inventory and coverage

| quantity | element count |
|---|---:|
| `P_total` / `P_trainable` | 5,464,791 |
| current protected-root parameter union | 14,423 |
| current selected atomic-slice raw sum | 9,366,208 |
| current selected global slice union | 5,463,232 |
| repeated/overlapping raw elements | 3,902,976 |
| unmapped unprotected auxiliary elements | 896 |
| missing selected unit/parameter slices | 0 |
| maximum removable union under current constraints | 4,877,872 |

The raw sum exceeds the model parameter count because dependency closures
overlap. It is not a globally unique parameter count. `unit_parameter_overlap.csv`
records the physical parameter element restrictions and overlap for every row.
For grouped Conv input slices, the audit restricts local axis-1 positions to
the owning physical group's output rows; it does not apply a local index to
all groups.

The historical report had 32 root-named domains; the current trace exposes 24
selected scopes. Several historical Conv1/Conv2 roots now map to one current
coupled closure, so the root-name difference is not itself missing coverage.
All historical-plan mappings used for the legality audit were exact
dependency-member slice mappings with high confidence.

## Planner versus independent maximum

| result | units pruned | atomic ratio | predicted full | physical full | physical prunable | max domain ratio | export/forward |
|---|---:|---:|---:|---:|---:|---:|---|
| observed planner manifest before fix | 320 | 0.046296296 | 0.217360920 | 0.217360920 | 0.217422947 | 0.718750 | pass/pass |
| independent legal-width solver | 4,616 | 0.667824074 | 0.892599918 | 0.892599918 | 0.892854633 | 0.796875 | pass/pass |
| importance-ranked planner maximum after fix | 4,616 | 0.667824074 | 0.892599918 | 0.892599918 | 0.892854633 | 0.796875 | pass/pass |

The independent method enumerates every legal retained width per domain and
maximizes removals without reusing the planner's global importance walk. Dense
domains obey alignment 4 and minimum width. Grouped domains preserve group
count and choose a common allowlisted width independently within each physical
group. The resulting model has 586,919 physical parameters.

After repair, virtual prediction, global slice union, and physical model count
agree exactly. Before the grouped virtual-shape fix, 13 grouped modules
overcounted retained parameters by 118,656 because local weight axis-1 indices
were subtracted as global input channels.

## Historical masks under current rules

No historical mask was repaired before diagnosis. The main rejection is a
real rule difference:

- eight `pyramid_backbone.resnet.layer2.*.conv2` domains prune 4 channels in
  each of 32 physical groups, leaving 12 per group;
- historical grouped surgery accepted 16 -> 12;
- current legal widths are 4, 8, 16, 32, ... and therefore reject 12.

| old mask | grouped-rule rejected entries | protected entries | current-rule result |
|---|---:|---:|---|
| 0.50 | 1,024 | 5 | reject without repair |
| 0.60 | 1,024 | 38 | reject without repair |
| 0.70 | 1,024 | 50 | reject without repair |

This proves both facts simultaneously: the old masks are not current-rule
candidates, and they were valid physical models under the historical rules.

## Constraint ablation

All deltas below are relative to the observed pre-fix `0.217361` result.
Planning-only rows are not labeled as physical evidence.

| experiment | single change | max predicted full | max physical full | delta | status |
|---|---|---:|---:|---:|---|
| A0 | observed pre-fix planner, full current constraints | 0.217361 | 0.217361 | 0 | physical pass |
| A1 | remove only per-domain cap | 0.938148 | not run | +0.720787 | planning only |
| A2 | restore old protected set only | INCONCLUSIVE | INCONCLUSIVE | n/a | current protected atoms lack replayable closure; retrace forbidden |
| A3 | old minimum retained ratio 0.4, keep current cap | 0.732581 | not run | +0.515220 | planning only |
| A4 | historical alignment 4 | 0.892600 | not run | +0.675239 | no change; current dense alignment is already 4 |
| A5 | fill missing mappings | 0.892600 | 0.892600 | +0.675239 | zero missing selected slices |
| A6 | global slice-union count | 0.892600 | 0.892600 | +0.675239 | physical pass |
| A7 | independent legal-width maximizer | 0.892600 | 0.892600 | +0.675239 | physical pass |
| A8 | observed historical 32-domain rules | 0.701269 | 0.701269 | +0.483908 | physical/eval pass, accuracy collapse |
| A9 | remove grouped legality only | 0.911223 | not run | +0.693862 | unsafe planning upper bound |
| A10 | remove dense alignment only | 0.894941 | not run | +0.677580 | planning only |
| A11 | remove current minimum width only | 0.892600 | not run | +0.675239 | no change |

The cap excludes 248,912 additional removable parameters relative to A1. A
0.4 minimum-retained ratio excludes 874,472 relative to the current independent
maximum. Neither explains the pre-fix ceiling. A2 cannot be isolated without
recreating protected dependency closures, which would violate this audit's
no-tracer-change rule.

The direct current-constraint coverage decomposition is: domain cap 248,912
elements, grouped legality 101,772 elements, dense alignment 12,794 elements,
and current minimum width 0 elements. A9 is deliberately not replayed because
its grouped shapes are structurally unsafe; it is only an upper-bound count.

## PyTorch real-data smoke

The final fixed manifest is validation dataset indices 0-9 on physical GPU4,
UUID `GPU-166702d7-bf30-18e0-83ff-83b316d37c0e`. All candidates completed
10/10 with zero skips using `pruning.eval.prune_and_eval.evaluate_one_model`.

| candidate | physical full rate | AP30 | AP50 | AP70 | mAP | result |
|---|---:|---:|---:|---:|---:|---|
| pre-fix planner max | 0.217361 | 0.833157 | 0.833157 | 0.741179 | 0.802498 | 10/10, 0 skip |
| fixed importance-ranked max | 0.892600 | 0.000044 | 0.000044 | 0.000012 | 0.000033 | 10/10, 0 skip; accuracy collapse |
| independent max | 0.892600 | 0.002912 | 0.000377 | 0.000010 | 0.001100 | 10/10, 0 skip; accuracy collapse |

These 10-frame values are structural smoke evidence, not formal accuracy
estimates. A new full 1,789-frame evaluation was not run: the historical
500-frame evidence already resolves the contradiction, and the current
extreme masks visibly collapse accuracy. No AP gate or search decision is made
from these extreme structures.

## Minimal fixes

- `search/stage1/conditional_repair.py`: unequal raw grouped requests now
  project conservatively to the largest common legal keep width, never adding
  pruning beyond any physical group's request.
- `search/anchors/joint_taylor_sweep.py`: explicitly enumerate the maximum
  legal state and avoid repeated parameter counting for duplicate repaired
  masks. This prevents importance-order skew from defining reachability.
- `search/proxy/virtual_shape_resolver.py`: correctly recover grouped logical
  input width from the coupled equal-width contract and local per-group axes.
- `search/audits/prune_rate_reachability.py` and
  `tools/audit_prune_rate_reachability.py`: add denominator-safe metrics,
  slice-union accounting, independent width solving, historical no-repair
  diagnostics, replay, ablation, and evidence generation.

No tracer, dependency graph, ChannelResolver, physical pruning core, candidate
gene semantics, QDQ, merge, or TensorRT code changed.

## Verification and artifacts

- 11 new reachability/accounting/replay assertions pass.
- 68 focused and related anchor/grouped/physical/search tests pass.
- Historical 0.50/0.60/0.70 plan replay parameter counts match saved physical
  artifacts exactly.
- Post-fix independent and planner prediction-to-physical error is zero.
- Modified Python files compile and `git diff --check` passes.
- `ruff` is unavailable in `univ2x-opt`; no package installation was attempted.

Primary evidence directory:

`outputs/20260715_150636_prune_rate_reachability_audit/`

It contains every requested artifact plus
`historical_unit_mapping.csv`, `parameter_count_reconciliation.csv`,
`pytorch_smoke_manifest.json`, and per-candidate smoke summaries. Outputs and
model artifacts remain untracked.

## Reproduction

```bash
cd /home/lixingfeng/UniAD_examine/heal_compress

conda run -n univ2x-opt python tools/audit_prune_rate_reachability.py \
  --smoke-gpu 4 \
  --smoke-frames 10

conda run -n univ2x-opt pytest -q \
  tests/test_anchor_parameter_accounting.py \
  tests/test_anchor_max_reachability.py \
  tests/test_historical_mask_replay.py \
  tests/test_prune_rate_metric_consistency.py

conda run -n univ2x-opt python -m py_compile \
  search/audits/prune_rate_reachability.py \
  search/anchors/joint_taylor_sweep.py \
  search/proxy/virtual_shape_resolver.py \
  search/stage1/conditional_repair.py \
  tools/audit_prune_rate_reachability.py

git diff --check
```

Final execution state:

```text
CONCLUSION=A_CURRENT_PLANNER_IMPLEMENTATION_ERROR
HISTORICAL_060_PHYSICAL_PROVEN=true
HISTORICAL_070_PHYSICAL_EXPORT=true
HISTORICAL_070_FAILURE=ACCURACY_COLLAPSE
CURRENT_0217361_MODEL_INHERENT_LIMIT=false
CURRENT_CONSTRAINT_MAX_PHYSICAL_FULL_RATE=0.892599918276838
PARAMETER_COUNT_BUG_FOUND=true
PARAMETER_COUNT_BUG_REMAINING_AFTER_FIX=false
GA_STARTED=false
STAGE_A_STARTED=false
STAGE_B_STARTED=false
TENSORRT_STARTED=false
```

--- Round 19 completed: 2026-07-16 06:00:51 CST ---
