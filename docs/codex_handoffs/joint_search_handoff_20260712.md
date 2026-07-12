# Joint Search Handoff - 2026-07-12

This is an engineering handoff for the next Codex session. It is based on the current workspace, current source tree, and the real artifacts under `tests/outputs/lidar_pyramid_joint_search_final_20260712_060922`.

## 1. 项目与目标

- Repository absolute path: `/home/lixingfeng/UniAD_examine/heal_compress`
- Current branch: `feature/pruning-quant-toolkit-cleanup`
- Checkpoint: `/home/lixingfeng/UniAD_examine/Auto_Search/original_models/dairv2s/LiDAROnly/lidar_pyramid/net_epoch_bestval_at17.pth`
- Model config: `/home/lixingfeng/UniAD_examine/Auto_Search/original_models/dairv2s/LiDAROnly/lidar_pyramid/config.yaml`
- Model: HEAL / OpenCOOD, DAIR-V2X, LiDAROnly, `lidar_pyramid`
- TensorRT root: `/home/lixingfeng/UniAD_examine/HEAL/prune_model/TensorRT-10.9_x86_cu118`
- TensorRT plugin path recorded by context: `/home/lixingfeng/UniAD_examine/heal_compress/quantization/plugins/pointpillar_scatter_trt/build/libpointpillar_scatter_trt.so`
- Environment split:
  - `univ2x-opt`: model loading, data, tracing, Fisher, pruning, Stage-1 proxy.
  - `modelopt`: TensorRT, CUDA plugin, engine build and engine execution subprocesses.
- Current final algorithm goal: complete a two-stage multi-round GA search for coupled-channel structured pruning plus existing coupled quantization groups with FP32/FP16/INT8 decisions, then deploy Top-K repaired candidates through physical pruning, ONNX, explicit Q/DQ, TensorRT, and fixed-manifest AP/latency evaluation.

## 2. 用户最终确认的算法契约

- Pruning gene is a real coupled-channel keep mask, not a coarse stage/block variable.
- `keep_mask = 1`, `prune_mask = 0`.
- Quantization gene is an existing coupled quantization group decision: `FP32`, `FP16`, or `INT8`.
- Stage-1 objective:

```text
F1 =
alpha * R_Fisher
+ beta * L_SQNR
+ gamma * R_Size_vs_FP32
+ delta * P_BOPS
```

- `R_Fisher` counts only pruned importance where `mask=0`.
- all-keep Fisher loss must be `0`.
- BOPS uses full `weight_bits * activation_bits`.
- BOPS reference is the original strict FP32 model.
- BOPS soft penalty:

```text
P_BOPS = max(0, R_BOPS / T_BOPS - 1)^2
```

- Outer-round `T_BOPS` tightens from `0.25` to `0.18`.
- Dense Conv/ConvTranspose/Linear repair before Stage-2 floors retained width to a multiple of 4.
- Grouped-conv safe per-group widths:

```text
{4, 8, 16, 32, 64, 128, 256, 512}
```

- Grouped conv requires equal retained count in each group, but retained positions may differ by group.
- Do not rerun `shared_local_mean` or `independent_group_topk` to override the GA mask.
- First-order Taylor is only used during repair for extra monotonic pruning to satisfy alignment/safe width.
- Repair only allows `1 -> 0`; it must never restore `0 -> 1`.
- Repaired phenotype must be rescored with the same four-term Stage-1 objective and deduplicated.
- The repaired unique Top-5 with smallest repaired F1 enters Stage-2.
- Stage-2 uses a fixed 300-frame AP/latency manifest.
- Each outer round selects one winner.
- Final winner is selected from all round winners.

## 3. 已完成的代码实现

Files below are current relevant implementation files. Most `search/**` files are untracked in Git because this search framework was added in the working tree.

### Core candidate and hashing

- `search/candidate.py`
  - Defines `CandidateGenotype`, `CandidatePhenotype`, `PrecisionDecision`.
  - Enforces precision normalization and sorted phenotype identity.
  - Connected to CLI path through canonicalization and Stage-1/Stage-2 orchestration.

- `search/canonicalization.py`
  - Defines `SearchSpaceSpec`.
  - Added `pruning_unit_metadata` for coupled-channel keep-mask generation and grouped repair.
  - `canonicalize_candidate()` now carries repaired metadata such as `group_keep_map_by_scope` and `group_prune_map_by_scope` into the phenotype.
  - Connected to formal CLI path.

- `search/hashing.py`
  - Provides candidate, search, physical, deployment, and eval hash helpers.
  - Used by Stage-1 proxy cache, artifact identity, and Stage-2 evaluator.

### CLI and orchestration

- `search/cli.py`
  - Formal CLI entry point.
  - Added overrides:
    - `--initial-population-size`
    - `--offspring-size`
    - `--skip-baselines`
  - `--skip-baselines` is for Stage-1 smoke/debug only; do not use it for final full search.
  - Connected to formal CLI path.

- `search/orchestration/lidar_pyramid_search.py`
  - Main real `lidar_pyramid` two-stage runner.
  - Builds real model context, runtime shapes, Fisher statistics, normalization, proxy evaluator, and Stage-2 evaluator.
  - Added support for `pruning.gene_type: coupled_channel_keep_mask`.
  - Writes `local_pruning_domains.json`.
  - Writes `stage1_objective_config.json`.
  - Applies outer-round BOPS target.
  - Uses `evaluate_batch()` in GA when backend is CUDA batched.
  - Runs raw candidate repair, repaired rescoring, repaired phenotype dedup, and Top-5 selection.
  - Writes `seen_raw_genotypes.jsonl` and `seen_repaired_phenotypes.jsonl`.
  - Added `resume_cache_report.json` code path, but this was added after the last resume run and is not present in the current run artifact.
  - Connected to formal CLI path.

- `search/orchestration/two_stage_search.py`
  - Lightweight runner for unit tests/dry-like flows.
  - Not the real `lidar_pyramid` path.

### Stage-1 GA and GPU proxy

- `search/ga/engine.py`
  - GA main loop supports `batch_evaluator`.
  - Formal path no longer loops over scalar `evaluate(candidate)` when CUDA batched backend is active.
  - Connected to formal CLI path.

- `search/ga/initialization.py`
  - Seeds FP32, FP16, forced INT8 and compressed seeds.
  - Connected to formal CLI path.

- `search/ga/immigrants.py`
  - Added grouped-aware random mask generation so grouped scopes keep at least 4 per group where possible.
  - This happens during raw genotype generation, not during Stage-1 repair.
  - Connected to formal CLI path.

- `search/ga/mutation.py`
  - Mutation calls grouped-aware seed repair to avoid generating unrecoverable grouped masks.
  - Connected to formal CLI path.

- `search/ga/crossover.py`
  - Crossover calls grouped-aware seed repair to keep grouped domains repairable.
  - Connected to formal CLI path.

- `search/stage1/proxy_evaluator.py`
  - `evaluate_batch()` performs canonicalization, cache lookup, miss collection, GPU batch scoring, scatter, and cache write.
  - Tracks scalar/batch/cache/GPU counters.
  - Guard: `gpu_proxy_required_but_not_active`.
  - Last GPU stats are preserved when a later cache-only batch has zero CUDA event time.
  - Connected to formal CLI path.

- `search/proxy/gpu_batch_proxy.py`
  - CUDA batched proxy scorer.
  - Encodes:
    - `pruning_choice_tensor`: `[B, num_pruning_units]`
    - `precision_choice_tensor`: `[B, num_precision_groups]`
  - Computes Fisher ratio, SQNR over retained weights, virtual channel resolution, Size vs FP32, BOPS vs FP32, soft BOPS penalty, F1.
  - Uses static tables on proxy GPU.
  - Connected to formal CLI path.

- `search/proxy/batch_channel_resolver.py`
  - Batched virtual width, params and MAC resolver.
  - Uses tensor ops for `C_in_after`, `C_out_after`, `groups_after`, `params_after`, and `MACs_after`.
  - Connected to formal CLI path.

- `search/proxy/fisher_proxy.py`
  - Fisher direction fixed:
    - pruned cost only for `mask=0`.
    - default returns ratio over all unit importance.
    - all-keep returns `0`.
  - Connected to scalar CPU reference and normalization.

- `search/proxy/sqnr_proxy.py`
  - SQNR uses retained mask and ignores pruned weights.
  - Connected to scalar CPU reference.

- `search/proxy/size_proxy.py`
  - Reports `R_size_vs_fp32` and `R_size_vs_fp16_deploy`.
  - Stage-1 objective now uses FP32 reference.

- `search/proxy/bops_proxy.py`
  - Reports Wbits x Abits BOPS.
  - Stage-1 objective now uses `R_bops_vs_fp32`.

- `search/proxy/objective.py`
  - Stage-1 objective changed to:
    - Fisher
    - SQNR
    - Size vs FP32
    - BOPS soft penalty only
  - Added `bops_soft_penalty()`.
  - Added `bops_target_for_outer_round()`.
  - Connected to scalar CPU reference and formal CLI.

- `search/proxy/parameter_slice_resolver.py`
  - Fixed raw grouped-conv input slice coordinates.
  - Absolute grouped input indices are remapped to local grouped weight input coordinates where needed.
  - This fixed a real CUDA `index out of bounds` during 64-candidate smoke.

### Pruning search and repair

- `search/pruning_space/mask_repair.py`
  - New search-layer repair utilities.
  - `dense_floor_repair()`: floors retained dense width to alignment using only `1 -> 0`.
  - `grouped_equal_count_floor_repair()`: floors grouped retained count to safe set and freezes group maps.
  - Unit-tested and used by formal orchestration.

- `search/stage1/repair_selection.py`
  - New repaired Top-K selector.
  - Repairs raw candidates, rescoring repaired phenotypes, deduplicates, and selects Top-K by repaired F1.
  - Uses batch rescore when available.
  - Connected to formal CLI path.

- `search/adapters/pruning_adapter.py`
  - When phenotype contains raw grouped units, Stage-2 request now reads repaired `group_keep_map_by_scope` and `group_prune_map_by_scope`.
  - Prevents fallback to default grouped top-k.
  - Connected to Stage-2 evaluator path, but Stage-2 has not been run for current Top-5.

- `search/stage2/lidar_pyramid_real_evaluator.py`
  - Stage-2 deployment path exists: physical materialization, ONNX export, Q/DQ, TensorRT build, eval.
  - Fixed dispatch so legal action ids use action catalog, while raw coupled-channel ids use `FormalPruningAdapter.request_from_phenotype()`.
  - Code entry exists, but current run has not executed Top-5 Stage-2.

- `search/stage2/objective.py`
  - Added `tau_ap` normalization for AP loss.
  - Stage-2 score now supports AP-priority normalized objective.

### Quantization groups

- `search/quantization_space/types.py`
- `search/quantization_space/group_builder.py`
- `search/quantization_space/legalizer.py`
- `search/quantization_space/codec.py`
  - Use existing coupled precision groups.
  - Precision genes are group ids, not synthetic per-module ids.
  - Unit-tested.
  - Connected to real context and Stage-1/Stage-2 profile expansion.

### Configs

- `search/configs/lidar_pyramid_joint_search_final.yaml`
  - Final intended config:
    - coupled-channel keep mask
    - FP32 Size/BOPS reference
    - CUDA batched proxy
    - 1024 initial, 512 active, 512 offspring
    - 4 outer rounds
    - 5 generations per round
    - Top-5 Stage-2
    - 300-frame Stage-2
  - File exists and is the recommended starting config.

- `search/configs/lidar_pyramid_joint_search_large_population.yaml`
  - Previous large-population config from earlier phase.
  - It still reflects earlier FP16-reference settings and is not the final contract.

### Tests added or updated

- `tests/test_search_final_contract.py`
  - Covers keep/prune mask semantics, Fisher direction, FP32 references, soft BOPS, outer-round target, dense repair, grouped repair, repaired Top-K, Stage-2 AP normalization, grouped input coordinate remap, grouped-aware immigrants/mutation, Stage-2 request group maps.

- Other relevant test files currently passing:
  - `tests/test_search_gpu_batch_integration.py`
  - `tests/test_search_gpu_batch_proxy.py`
  - `tests/test_search_bops_budget.py`
  - `tests/test_search_bops_proxy.py`
  - `tests/test_search_grouped_conv_actions.py`
  - `tests/test_search_independent_group_topk.py`
  - `tests/test_search_quantization_groups.py`
  - `tests/test_search_large_population.py`
  - and others listed in Section 6.

## 4. 剪枝器修改边界

Modified `pruning/` files:

- `pruning/api.py`
- `pruning/importance/__init__.py`
- `pruning/importance/second_order_fisher.py`

Purpose:

- These changes are only for SECOND_ORDER_FISHER importance scoring and formal `score_pruning_units` entry support.

Explicit boundaries:

- `tracer/` was not modified.
- `quantization/` was not modified.
- Pruning dependency tracing was not modified.
- Pruning physical materialization was not modified.
- Pruning legalizer was not modified.
- Pruning snapshot/hash/structure validation was not modified.
- Grouped-conv physical pruning rules were not modified.

Boundary command result:

```bash
git diff --name-only | grep -E '^(tracer|quantization)/' || true
# no output

git diff --name-only | grep -E '^pruning/' || true
pruning/api.py
pruning/importance/__init__.py
pruning/importance/second_order_fisher.py
```

## 5. 当前真实运行结果

Current run directory:

```text
tests/outputs/lidar_pyramid_joint_search_final_20260712_060922
```

Important files:

- `tests/outputs/lidar_pyramid_joint_search_final_20260712_060922/context_report.json`
- `tests/outputs/lidar_pyramid_joint_search_final_20260712_060922/run_manifest.json`
- `tests/outputs/lidar_pyramid_joint_search_final_20260712_060922/round_000/stage1_objective_config.json`
- `tests/outputs/lidar_pyramid_joint_search_final_20260712_060922/round_000/generation_000.csv`
- `tests/outputs/lidar_pyramid_joint_search_final_20260712_060922/round_000/generation_001.csv`
- `tests/outputs/lidar_pyramid_joint_search_final_20260712_060922/round_000/generation_002.csv`
- `tests/outputs/lidar_pyramid_joint_search_final_20260712_060922/round_000/generation_003.csv`
- `tests/outputs/lidar_pyramid_joint_search_final_20260712_060922/round_000/generation_004.csv`
- `tests/outputs/lidar_pyramid_joint_search_final_20260712_060922/round_000/stage1_scores.csv`
- `tests/outputs/lidar_pyramid_joint_search_final_20260712_060922/round_000/stage1_topk.json`
- `tests/outputs/lidar_pyramid_joint_search_final_20260712_060922/round_000/repair_report.json`
- `tests/outputs/lidar_pyramid_joint_search_final_20260712_060922/archives/fisher_statistics.pt`
- `tests/outputs/lidar_pyramid_joint_search_final_20260712_060922/archives/proxy_archive.jsonl`

Selected GPU and proxy:

- selected physical GPU: `0`
- runtime device: `cuda:0`
- proxy backend: `cuda_batched`
- proxy device: `cuda:0`
- proxy batch size: `128`

Population:

- initial population: `1024`
- active population: `512`
- generations: `5`
- outer rounds actually run: `1` (`round_000` only)
- Stage-1-only: true

Search space:

- coupled-channel gene count: `1504`
- precision-group gene count: `23`
- weighted module count: `69`
- trace hash: `95c4a6bdfc77b6c3a5c184e7d4ee2652d06cf4f507b17e96791682117742dedc`
- checkpoint hash: `d20a01079cc09b1f313a37bd5ccd5174390a93f5c2b02dc9a942929e1227caca`
- total atomic units reported by tracer: `7383`
- total coupled units reported by tracer: `7383`

Fisher statistics:

- path: `tests/outputs/lidar_pyramid_joint_search_final_20260712_060922/archives/fisher_statistics.pt`
- parameter tensors: `199`
- Fisher elements: `5464791`
- finite ratio: `1.0`
- nonzero ratio: `0.6886027297292797`
- NaN count: `0`
- Inf count: `0`

Round 0 objective:

- file: `tests/outputs/lidar_pyramid_joint_search_final_20260712_060922/round_000/stage1_objective_config.json`
- `T_BOPS`: `0.25`
- alpha: `0.55`
- beta: `0.25`
- gamma: `0.05`
- delta: `0.15`
- penalty formula: `squared_relative_excess`

Per-generation CSV row counts and summaries:

```text
generation_000.csv: 1024 candidates, target 0.25, feasible 564, mean INT8 MACs ratio 0.1998846797623628
generation_001.csv: 512 candidates, target 0.25, feasible 316, mean INT8 MACs ratio 0.1469478381134195
generation_002.csv: 512 candidates, target 0.25, feasible 273, mean INT8 MACs ratio 0.06564670748173285
generation_003.csv: 512 candidates, target 0.25, feasible 246, mean INT8 MACs ratio 0.045186058134390805
generation_004.csv: 512 candidates, target 0.25, feasible 218, mean INT8 MACs ratio 0.07445163171462355
```

Repair report:

- file: `tests/outputs/lidar_pyramid_joint_search_final_20260712_060922/round_000/repair_report.json`
- processed raw candidates: `3072`
- legal repaired phenotypes: `2964`
- repair failed count: `7`
- duplicate repaired phenotype count: `101`
- repaired Top-5 selected: `5`
- failure reasons:
  - `grouped_no_safe_width_le_min_keep:0`: `3`
  - `grouped_no_safe_width_le_min_keep:2`: `4`

Top-5 repaired candidates from `stage1_topk.json`:

```text
1. 22727ae55980912c61bab5f61cf36337847bff14170e033e0b026b1b1417c662
   repaired F1: 0.02515567986120007
   pruned units: 0
   realized INT8 layers in phenotype: 0

2. 939f92ea9dc1dde83fe739f3787bdaf6b4735c396219fa7a899fe9c4ea304c53
   repaired F1: 0.07017609498680816
   pruned units: 644
   realized INT8 layers in phenotype: 0

3. 5fd4d4acda8ff5940f4de50a7f96c168815d411006eea4b68f55b7234ad9d612
   repaired F1: 0.07255667035005917
   pruned units: 644
   realized INT8 layers in phenotype: 0

4. 8c6d886cace9cb26f1b917f4007aa6d276a88e5d4fe766c220e365bfc8620ae1
   repaired F1: 0.07385871710269507
   pruned units: 644
   realized INT8 layers in phenotype: 1

5. d1725da8f477fa0acf5aba34fca83f962c5d92b6164c8dfe856c85b64492adc7
   repaired F1: 0.0948538369474253
   pruned units: 656
   realized INT8 layers in phenotype: 0
```

Cache and GPU counters:

- The current `run_manifest.json` was overwritten by the final Stage-1-only resume run.
- Current manifest after resume:
  - cache hits: `6035`
  - cache misses: `0`
  - GPU batch count: `0`
  - scalar evaluate count: `0`
  - candidates/s: `0.0`
  - GPU peak memory: `0`
- The fresh 5-generation Stage-1 run before resume did execute CUDA batched scoring; observed from the immediately preceding run and proxy archive:
  - cache misses during fresh run: `5928`
  - GPU batch count during fresh run: `48`
  - scalar evaluate count: `0`
  - candidates/s during fresh run: about `252.7`
  - GPU peak memory during fresh run: about `2092668416` bytes
- `archives/proxy_archive.jsonl` currently has `11856` rows because it includes the earlier pre-cache-key-fix entries plus the current objective-key entries.

Artifact file list required by user command:

```text
tests/outputs/lidar_pyramid_joint_search_final_20260712_060922/archives/fisher_statistics_manifest.json
tests/outputs/lidar_pyramid_joint_search_final_20260712_060922/archives/fisher_statistics.pt
tests/outputs/lidar_pyramid_joint_search_final_20260712_060922/archives/proxy_archive.jsonl
tests/outputs/lidar_pyramid_joint_search_final_20260712_060922/archives/proxy_normalization.json
tests/outputs/lidar_pyramid_joint_search_final_20260712_060922/baseline/eval_manifest.json
tests/outputs/lidar_pyramid_joint_search_final_20260712_060922/commands.sh
tests/outputs/lidar_pyramid_joint_search_final_20260712_060922/context_report.json
tests/outputs/lidar_pyramid_joint_search_final_20260712_060922/environment.json
tests/outputs/lidar_pyramid_joint_search_final_20260712_060922/global_summary.csv
tests/outputs/lidar_pyramid_joint_search_final_20260712_060922/global_summary.json
tests/outputs/lidar_pyramid_joint_search_final_20260712_060922/local_pruning_domains.json
tests/outputs/lidar_pyramid_joint_search_final_20260712_060922/resolved_config.yaml
tests/outputs/lidar_pyramid_joint_search_final_20260712_060922/round_000/best_candidate.json
tests/outputs/lidar_pyramid_joint_search_final_20260712_060922/round_000/generation_000.csv
tests/outputs/lidar_pyramid_joint_search_final_20260712_060922/round_000/generation_001.csv
tests/outputs/lidar_pyramid_joint_search_final_20260712_060922/round_000/generation_002.csv
tests/outputs/lidar_pyramid_joint_search_final_20260712_060922/round_000/generation_003.csv
tests/outputs/lidar_pyramid_joint_search_final_20260712_060922/round_000/generation_004.csv
tests/outputs/lidar_pyramid_joint_search_final_20260712_060922/round_000/repair_report.json
tests/outputs/lidar_pyramid_joint_search_final_20260712_060922/round_000/round_summary.json
tests/outputs/lidar_pyramid_joint_search_final_20260712_060922/round_000/stage1_objective_config.json
tests/outputs/lidar_pyramid_joint_search_final_20260712_060922/round_000/stage1_scores.csv
tests/outputs/lidar_pyramid_joint_search_final_20260712_060922/round_000/stage1_topk.json
tests/outputs/lidar_pyramid_joint_search_final_20260712_060922/run_manifest.json
tests/outputs/lidar_pyramid_joint_search_final_20260712_060922/runtime_layer_shapes.json
tests/outputs/lidar_pyramid_joint_search_final_20260712_060922/seen_raw_genotypes.jsonl
tests/outputs/lidar_pyramid_joint_search_final_20260712_060922/seen_repaired_phenotypes.jsonl
```

## 6. 已验证测试

Commands actually executed in this session:

```bash
python -m py_compile $(find search -name '*.py') pruning/importance/second_order_fisher.py pruning/api.py pruning/importance/__init__.py
```

Result:

```text
exit code 0
no output
```

Pytest command actually executed:

```bash
pytest -q \
  tests/test_search_candidate_codec.py \
  tests/test_search_cache_reuse.py \
  tests/test_search_topk_diversity.py \
  tests/test_search_tool_adapters.py \
  tests/test_two_stage_joint_search.py \
  tests/test_search_baseline_engines.py \
  tests/test_search_baseline_precision_validation.py \
  tests/test_search_full_val_manifest.py \
  tests/test_search_second_order_fisher.py \
  tests/test_search_local_domain_ranking.py \
  tests/test_search_channel_alignment.py \
  tests/test_search_grouped_conv_actions.py \
  tests/test_search_independent_group_topk.py \
  tests/test_search_quantization_groups.py \
  tests/test_search_runtime_shapes.py \
  tests/test_search_bops_proxy.py \
  tests/test_search_bops_budget.py \
  tests/test_search_gpu_batch_proxy.py \
  tests/test_search_gpu_batch_integration.py \
  tests/test_search_large_population.py \
  tests/test_search_int8_realization.py \
  tests/test_search_final_contract.py
```

Result:

```text
81 passed in 2.53s
```

Diff checks:

```bash
git diff --check
# exit code 0, no output

git diff --name-only | grep -E '^(tracer|quantization)/' || true
# no output
```

Process check:

```bash
pgrep -af 'python -m search\.cli|trtexec|trt_build_worker|evaluation_worker' || true
```

Output only matched the `pgrep` command itself:

```text
2060954 /bin/bash -c pgrep -af 'python -m search\.cli|trtexec|trt_build_worker|evaluation_worker' || true
```

No real search, TensorRT build, or evaluation worker was running at handoff time.

## 7. 当前尚未完成

The following are false for the current real artifacts:

```text
Stage-2 Top-5 physical model export: false
Top-5 pruned ONNX: false
Top-5 Q/DQ ONNX: false
Top-5 TensorRT engines: false
Top-5 300-frame AP/latency: false
round winner: false
4 outer rounds real run: false
final winner: false
final full validation reevaluation: false
end-to-end validated: false
```

Important distinction:

- Code entry exists for Stage-2 deployment and evaluation.
- The current run has only executed Stage-1 and repaired Top-5 selection.
- No current `round_000/stage2/<candidate_hash>/` directories exist.
- No current `engine.plan`, `qdq.onnx`, `exported.onnx`, `physical_snapshot.json`, or 300-frame candidate `evaluation.json` exists for the Top-5.
- Baseline full validation engines/evaluations are not present under this run. Baselines were skipped for the Stage-1-only smoke/resume commands.

## 8. 下一会话的唯一推荐执行顺序

1. Read this handoff file first.
2. Check workspace status and active processes:

```bash
cd /home/lixingfeng/UniAD_examine/heal_compress
git status --short
pgrep -af 'python -m search\.cli|trtexec|trt_build_worker|evaluation_worker' || true
```

3. Inspect current run artifacts:

```bash
find tests/outputs/lidar_pyramid_joint_search_final_20260712_060922 -maxdepth 3 -type f | sort
cat tests/outputs/lidar_pyramid_joint_search_final_20260712_060922/round_000/stage1_topk.json
cat tests/outputs/lidar_pyramid_joint_search_final_20260712_060922/round_000/repair_report.json
```

4. Do not repeat the completed Stage-1 5-generation run unless code changed.
5. Continue from existing `round_000` repaired Top-5.
6. First run one candidate through physical pruning, ONNX export, Q/DQ, TensorRT engine, and a small evaluation/debug path.
7. Verify repaired mask equals physical pruning indices exactly:
   - Stage-1 repaired phenotype pruned ids
   - `SamplingPruningRequest.prune_indices`
   - physical plan
   - physical snapshot
   - ONNX dimensions
   - TensorRT layer shapes
8. After one candidate succeeds, run all Top-5 on the fixed 300-frame Stage-2 protocol.
9. Select the `round_000` winner using normalized Stage-2 F2.
10. Run outer rounds 1-3 with the same code and cache, using `T_BOPS` schedule `0.2267`, `0.2033`, `0.18`.
11. Reevaluate all round winners on final full validation if required by final config.
12. Verify resume and layered cache:
    - proxy cache
    - physical artifact cache
    - ONNX cache
    - calibration cache
    - Q/DQ cache
    - TensorRT engine cache
    - evaluation cache

## 9. 继续运行所需的精确命令

### Test command

```bash
cd /home/lixingfeng/UniAD_examine/heal_compress
python -m py_compile $(find search -name '*.py') pruning/importance/second_order_fisher.py pruning/api.py pruning/importance/__init__.py
pytest -q \
  tests/test_search_candidate_codec.py \
  tests/test_search_cache_reuse.py \
  tests/test_search_topk_diversity.py \
  tests/test_search_tool_adapters.py \
  tests/test_two_stage_joint_search.py \
  tests/test_search_baseline_engines.py \
  tests/test_search_baseline_precision_validation.py \
  tests/test_search_full_val_manifest.py \
  tests/test_search_second_order_fisher.py \
  tests/test_search_local_domain_ranking.py \
  tests/test_search_channel_alignment.py \
  tests/test_search_grouped_conv_actions.py \
  tests/test_search_independent_group_topk.py \
  tests/test_search_quantization_groups.py \
  tests/test_search_runtime_shapes.py \
  tests/test_search_bops_proxy.py \
  tests/test_search_bops_budget.py \
  tests/test_search_gpu_batch_proxy.py \
  tests/test_search_gpu_batch_integration.py \
  tests/test_search_large_population.py \
  tests/test_search_int8_realization.py \
  tests/test_search_final_contract.py
git diff --check
git diff --name-only | grep -E '^(tracer|quantization)/' || true
```

### Existing current candidate source path

This path exists:

```text
tests/outputs/lidar_pyramid_joint_search_final_20260712_060922/round_000/stage1_topk.json
```

Warning: `stage1_topk.json` is a list of selection records, not a direct `--candidate-config` input for the current `--stage2-only` CLI. The current CLI expects each `--candidate-config` to point to a single `CandidatePhenotype` or `CandidateGenotype` JSON. Therefore, a direct Stage-2-only command with existing individual candidate config paths does not yet exist.

Before Stage-2-only can be copied and run, the next session should either:

- split the five `phenotype` objects from `stage1_topk.json` into real files, for example under:

```text
tests/outputs/lidar_pyramid_joint_search_final_20260712_060922/round_000/stage2_candidate_configs/
```

or

- update `search.cli` / `_load_candidate()` to accept a Top-K selection list.

Do not pretend that `stage1_topk.json` can be passed directly to `--candidate-config`; it cannot with the current loader.

### Stage-2-only command template after individual candidate config files exist

Use only after creating real per-candidate JSON files from `stage1_topk.json`:

```bash
cd /home/lixingfeng/UniAD_examine/heal_compress
python -m search.cli \
  --config search/configs/lidar_pyramid_joint_search_final.yaml \
  --checkpoint /home/lixingfeng/UniAD_examine/Auto_Search/original_models/dairv2s/LiDAROnly/lidar_pyramid/net_epoch_bestval_at17.pth \
  --output-root tests/outputs \
  --gpu-id 0 \
  --resume tests/outputs/lidar_pyramid_joint_search_final_20260712_060922 \
  --stage2-only \
  --candidate-config tests/outputs/lidar_pyramid_joint_search_final_20260712_060922/round_000/stage2_candidate_configs/candidate_01_22727ae55980912c61bab5f61cf36337847bff14170e033e0b026b1b1417c662.json
```

The candidate config path in this template does not exist yet. The only currently existing candidate source is `stage1_topk.json`.

### Resume command that was run successfully for proxy cache

```bash
cd /home/lixingfeng/UniAD_examine/heal_compress
python -m search.cli \
  --config search/configs/lidar_pyramid_joint_search_final.yaml \
  --checkpoint /home/lixingfeng/UniAD_examine/Auto_Search/original_models/dairv2s/LiDAROnly/lidar_pyramid/net_epoch_bestval_at17.pth \
  --output-root tests/outputs \
  --gpu-id auto \
  --resume tests/outputs/lidar_pyramid_joint_search_final_20260712_060922 \
  --outer-rounds 1 \
  --initial-population-size 1024 \
  --population-size 512 \
  --offspring-size 512 \
  --generations 5 \
  --stage1-only \
  --skip-baselines
```

Current resume result in `run_manifest.json`:

```text
cache_hit_count: 6035
cache_miss_count: 0
gpu_batch_count: 0
scalar_evaluate_call_count: 0
```

### Full multi-round command

This has not been run to completion. It will run baselines and Stage-2 if `--stage1-only` and `--skip-baselines` are omitted:

```bash
cd /home/lixingfeng/UniAD_examine/heal_compress
python -m search.cli \
  --config search/configs/lidar_pyramid_joint_search_final.yaml \
  --checkpoint /home/lixingfeng/UniAD_examine/Auto_Search/original_models/dairv2s/LiDAROnly/lidar_pyramid/net_epoch_bestval_at17.pth \
  --output-root tests/outputs \
  --gpu-id auto
```

Before running this full command, fix the candidate-config/list issue or verify Stage-2 Top-5 selection path can consume repaired records directly during non-stage2-only mode.

## 10. 风险和未决问题

- Repaired phenotype group maps:
  - Unit test verifies `group_keep_map_by_scope` enters `SamplingPruningRequest`.
  - Real Stage-2 has not yet executed this on Top-5, so physical replay is unverified.

- Stage-2 default selector risk:
  - Code path should not call default grouped top-k for raw coupled-channel ids.
  - Real Stage-2 must verify no `shared_local_mean` or `independent_group_topk` overrides repaired group maps.

- Physical hash:
  - Stage-2 code computes physical hash from legal physical plan and structure snapshot.
  - No current Top-5 physical hash exists because Stage-2 has not run.

- 300-frame manifest:
  - `baseline/eval_manifest.json` exists and context reports `frame_count_with_warmup: 330`.
  - It was generated for Stage-1-only context with `num_frames=300`, `warmup_frames=30`.
  - No Top-5 300-frame evaluation has run.

- Baseline AP/latency reference:
  - Baseline full validation engines/evaluations are not present in current run because `--skip-baselines` was used for Stage-1 smoke/resume.
  - Final Stage-2 F2 requires same 300-frame manifest baseline references:
    - AP reference: original strict FP32.
    - latency reference: original strict FP16.

- Strict INT8 fallback:
  - Quantization group legality and fallback report code exists.
  - Current Top-5 phenotypes mostly have 0 realized INT8 layers; one Top-5 phenotype has 1 INT8 layer in Stage-1 profile.
  - No TensorRT INT8 realization has been verified for current Top-5.

- Top-5 structure uniqueness:
  - Top-5 repaired phenotype hashes are unique.
  - `repair_report.json` reports `duplicate_repaired_phenotype_count: 101` in the processed raw pool, and selected count `5`.

- Resume cache:
  - Proxy resume is verified.
  - Physical/ONNX/calibration/QDQ/engine/eval resume is not verified for current final run because Stage-2 has not run.

- Current run manifest caveat:
  - `run_manifest.json` is post-resume and shows cache miss `0`, GPU batch `0`.
  - Fresh GPU scoring occurred before the final resume, but those fresh counters were overwritten in manifest.
  - Use `proxy_archive.jsonl`, generation CSV files, and this handoff to reconstruct the history.

## Required pre-write command outputs

These were executed before writing this handoff.

```bash
git status --short
```

Output:

```text
 M pruning/api.py
 M pruning/importance/__init__.py
 M pruning/importance/second_order_fisher.py
 M search/__init__.py
?? scripts/run_two_stage_joint_search.sh
?? search/adapters/
?? search/baselines/
?? search/cache/
?? search/candidate.py
?? search/candidate_codec.py
?? search/canonicalization.py
?? search/cli.py
?? search/configs/
?? search/ga/
?? search/hashing.py
?? search/integration/
?? search/orchestration/
?? search/proxy/
?? search/pruning_space/
?? search/quantization_space/
?? search/stage1/
?? search/stage2/
?? search/surrogate/
?? tests/test_grouped_conv_stage_sensitivity_v2.py
?? tests/test_search_baseline_engines.py
?? tests/test_search_baseline_precision_validation.py
?? tests/test_search_bops_budget.py
?? tests/test_search_bops_proxy.py
?? tests/test_search_cache_reuse.py
?? tests/test_search_candidate_codec.py
?? tests/test_search_channel_alignment.py
?? tests/test_search_final_contract.py
?? tests/test_search_full_val_manifest.py
?? tests/test_search_gpu_batch_integration.py
?? tests/test_search_gpu_batch_proxy.py
?? tests/test_search_grouped_conv_actions.py
?? tests/test_search_independent_group_topk.py
?? tests/test_search_int8_realization.py
?? tests/test_search_large_population.py
?? tests/test_search_local_domain_ranking.py
?? tests/test_search_quantization_groups.py
?? tests/test_search_runtime_shapes.py
?? tests/test_search_second_order_fisher.py
?? tests/test_search_tool_adapters.py
?? tests/test_search_topk_diversity.py
?? tests/test_two_stage_joint_search.py
?? tools/experiments/grouped_conv_stage_sensitivity/
?? tools/experiments/run_grouped_conv_stage_sensitivity.py
```

```bash
git diff --name-only
```

Output:

```text
pruning/api.py
pruning/importance/__init__.py
pruning/importance/second_order_fisher.py
search/__init__.py
```

```bash
git diff --check
```

Output:

```text
no output; exit code 0
```

```bash
pgrep -af 'python -m search\.cli|trtexec|trt_build_worker|evaluation_worker' || true
```

Output:

```text
2060954 /bin/bash -c pgrep -af 'python -m search\.cli|trtexec|trt_build_worker|evaluation_worker' || true
```

```bash
find tests/outputs/lidar_pyramid_joint_search_final_20260712_060922 -maxdepth 3 -type f | sort
```

Output is the artifact file list shown in Section 5.
