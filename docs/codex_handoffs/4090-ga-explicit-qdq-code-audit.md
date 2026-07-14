# 4090 GA explicit-QDQ code audit

## Audit scope and immutable starting point

This audit covers the production lidar-pyramid structured-pruning and mixed-precision search path at commit `9145d0351170c0359e9d05d5ee82c9bfb383bedb`. It treats H800 artifacts as historical evidence only. No H800 plugin, ONNX, TensorRT engine, calibration cache, timing cache, latency LUT, or latency result is accepted for the 4090 gate.

The required preflight was executed before any edit:

```text
$ git branch --show-current
feature/heal-compress-h800-sync-4090

$ git status
On branch feature/heal-compress-h800-sync-4090
Your branch is up to date with 'origin/feature/heal-compress-h800-sync-4090'.
nothing to commit, working tree clean

$ git merge-base --is-ancestor b862b3d8ad061bd12580776226c75f564918298d HEAD
$ echo $?
0
```

The H800 source-of-truth handoff `docs/codex_handoffs/H800_EXPLICIT_QDQ_ACCEPTANCE_20260714.md` was read in full. Its Round 5-6 correction supersedes the earlier E27 conclusion: the readiness baseline is canonical 70 compute entries with 67 INT8 and 3 FP16 entries.

## Formal entry points

- CLI: `python -m search.cli --config <yaml>` in `search/cli.py::main`.
- Real lidar-pyramid orchestration: `search/orchestration/lidar_pyramid_search.py::LidarPyramidTwoStageSearch.run`.
- Context and search-space construction: `search/integration/lidar_pyramid_context.py::build_lidar_pyramid_context`.
- Real deployment worker: `search/stage2/lidar_pyramid_real_evaluator.py::LidarPyramidRealEvaluator`.
- Baseline-only gate: `python -m search.cli ... --baseline-only`.
- Production E67 baseline selector: `matched_legacy_int8` (or the equivalent maximal-legal selection), not the E27-only `trusted_explicit_qdq_int8` module tuple.

The existing H800 acceptance YAML still requests `trusted_explicit_qdq_int8`, which is the historical 27-module control. A separate 4090 readiness configuration must explicitly request `strict_fp16` and `matched_legacy_int8` and must use the local TensorRT root `/home/lixingfeng/UniAD_examine/HEAL/prune_model/TensorRT-10.9_x86_cu118`.

## Candidate schema and data flow

The active formal schema is:

```text
CandidateGenotype
  pruning_genes: pruning atomic/action ID -> 1 keep / 0 remove
  precision_genes: quantization group ID -> FP32 / FP16 / INT8

CandidatePhenotype
  pruned_unit_ids: repaired physical pruning unit/action IDs
  precision_profile: canonical weighted module -> requested/legalized precision
  metadata: requested group profile, legalized group profile, group expansion,
            merge contract, and pruning repair maps
```

`CandidateGenotype.from_dict` accepts `prune_vars` and `bitwidth_vars` compatibility aliases. The older `search/search_space.py::CandidateEncoding` uses those names directly. Neither formal schema currently exposes the required external names `group_mask` and `layer_bitwidth`. The semantics are separable, but a strict audited serialization/crosswalk is still required before Stage A.

Stage-1 to Stage-2 flow in the current code:

1. `build_lidar_pyramid_context` traces the real model and constructs pruning and quantization spaces.
2. `search/ga/engine.py::GeneticSearchEngine.run` creates and mutates `CandidateGenotype` objects.
3. `search/canonicalization.py::repair_genotype` legalizes precision groups; pruning mask repair is applied later by `_repair_raw_keep_mask`.
4. `Stage1ProxyEvaluator` evaluates Fisher, SQNR, parameter-size, and BOPS proxies.
5. `_repair_raw_keep_mask` performs monotonic dense/grouped channel repair.
6. `select_repaired_stage2_topk` canonicalizes, proxy-rescores, and candidate-hash deduplicates repaired phenotypes.
7. `LidarPyramidRealEvaluator.evaluate_candidate` performs physical replay, ONNX export, canonical mapping, calibration, explicit QDQ insertion, TRT build/inspection, and engine evaluation.
8. `compute_stage2_score` calculates `F2 = eta_map * L_map_real + eta_latency * R_latency_real` and can apply an AP-drop penalty.

## Pruning dependency groups and repair

Pruning ownership is confined to the structural path:

- Trace source: `trace_result.atomic_prune_units` and `trace_result.coupled_channel_units` in `build_lidar_pyramid_context`.
- Search subset: `_select_search_atomic_units`.
- Dependency closure/action catalog: `search/pruning_space/action_catalog.py::build_pruning_action_catalog`.
- Local pruning domains: `search/pruning_space/local_domains.py::build_local_pruning_domains`.
- Dense repair: `dense_floor_repair`.
- Grouped-convolution repair: `grouped_equal_count_floor_repair`.
- Physical request/plan/materialization: `FormalPruningAdapter`, `validate_repaired_physical_plan`, and `LidarPyramidRealEvaluator._materialize_physical`.
- Final physical identity: `search/hashing.py::physical_hash`, using the legalized physical plan, physical snapshot, checkpoint hash, and pruning policy version.

Pruning metadata carries closure members, root channel indices, scope IDs, grouped-convolution group maps, and physical keep/remove decisions. No quantization group is consulted to create a channel mask in the reviewed formal path.

## Quantization groups and canonical crosswalk

Quantization ownership is separate from pruning:

- Precision dependency source: `tracer/precision_coupling_tracer.py::build_precision_coupling_groups`.
- Model-specific contract split: `search/integration/lidar_pyramid_context.py::_build_precision_groups`.
- GA quantization variables: `search/quantization_space/group_builder.py::build_quantization_search_groups`.
- Precision legalization: `legalize_group_precision_genes`.
- ONNX canonical origin mapping: `quantization/export/origin_mapping.py::build_onnx_origin_map`.
- Canonical precision join: `quantization/precision/canonical_mapping.py::build_canonical_precision_mapping`.

`_build_precision_groups` explicitly states that precision coupling is a deployment contract rather than a pruning dependency scope. Residual and concat branches can choose independent compute precision; FP16 merge behavior is represented by explicit deployment metadata. Each weighted module becomes its own search precision group after multi-member merge groups are split. This is the intended separation, but the required machine-readable namespace/crosswalk audit is not yet emitted.

The corrected canonical mapping statically contains the protected parameter-free entry `pyramid_backbone.functional_affine_grid_matmul`, with precision group `protected_functional_affine_grid`, requested/realized FP16, and the affine-grid member MatMul constraints. Duplicate canonical names and origin/profile set mismatches fail closed.

## Channel resolution

There are two different implementations:

- `search/proxy/batch_channel_resolver.py::BatchChannelResolver` resolves pruning choices to virtual per-layer `C_in`, `C_out`, groups, parameter counts, and MACs for Stage-1.
- `opencood/tools/compression/latency_lut/channel_resolver.py::ChannelResolver` resolves already materialized deployment units and precision configuration for the legacy LUT path.

The legacy `ChannelResolver` explicitly raises `NotImplementedError` for raw `group_mask`. The formal search path bypasses it and uses parameter slices plus `BatchChannelResolver`/virtual shapes, followed by physical prune replay in Stage-2. Therefore the requested single audited mapping

```text
candidate -> physical channels -> deployment units -> precision profile
```

does not currently exist as one fail-closed contract. This is a Stage-A blocker until an audit/crosswalk proves the formal resolvers agree with physical replay.

## Physical replay and ONNX export

`LidarPyramidRealEvaluator._materialize_physical` performs the required physical path:

```text
phenotype
  -> pruning request
  -> build plan
  -> legalize plan
  -> validate repaired plan
  -> materialize a new model
  -> physical snapshot and validation
  -> physical hash
```

All-keep models additionally require exact ordered state-dict key, shape, dtype, value, per-tensor hash, and parameter-count identity.

`_export_qdq` wraps the physically pruned model with the fixed-K TRT-compatible export wrapper, exports ONNX via `export_pruned_signal_maxk_onnx`, records `origin_map.json`, and runs the export/checker validation in the formal quantization exporter. ONNX cache lookup is currently keyed only by `physical_hash`; it verifies the cached ONNX byte hash but does not include the code commit or export recipe in the index key.

## Production explicit QDQ and scale generation

The reviewed production recipe is configured by `QDQConfig`:

- semantic output boundary policy: `semantic_post_relu_or_post_merge_v1`;
- merge policy: FP16 merge;
- symmetric zero point: `0`;
- weight granularity: per-channel;
- current policy version: `explicit-qdq-canonical-fp16-int8-v5-semantic-activation-boundary`.

Topology is generated by `quantization/precision/qdq_inserter.py::insert_explicit_qdq`. It resolves activation output boundaries before inserting Q/DQ, inserts activation-input and weight Q/DQ first, inserts output Q/DQ on the resolved semantic tensor, inserts explicit FP16 merge casts, audits weighted boundaries/merge contracts, and stores `qdq_topology_hash` in calibration metadata.

Per-channel weight scales are derived from the exported physical ONNX initializer by the calibration provider. The QDQ inserter requires a one-dimensional scale, a declared axis, a valid normalized axis, and:

```text
len(weight_scale) == weight_initializer.shape[weight_axis]
```

Conv weight axis is expected to be 0. The scale/zero-point nodes are generated symmetrically with zero point 0.

Activation scales for the production route are generated by `IInt8EntropyCalibrator2` in `search/integration/tensorrt_entropy_calibration_worker.py`, invoked through `build_tensorrt_entropy_calibration_cache_modelopt`. The fixed train200 NPZ manifest identity, physical ONNX hash, precision-selected modules, recipe, semantic policy, merge policy, and manifest hash participate in the calibration identity. A fresh 4090 run must force rebuild.

## TensorRT build and precision realization

`LidarPyramidRealEvaluator._build_engine` constructs `TensorRTBuildConfig` with FP16, conditional INT8, TF32 disabled, `precisionConstraints=obey`, fixed shape profiles, plugin path, layer-info export, and skip-inference build mode. `search/stage2/trt_build_worker.py` then:

1. builds and serializes the engine;
2. validates EngineInspector structure against the physical snapshot;
3. validates canonical requested/legalized realization;
4. returns a failing status on structure or precision mismatch.

The evaluator separately joins merge graph contracts to EngineInspector formats and fails on invalid merge realization, then produces the production boundary report. An explicit serialize/deserialize probe is still required by the 4090 readiness gate; serialization alone is not sufficient evidence.

## Stage-1 objective and BOPS

Stage-1 uses:

- `FisherTaylorProxy` with the configured Fisher/L1 fallback behavior;
- `SQNRProxy`;
- `SizeProxy`;
- `BOPSProxy`;
- `ProxyObjective` for F1;
- optional latency-related search metadata/LUT components outside the active formal objective.

`BOPSProxy` uses virtual post-pruning `C_in/C_out`, runtime shapes, groups, and full `weight_bits * activation_bits`. It reports retention against FP32 and FP16 baselines. It uses the Stage-1 legalized phenotype precision, so it is a proxy/admission estimate, not final realized BOPS.

There is no Stage-2 realized-BOPS implementation in the reviewed code. No post-engine budget check recomputes BOPS from the physical snapshot and EngineInspector realized precision. This is a hard blocker for the requested fixed interval `[target - 0.005, target + 0.005]`.

## Evaluation worker, warmup, latency, and manifests

`write_eval_manifest` writes deterministic warmup/evaluation frame IDs and a manifest hash. With `reset_after_warmup=true`, the evaluation worker runs separate warmup and evaluation phases. Warmup rows are marked and excluded from AP and reported latency distributions. The worker fails when the fixed manifest is incomplete, when evaluation or warmup frames are skipped, or when counts differ.

The worker reports AP@0.3/0.5/0.7, mAP, p50/p90/p95 for forward/postprocess/total, evaluated frame count, skipped frame count, exact frame IDs, and latency rows. It does not currently report FPS or GPU UUID/utilization/temperature/power/memory telemetry. Those are required additions for Stage A/B reporting and latency trust gates.

## Cache and signature audit

Current hash locations:

- raw genotype: `_raw_genotype_hash`;
- Stage-1 search cache: `search_hash`;
- repaired candidate: `candidate_hash`;
- physical artifact: `physical_hash`;
- deployment: `deployment_hash`;
- real evaluation: `eval_hash` plus the evaluator's real-cache key;
- calibration: `physical_hash + hash(calibration_identity)` path.

The current deployment hash includes physical hash, realized module precision profile, calibration scale hash, ONNX export config hash, TRT version, GPU compute capability, builder flags, optimization profiles, plugin hashes, and a group/merge contract hash. It does not explicitly include all required fields: code commit, base ONNX hash, canonical mapping hash, legalized profile hash, calibration manifest hash/recipe as independent fields, CUDA version, and QDQ topology hash as a first-class field. Some are nested indirectly, which is insufficient for a fail-closed lineage audit. Pre-`b862b3d` caches are not version-invalidated by a required code-commit component.

## Blocking findings before readiness and Stage A

1. The existing GA performs all generations inside one outer round and selects/deploys one aggregate Top-5 only after the full GA run. It does not deploy each generation's independent Top-5.
2. Stage-2 has no realized-BOPS computation or exact budget interval admission gate.
3. Candidate serialization does not expose and audit the required `group_mask` / `layer_bitwidth` namespaces and explicit canonical crosswalk.
4. The required pruning/quantization group audit counters are not emitted.
5. Deployment/cache signatures do not explicitly contain the full required lineage.
6. Candidate Top-5 deduplication uses repaired candidate hash before physical/deployment hashes exist; there is no deployment-time backfill loop to guarantee five unique deployable `physical_hash + deployment_hash` pairs.
7. The current Stage-2 worker lacks FPS and required GPU telemetry/competition gating.
8. The H800 acceptance YAML selects the historical E27 trusted control, not the corrected E67 matched-coverage readiness baseline.
9. The current formal Stage-2 hard gate can penalize AP loss but does not fail admission before winner selection unless a concrete `max_map_drop` is configured.
10. The production boundary report does not emit all requested aggregate counter names, although its per-layer audit contains most source evidence.

Because these are reproducible formal-path gaps, GA Stage A must not start. The next implementation round must add tests first, then minimally add group/crosswalk audit, complete cache lineage, realized-BOPS admission, per-generation Top-5 orchestration/backfill, deterministic shared 300/500 manifests, and the 4090 readiness config/report path. Tracer and pruning planner behavior should remain unchanged.

## Audit verdict

- H800 correction commit present: `true`
- Static E67 production recipe present: `true`, through `matched_legacy_int8`/maximal legal selection
- Pruning and quantization constructors visibly separate: `true`
- Required machine-readable group separation audit complete: `false`
- 4090 fresh deployment gate complete: `false`
- Existing orchestration satisfies per-generation Top-5: `false`
- Existing Stage-2 satisfies realized BOPS budget gating: `false`
- READY_FOR_GA: `not evaluated; blocked pending implementation and fresh 4090 gate`

