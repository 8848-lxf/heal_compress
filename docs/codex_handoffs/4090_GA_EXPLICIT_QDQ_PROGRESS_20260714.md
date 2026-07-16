# 4090 GA explicit-QDQ progress handoff

## Round 1 - branch gate and formal-path code audit

Starting point: branch `feature/heal-compress-h800-sync-4090`, HEAD `9145d0351170c0359e9d05d5ee82c9bfb383bedb`. The working tree was clean and `b862b3d8ad061bd12580776226c75f564918298d` was verified as an ancestor with exit code 0. No command targeted `feature/heal-compress-h800`, no GA was started, and no H800 binary artifact was reused.

Completed work:

- Read the H800 explicit-QDQ acceptance handoff in full and treated Round 5-6 as the current source of truth.
- Located the production CLI, lidar-pyramid orchestration, context construction, Stage-1 GA/proxy path, physical prune replay, ONNX export, canonical mapping, semantic QDQ insertion, EntropyCalibration2 worker, TRT build/inspection, evaluation worker, and cache hashes.
- Wrote `docs/codex_handoffs/4090-ga-explicit-qdq-code-audit.md` with the exact data flow, group ownership, topology/scale generation, cache locations, and realized-BOPS status.
- Confirmed the static production implementation contains the canonical 70-entry functional affine-grid mapping, semantic post-ReLU/post-merge boundary policy, per-channel weight-scale length validation, zero point 0, FP16 merge contracts, and TensorRT entropy calibration path.
- Identified fail-closed blockers before Stage A: aggregate-per-round rather than per-generation Top-5, missing Stage-2 realized BOPS gate, incomplete cache lineage, missing group/crosswalk audit counters, missing deployment-hash backfill, missing GPU telemetry/FPS, and stale E27 selection in the H800 acceptance YAML.

Files added:

- `docs/codex_handoffs/4090-ga-explicit-qdq-code-audit.md`: formal code audit and readiness blockers.
- `docs/codex_handoffs/4090_GA_EXPLICIT_QDQ_PROGRESS_20260714.md`: append-only 4090 continuation anchor.

Current gates:

- `READY_FOR_GA`: not evaluated.
- Stage A started: no.
- Stage B allowed: no.

Next round: add focused failing tests for group namespace/crosswalk audit, complete cache/deployment signature, physical+EngineInspector realized BOPS, and per-generation Top-5 orchestration. Keep tracer and pruning planner unchanged. Then run the required CPU regression suites and the fresh 4090 baseline gate before deciding `READY_FOR_GA`.

--- Round 1 completed: 2026-07-14 15:01:44 CST ---

## Round 2 - fail-closed GA deployment integration

Starting point: branch `feature/heal-compress-h800-sync-4090`, commit `2cf45df`. No GPU experiment or GA run was started in this round. Implementation followed test-first red/green cycles for each new behavior.

Implemented production changes:

- `search/group_separation_audit.py` adds the machine-readable pruning/quantization namespace and canonical crosswalk audit. It emits all required counters and raises `pruning_quant_group_audit_failed` before calibration/QDQ/TRT when collision, unmapped weighted layer, duplicate canonical ownership, or pruning-scope precision inheritance is detected.
- `search/quantization_space/group_builder.py` explicitly marks precision groups as `group_namespace=quantization` and `precision_group_source=precision_coupling_tracer` without changing the tracer or pruning planner.
- `search/candidate.py` exposes the required `group_mask` and `layer_bitwidth` serialization while preserving existing `pruning_genes`/`precision_genes` compatibility.
- `search/canonicalization.py`, `search/hashing.py`, and `search/integration/lidar_pyramid_context.py` propagate the exact Git commit through search/candidate cache identities. The deployment signature now requires explicit code commit, physical model, base ONNX, QDQ topology, canonical mapping, legalized/realized profiles, calibration manifest/recipe, TensorRT/CUDA/GPU architecture, and plugin binary hashes.
- `search/stage2/trt_build_worker.py` records the actually loaded modelopt TensorRT, CUDA, PyTorch, GPU architecture, and GPU name rather than trusting configuration constants.
- `search/stage2/realized_bops.py` resolves actual per-canonical precision from EngineInspector and computes final BOPS from physical runtime MACs with full `weight_bits * activation_bits`, fixed original-FP32 reference, physical parameter retention, and weight-storage retention.
- `search/stage2/lidar_pyramid_real_evaluator.py` now runs group audit before calibration, validates EngineInspector precision, profiles the physical model, applies the realized BOPS interval before AP evaluation, writes deployment lineage/signature artifacts, and records realized BOPS/parameter/storage metrics in evaluation output.
- `search/ga/engine.py` adds a synchronous per-generation callback after scoring and before reproduction.
- `search/orchestration/generation_stage2.py` implements per-generation Stage-2 deployment with ranked backfill, exact `physical_hash + deployment_hash` uniqueness, finite-F2 admission, failure records, and the required `generation_XXX_{top5,stage2,winner}` artifacts.
- `search/orchestration/lidar_pyramid_search.py` adds opt-in `per_generation_stage2`, exact two-sided Stage-1 BOPS interval gating, genuine-compression filtering, per-generation repair/rescore/deploy/backfill, and bypasses the historical aggregate-round Top-5 path for the 4090 config.
- `search/stage2/objective.py` turns a configured `max_map_drop` into a real `accuracy_hard_gate_failed` admission failure instead of a finite penalty that could still win.
- `search/integration/data_provider.py` adds deterministic evaluation offsets so the shared 500-frame budget-final manifest can avoid the shared 300-frame Stage-2 set.
- Added `search/configs/lidar_pyramid_4090_explicit_qdq_readiness.yaml` with strict FP16 plus corrected E67 `matched_legacy_int8`, train200 TensorRT EntropyCalibration2, local TensorRT root, fresh calibration, and fixed 200-frame gate.
- Added `search/configs/lidar_pyramid_4090_ga_stage_a.yaml` with 1024/512/512 GA sizes, five generations, per-generation Top-5, target 0.21 +/- 0.005, 300-frame Stage-2, and a non-overlapping 500-frame manifest specification.

New focused tests:

- `tests/test_search_group_separation_audit.py`
- `tests/test_search_trt_runtime_provenance.py`
- `tests/test_search_stage2_realized_bops.py`
- `tests/test_search_generation_stage2.py`

Extended regression coverage includes candidate codec, deployment signature, AP hard gate, evaluation manifest offset, GA callback/global dedup, quantization group provenance, and both 4090 YAML contracts.

Verification evidence:

- `265 passed` across every `tests/test_search*.py`, `tests/test_two_stage_joint_search.py`, formal package CPU tests, formal lidar-pyramid orchestration, tooling smoke, and no-test-dependency checks;
- all modified Python files passed `python -m py_compile`;
- `git diff --check` passed;
- seven warnings are existing ModelOpt/PyTorch compatibility/deprecation warnings, not test failures.

Current gates and remaining work:

- `READY_FOR_GA`: not evaluated.
- Stage A started: no.
- Stage B allowed: no.
- The fresh 4090 plugin build, plugin load/engine deserialize probes, strict FP16/E67 10-frame smoke and common-manifest 200-frame gate remain pending.
- The 500-frame generation-winner re-evaluation executor and runtime GPU telemetry/competition gate remain pending; Stage A cannot be declared complete until they are implemented and tested.

--- Round 2 completed: 2026-07-14 16:16:10 CST ---

## Round 3 - fresh plugin gate, final-winner executor, and GPU isolation

Starting point: branch `feature/heal-compress-h800-sync-4090`, commit `4591231`.
No Stage-A population was generated and no 300/500-frame result was collected.

Implemented production changes:

- `search/integration/runtime_environment.py` no longer assumes a
  `/home/lixingfeng/miniconda3` activation script. It resolves the actual conda
  prefix and activates `/home/lixingfeng/anaconda3/envs/modelopt` on this host.
- The same module now captures GPU UUID/name/driver, memory, utilization,
  temperature, power and compute process PID/user/command/duration. Its
  fail-closed isolation audit distinguishes the current search PID from foreign
  compute processes and raises `gpu_competition_detected` before latency work.
- `search/stage2/trt_build_worker.py` now loads the production plugin and
  explicitly deserializes the just-written engine. It records engine bytes,
  I/O tensor count and plugin ordering, and returns
  `engine_deserialize_failure` on any lifecycle error.
- `search/orchestration/budget_final.py` evaluates strict-FP32/strict-FP16
  references and every generation winner on one deterministic final manifest,
  requires exact N/N with zero skips, reuses only identical physical+deployment
  identities, computes the same F2 policy, and writes the budget winner and CSV.
- `search/orchestration/lidar_pyramid_search.py` invokes the final executor
  automatically after all per-generation winners for a budget have completed.
- `search/stage2/lidar_pyramid_real_evaluator.py` passes the concrete engine path
  into final reevaluation and records isolation telemetry immediately before
  and after every real evaluation.

Fresh local evidence:

- clean SM89 plugin build passed with TensorRT 10.9/CUDA 11.8;
- plugin SHA256 is
  `f5fd5b17cfe5f560452f1cf6f37b695c825fd02b263061ed04140895d24f24c1`;
- TensorRT registry load found `PointPillarScatterTRT` v1;
- fresh minimal engine serialization/deserialization passed on physical GPU 1;
- engine SHA256 is
  `2851187fa2096c8d33d2f7f042e96fd2d5d1a5bb7ed09c3e0295cf2eb9aa35d6`;
- this minimal gate was not benchmarked and contributes no latency claim.

Verification:

- focused integration/runtime suite: 32 passed;
- full formal/search CPU suite in `univ2x-opt`: 272 passed, 7 known warnings;
- all modified Python files passed `python -m py_compile`;
- the first broad invocation named two nonexistent test files and ran no tests;
  it was corrected to the repository's actual
  `test_formal_tooling_smoke.py` and
  `test_formal_pruner_no_test_dependency.py` paths;
- a diagnostic broad run in `modelopt` had 271 pass and one environment-only
  import failure because that TRT environment lacks the Python `modelopt`
  package; the complete suite passed in the intended CPU environment.

Readiness result and blocker:

- all eight RTX 4090 GPUs have a foreign long-running compute process owned by
  `guohongze`, so none passes the new isolation gate;
- strict FP16/E67 10-frame smoke and common 200-frame gate were not started;
- `docs/codex_handoffs/4090-ga-explicit-qdq-readiness-report.md` records the
  full fail-closed decision and the exact remaining gates;
- `READY_FOR_GA = false`;
- Stage A started: no;
- `STAGE_B_ALLOWED = false`.

Next round: poll for an isolated 4090. Once one is free, rerun the formal
readiness config from scratch, require actual 70-entry/67+3/boundary/scale/
entropy/merge/precision/topology audits and 200/200 zero-skip results, and only
then change `READY_FOR_GA` or start generation 0.

--- Round 3 completed: 2026-07-14 16:49:04 CST ---

## Round 4 - explicit shared-GPU authorization before readiness

Starting point: branch `feature/heal-compress-h800-sync-4090`, commit
`fce9430`. The user explicitly directed the experiment to continue on cards
with free capacity despite the resident low-utilization processes.

Implemented changes:

- `search/integration/runtime_environment.py` keeps the default foreign-process
  rejection, but adds an explicit shared-card authorization and a sampled GPU
  utilization ceiling. Foreign PID/user/command/memory/duration remain in every
  report even when the gate passes.
- `search/integration/lidar_pyramid_context.py` records the shared-card policy
  in the formal context report.
- `search/orchestration/lidar_pyramid_search.py` parses the runtime policy once
  and passes it through context construction and the initial preflight.
- `search/stage2/lidar_pyramid_real_evaluator.py` and
  `search/orchestration/budget_final.py` apply the identical policy before and
  after all real 200/300/500-frame evaluations.
- The shared-card policy remains available as an explicit opt-in, but before
  push the user freed GPUs 4-7. Both formal 4090 configs therefore select
  physical GPU 5 with `allow_foreign_gpu_processes: false` and retain the
  20-percent unexplained-utilization ceiling. Other configs are unchanged.

Test-first evidence:

- low-utilization shared-card audit failed before the API existed, then passed;
- utilization 21 percent is rejected at a configured 20 percent ceiling;
- default foreign-process rejection remains covered;
- YAML contract tests first failed on the previous `gpu_id: auto`, passed for
  the shared GPU-1 authorization, then failed/passed again when the final
  experiment selection moved to isolated GPU 5;
- 19 focused runtime/config/final-evaluation tests passed.

Final live preflight on GPU 5:

- UUID: `GPU-d4b8342a-7038-f567-ce40-a4705b23854b`;
- free memory: 24,080 MiB;
- sampled utilization: 0 percent;
- compute PID count: 0;
- strict isolation gate: passed.

Current gates:

- `READY_FOR_GA = false` pending the fresh strict-FP16/E67 engine and 200-frame
  results;
- Stage A started: no;
- Stage B allowed: no.

Next round: commit this policy so deployment signatures bind the correct code,
then run the fresh readiness config on GPU 1. Do not start generation 0 unless
the full E67 and strict-FP16 audits pass.

--- Round 4 completed: 2026-07-14 17:34:28 CST ---

## Round 5 - first 4090 E67 run and compiler-backend merge audit fix

Starting point: commit `064a05d`, isolated physical GPU 5. Stage A was not
started.

First readiness run:

- output: `outputs/4090_ga_explicit_qdq_readiness_20260714_024246/`;
- strict FP16 engine build, deserialize, structure and precision passed;
- strict FP16 was stopped before evaluation by
  `concat_merge_not_compatible:/Concat_9:not_yet_verified`;
- E67 completed 200/200 with zero skips and mAP 0.649823;
- E67 AP03/AP05/AP07 were 0.758102/0.706666/0.484700;
- E67 canonical profile was 67 INT8 / 3 FP16 / 0 unresolved;
- the local all-keep topology hash exactly matched the H800 accepted hash;
- the run was not promoted because strict FP16 had failed closed.

Root cause:

- the 4090 TensorRT 10.9 compiler backend represented `/Concat_9` as three
  `__myl_Move` layers;
- their Layer Name and Metadata omitted the ONNX concat name;
- each output tensor was exactly `/Concat_9_output_0` with datatype Half;
- `_engine_merge_precision_realization` matched only Layer Name and therefore
  incorrectly emitted `not_yet_verified`.

Minimal test-first fix:

- `tests/test_search_merge_realization.py` reproduces the real compiler-backend
  layer-info shape and failed before implementation;
- `search/stage2/lidar_pyramid_real_evaluator.py` now falls back to exact merge
  output-tensor matching and retains the existing datatype checks;
- the preserved strict-FP16 artifact re-audits as FP16 with no merge issues;
- full formal/search CPU regression: 276 passed, 7 known warnings;
- no QDQ topology, precision policy, calibration, scoring or BOPS rule changed.

Cache/result policy:

- the first readiness engines and results remain diagnostic only;
- the fix must be committed before rerun so code/deployment signatures change;
- readiness restarts fresh in a new timestamp directory;
- `READY_FOR_GA = false` until the rerun completes both baselines.

Multi-GPU follow-up design requested by the user:

- keep readiness single-GPU on isolated GPU 5;
- before Stage A, add process-isolated Stage-2 workers on GPUs 4/6/7;
- keep Stage-1 on GPU 5;
- normalize each candidate latency against a strict-FP16 reference measured on
  that same physical GPU UUID;
- preserve deterministic ranking, uniqueness, backfill and per-candidate cache
  signatures at the coordinator.

--- Round 5 completed: 2026-07-14 17:56:09 CST ---

## Round 6 - E67 fail-closed result and four-GPU Stage-2 process pool

Starting point: commit `a28b95b`, branch
`feature/heal-compress-h800-sync-4090`. The required H800 fix remained an
ancestor and no command targeted the H800 branch.

Second fresh readiness:

- output: `outputs/4090_ga_explicit_qdq_readiness_20260714_030135/`;
- strict FP16 passed 200/200 with zero skips and mAP 0.724576;
- E67 preserved the 67 INT8 / 3 FP16 canonical profile and accepted static
  topology, but the engine inspector proved `/Concat_9` was FP32/mixed;
- the precise failure was
  `concat_merge_not_compatible:/Concat_9:FP32_or_mixed`;
- strong typing, common-scale graph lowering and an isolated Force-FP16 plugin
  were investigated; all full-graph alternatives were rejected after
  reproducible TensorRT crashes and all experimental source was removed;
- `READY_FOR_GA = false`, so generation 0 and all 300/500-frame candidate work
  remain unstarted.

Implemented multi-GPU search infrastructure:

- `search/orchestration/stage2_process_pool.py`: persistent one-process-per-GPU
  coordinator, atomic queues, ordered collection, timeout/crash propagation,
  exact candidate cache and cleanup;
- `search/stage2/candidate_worker.py`: one formal context/evaluator per GPU,
  strict startup preflight, same-GPU FP32 AP/FP16 latency references and
  fail-closed task output;
- `search/orchestration/generation_stage2.py`: concurrent ranked waves with
  deterministic uniqueness/backfill/winner semantics;
- `search/orchestration/lidar_pyramid_search.py`: process-pool lifecycle and
  candidate dispatch integration;
- `search/integration/lidar_pyramid_context.py`,
  `search/stage2/lidar_pyramid_real_evaluator.py`, and
  `search/orchestration/budget_final.py`: exact controller-PID allowlist
  propagation;
- `search/configs/lidar_pyramid_4090_ga_stage_a.yaml`: GPU 4/5/6/7 worker pool.

Verification and live evidence:

- 18 focused multi-GPU tests passed;
- 312 formal search/package tests passed with 7 known warnings;
- broad diagnostic suite: 801 passed, 4 unrelated pre-existing failures;
- live four-worker startup smoke passed at
  `outputs/4090_stage2_pool_startup_smoke_20260714_043413/`;
- GPUs 4/5/6/7 matched their expected UUIDs, each worker initialized one formal
  context, and all processes exited without residue;
- `docs/codex_handoffs/4090-ga-multigpu-stage2-integration.md` contains the
  detailed scheduling, cache, isolation and file-level design.

Current gates:

- multi-GPU framework implemented: yes;
- fresh E67 production merge gate: failed;
- `READY_FOR_GA = false`;
- Stage A started: no;
- Stage B allowed: no.

--- Round 6 completed: 2026-07-14 19:20:45 CST ---

## Round 7 - strongly typed plugin and full-E67 structural closure

Starting point: commit `2a23ebe`, branch
`feature/heal-compress-h800-sync-4090`. The required H800 fix remained an
ancestor and no command targeted the H800 branch.

Implemented production changes:

- added a deterministic typed-graph pass for explicit canonical FP16/FP32
  Casts, QDQ dtype closure, output precision contracts, residual/concat,
  functional MatMul, GridSample and floating elementwise paths;
- added a production strongly typed builder mode that rejects all weak
  precision flags and constraints;
- routed formal Stage-2 through `typed_qdq.onnx` only;
- included typed mode and fixed plugin boundary in context and cache identity;
- added separate FP16/FP32 readiness configs on physical GPU4;
- made Stage A fail before GPU/model initialization while the selected plugin
  boundary remains unset.

Fresh plugin minimum-graph evidence on GPU4:

- FP16 parser/build/serialize/deserialize passed, inspector Half/Half, parity
  max absolute error 0;
- FP32 parser/build/serialize/deserialize passed, inspector Float/Float, parity
  max absolute error 0;
- INT8 plugin boundary is rejected;
- plugin SHA256 is
  `5a5224f15831cb0712f945f623b4e252d0b617a28451de57afc98ab118d9855b`;
- the existing V2 dynamic plugin is compatible with TensorRT 10.9 strongly
  typed networks, so its C++ implementation did not require modification.

Full E67 structural diagnostic:

- both FP16-boundary and FP32-boundary typed graphs have 269 explicit Casts,
  zero unresolved tensor dtypes and unchanged 67/3 canonical coverage;
- both engines built and deserialized fresh with `--stronglyTyped`;
- both canonical precision audits report 67 INT8 / 3 FP16 / 0 unresolved;
- all 20 merge contracts pass in both engines, including `/Concat_9`;
- FP16 and FP32 plugin inspector boundaries match their ONNX contracts.

Verification:

- 333 formal/search/strongly-typed tests passed with 7 known environment
  warnings;
- all modified Python files compile;
- `git diff --check` passes;
- generated binaries and model artifacts remain ignored.

Current gate:

- `STRONGLY_TYPED_PLUGIN_FP16_PASS = true`;
- `STRONGLY_TYPED_PLUGIN_FP32_PASS = true`;
- `SELECTED_PLUGIN_BOUNDARY = NONE`;
- `STRONGLY_TYPED_E67_PASS = false`;
- `READY_FOR_GA = false`;
- Stage A started: no.

The structural E67 builds are diagnostic because they used an already-created
local 4090 QDQ graph. The next round must commit this code, then fresh-run the
strict FP16 reference and both E67 boundaries with new calibration/build
artifacts, 10-frame smoke and fixed 200-frame evaluation before selecting the
production boundary.

--- Round 7 completed: 2026-07-15 01:09:38 CST ---

## Round 8 - first post-commit readiness failure and ConvTranspose dtype closure

Starting point: commit `fa9b542`, branch
`feature/heal-compress-h800-sync-4090`. Local HEAD matched the remote branch,
the worktree was clean, and the required H800 fix remained an ancestor.

Fresh 4090 evidence:

- rebuilt the formal PointPillarScatterTRT binary for SM89 with TensorRT 10.9
  and CUDA 11.8; the binary used by this run has SHA256
  `91aec743dca383151b995a60d004cd254ce115ed16152873c1f46754bb15022d`;
- ran on physical GPU4, UUID
  `GPU-166702d7-bf30-18e0-83ff-83b316d37c0e`;
- diagnostic output is isolated at
  `outputs/4090_strongly_typed_e67_fp16_readiness_20260714_101615/`.

Fail-closed result:

- strict FP16 typed parsing failed because canonical ConvTranspose activation
  was Half while its Float weight initializer was not Cast;
- the exact TensorRT error named
  `__canonical__pyramid_backbone_deblocks_0_0__ConvTranspose__call00061`;
- the E67 branch independently completed a fresh train200 entropy calibration,
  strongly typed build/deserialization and 200/200 evaluation with zero skips;
- its canonical realization was 67 INT8 / 3 FP16 / 0 unknown, all merge audits
  passed, and mAP was 0.652240, but it is diagnostic only because the strict
  FP16 reference failed in the same run;
- `READY_FOR_GA = false`; neither multi-GPU Top-5 nor Stage A started.

Minimal TDD fix:

- added a regression graph with a Float ConvTranspose initializer and an FP16
  canonical contract; it failed before the implementation change;
- `quantization/precision/typed_graph.py` now includes ConvTranspose activation,
  weight and optional bias in the explicit floating compute-input dtype closure;
- `tests/test_strongly_typed_qdq_graph.py` verifies both activation and weight
  inputs are produced by FP16 Cast nodes;
- strongly typed focused regression: 21 passed;
- an attempted full historical-suite collection in `modelopt` was invalidated
  by 16 pre-existing imports of unavailable `opencood.tools.compression`; no
  collected test result from that invocation is counted as passing evidence.

All artifacts from the failed run remain diagnostic. The next readiness run
must start in a new timestamp directory from the committed fix.

--- Round 8 completed: 2026-07-15 01:28:00 CST ---

## Round 9 - strongly typed E67 readiness accepted, FP32 scatter selected

Accepted code commit: `34ebd4d`, branch
`feature/heal-compress-h800-sync-4090`. The H800 fix remains an ancestor and
no H800 branch command was issued.

Fresh same-GPU acceptance on physical GPU4:

- strict FP16 with FP16 scatter: 200/200, zero skips, mAP 0.725406;
- E67 with FP16 scatter: canonical 67/3/0, 200/200, zero skips, mAP 0.651251,
  forward p50/p90/p95 4.0027/4.2952/4.8178 ms;
- E67 with FP32 scatter: canonical 67/3/0, 200/200, zero skips, mAP 0.650970,
  forward p50/p90/p95 3.9881/4.1067/4.4139 ms;
- both E67 variants used the same train200 tensor-manifest hash and fixed
  200-frame validation-manifest hash;
- both passed typed ONNX/Inspector agreement, semantic QDQ, per-channel weight,
  EntropyCalibration2 lineage, precision realization and all merge contracts;
- PointPillarScatterTRT remained floating, with Half/Half versus Float/Float
  Inspector I/O and no adjacent QDQ.

Production decision:

- selected scatter boundary: FP32;
- mAP delta versus FP16 scatter: -0.000281;
- p50/p90/p95 all improved, so the predefined latency tie-break selects FP32;
- `search/configs/lidar_pyramid_4090_ga_stage_a.yaml` now fixes `fp32` and the
  config regression test enforces it;
- the plugin boundary is not a GA gene and does not change canonical 67/3.

Verification:

- focused selected-config/strongly-typed suite: 23 passed;
- broad formal environment: 823 passed, 4 known unrelated historical failures;
- exact known-failure deselection: 823 passed, 4 deselected;
- no generated model, plugin, calibration or engine artifact is tracked.

Current gates:

- `STRONGLY_TYPED_PLUGIN_FP16_PASS = true`;
- `STRONGLY_TYPED_PLUGIN_FP32_PASS = true`;
- `SELECTED_PLUGIN_BOUNDARY = FP32`;
- `STRONGLY_TYPED_E67_PASS = true`;
- `READY_FOR_GA = true`;
- four-GPU Top-5 smoke: not started at this anchor;
- Stage A: not started at this anchor;
- Stage B: not allowed in this task.

--- Round 9 completed: 2026-07-15 01:54:00 CST ---

## Round 10 - production-sized four-GPU Stage-2 smoke configuration

Starting point: commit `dc72dba`, with strongly typed readiness accepted and
the production PointPillarScatterTRT boundary fixed to FP32.

Added
`search/configs/lidar_pyramid_4090_ga_stage2_multigpu_smoke.yaml` for the
required live gate before Stage A:

- Stage-1 remains on GPU5 and retains the production population sizes
  1024/512/512;
- the smoke executes exactly one GA generation rather than shrinking the
  candidate population;
- Stage-1 admission remains target BOPS 0.21 +/- 0.005 with real repair,
  uniqueness and backfill;
- each generation still requires five unique deployable candidates;
- persistent Stage-2 workers use physical GPUs 4/5/6/7 and the selected FP32
  scatter boundary;
- every worker lazily builds same-GPU strict FP32 AP and strict FP16 latency
  references before evaluating its candidate;
- every candidate still performs physical prune replay, typed ONNX, explicit
  QDQ, fresh train200 calibration, strongly typed engine build and evaluation;
- smoke evaluation uses 10 warmup frames, resets latency, then measures 10
  frames; budget-final evaluation is disabled for this pre-Stage-A gate;
- external artifact/cache reuse is disabled.

`tests/test_search_large_population.py` now pins the smoke's full Stage-1
sizes, one-generation scope, Top-5, BOPS interval, four-GPU assignment, FP32
boundary, 10/10 evaluation and empty budget-final block.

Verification:

- smoke config/process-pool/worker/backfill focused suite: 15 passed;
- this anchor contains configuration and tests only; no live candidate result
  is claimed yet;
- next action is the live four-GPU Top-5 run followed by serial replay of the
  same five phenotypes.

--- Round 10 completed: 2026-07-15 02:01:00 CST ---

## Round 11 - four-GPU smoke fail-closed and strict-FP32 protected exception fix

Live failed-run evidence:

`outputs/4090_ga_qdq_stage2_multigpu_smoke_20260714_105857/`

The run used commit `fb8d285`, initialized four persistent workers on physical
GPUs 4/5/6/7, completed a production-sized 1024-candidate Stage-1 generation,
and found 24 unique, legal, genuinely-compressed candidates in the target BOPS
interval. The first four candidates were dispatched concurrently.

All four workers successfully built strongly typed strict FP32 and strict FP16
reference engines. Candidate deployment was then stopped before physical
pruning because the strict FP32 accuracy-reference validator rejected every
worker with `ordinary_weighted_layer_realized_fp16`. Backfill correctly tried
all 24 ranked candidates, selected zero, wrote the failure report, and closed
all workers with return code zero. No failed candidate entered winner scoring.

Four-card Inspector and canonical evidence identified one deterministic layer:

- each strict FP32 engine realized 69 FP32 and 1 FP16 canonical compute entry;
- the only FP16 row was
  `pyramid_backbone.functional_affine_grid_matmul`, represented by the stable
  canonical `pyramid_backbone_fun_*__MatMulGroup__` marker;
- canonical realization passed with INT8=0, FP16=1 and unresolved=0;
- this parameter-free functional layer is the existing production-protected
  FP16 exception, not an ordinary weighted fallback.

Minimal fail-closed fix:

- `search/baselines/original_engines.py` classifies the stable functional
  MatMulGroup marker separately from ordinary FP16 rows;
- strict FP32 allows it only when canonical realization simultaneously reports
  passed, zero INT8, zero unresolved and an exact matching FP16 count;
- any other FP16 row, more protected rows than declared, or a failed/missing
  canonical contract still fails;
- the report now exposes `protected_functional_fp16_count`,
  `allowed_protected_fp16_count` and `ordinary_weighted_fp16_count`.

Tests and direct replay:

- added passing and adversarial tests for verified/unverified functional FP16;
- all baseline validation tests: 8 passed;
- related strongly typed/baseline/process-pool/config suite: 52 passed;
- all four failed-run engine Inspectors re-audited as
  `69 FP32 / 1 protected FP16 / 0 ordinary FP16`, with empty issues;
- modified Python files compile and `git diff --check` passes.

This fix changes Stage-2 admission semantics, so the failed smoke directory is
diagnostic only. The four-GPU smoke must restart from generation 0 in a new
timestamp directory after the fix is committed.

--- Round 11 completed: 2026-07-15 02:07:00 CST ---

## Round 12 - four-GPU deployment proven; near-zero AP admission closed

Live generation-0 evidence:

`outputs/4090_ga_qdq_stage2_multigpu_smoke_20260714_110751/`

The run used commit `f1c8d17` and completed the production-sized Stage-1
population plus real Stage-2 deployment on persistent workers bound to physical
GPUs 4/5/6/7. It produced 24 unique legal Stage-1 candidates and dispatched
candidate builds in concurrent four-GPU waves. Every dispatched candidate used
an independent physical-prune replay, typed ONNX, explicit QDQ, train200
calibration, strongly typed engine build/deserialization and 10-frame smoke
evaluation. Each worker also built and validated its same-GPU strict FP32 and
strict FP16 references; all worker processes exited cleanly.

The run attempted eight ranked candidates before recording five nominal Top-5
rows:

- two candidates failed closed because realized BOPS was below 0.205;
- one admitted candidate produced mAP 0.334393;
- four admitted candidates produced mAP 0.000000, 0.000000, 0.031073 and
  0.033898;
- the same-GPU strict FP32 reference produced mAP 0.802498 on the fixed
  10-frame manifest.

This proves that four GPUs are being used concurrently for physical deployment,
calibration, engine construction and evaluation, but the resulting Top-5 is not
an acceptable smoke result. The smoke configuration had `max_map_drop: null`,
so the existing objective could not classify near-zero AP collapse as an
admission failure. This result is diagnostic only and must not be reused by
Stage A.

Minimal fail-closed configuration fix:

- `search/configs/lidar_pyramid_4090_ga_stage2_multigpu_smoke.yaml` now sets
  `max_map_drop: 0.75` for this pre-Stage-A diagnostic gate;
- relative to the observed strict FP32 reference, this requires candidate mAP
  of at least approximately 0.0525 and therefore rejects all four near-zero
  rows while preserving the 0.334393 diagnostic candidate;
- the existing Stage-2 objective emits `accuracy_hard_gate_failed`, assigns the
  failure score and continues ranked backfill;
- `search/configs/lidar_pyramid_4090_ga_stage_a.yaml` remains unchanged: this
  smoke-only threshold is not being silently promoted to the formal Stage-A/B
  acceptance policy.

Tests:

- config/process-pool/generation/round selection regression: 14 passed;
- objective/final-contract/worker/backfill regression: 19 passed;
- `tests/test_search_large_population.py` now pins the smoke accuracy gate;
- `git diff --check` passes.

The next live run must restart generation 0 in a new timestamp directory. It
must obtain five unique candidates that pass both realized BOPS and the
temporary AP gate before serial consistency replay can begin.

--- Round 12 completed: 2026-07-15 02:34:14 CST ---

## Round 13 - four-GPU smoke exhausted the generation and failed closed

Live run:

`outputs/4090_ga_qdq_stage2_multigpu_smoke_20260714_113749/`

The run used commit `a9ed2cc`, explicitly selected GPU5 for Stage-1, and
started one persistent Stage-2 worker on each physical GPU 4/5/6/7. All four
workers passed startup isolation. The 1024-candidate Stage-1 population yielded
24 unique legal, genuinely-compressed candidates in the legalized BOPS
interval, and ranked backfill deployed all 24 in six concurrent four-GPU waves.

Every attempted candidate produced independent physical-prune, typed ONNX,
explicit-QDQ, train200 EntropyCalibration2, strongly typed engine and realized
BOPS artifacts. Thirteen candidates entered the fixed 10-frame evaluation;
eleven were rejected before evaluation because their engine-realized BOPS was
outside `[0.205, 0.215]`.

Final admission distribution:

- `realized_BOPS_out_of_budget`: 11;
- `accuracy_hard_gate_failed`: 11;
- accepted: 2;
- accepted candidate 1: BOPS 0.211216, mAP 0.334322;
- accepted candidate 2: BOPS 0.208798, mAP 0.093220;
- required unique candidates: 5;
- terminal error: `insufficient_unique_deployable_candidates:2<5`.

The accuracy gate correctly rejected mAP values of 0.000000, 0.008475,
0.030367 and 0.033898 instead of letting collapse enter winner selection. The
two accepted candidates have distinct physical and deployment hashes. No
candidate was duplicated to fill the Top-5.

All candidate structural/deployment audits inspected so far passed, including
physical validation, production QDQ boundary, requested/legalized/realized
precision, merge realization and engine structure. This means the smoke
failure is not being relabeled as a weakly typed or audit-passing success. A
same-physical-model all-floating diagnostic is required next to separate
physical pruning damage from mixed-precision damage.

Worker cleanup:

- all four workers wrote `ready.json` and `stop` markers;
- no controller, candidate worker, calibration worker, TRT builder or
  evaluation worker remained after failure;
- GPUs 4/5/6/7 returned to 0% utilization and approximately 1-5 MiB memory.

Launch reproducibility fix:

- the first launch attempt exposed that `search.cli` used a default
  `--gpu-id auto` value that silently overwrote YAML `runtime.gpu_id: "5"`;
- this caused the controller to occupy GPU4 and the GPU4 worker correctly
  failed isolation instead of sharing an undeclared controller;
- `search/cli.py` now leaves the CLI value unset unless the flag is explicitly
  supplied, preserving the configured GPU; dry-run selection uses the merged
  runtime config;
- an explicit `--gpu-id` still overrides the YAML value;
- two CLI integration tests reproduce both paths; the preserve-config test
  failed before the fix and passes afterward;
- related CLI/config/worker/process-pool regression: 22 passed;
- modified Python files compile and `git diff --check` passes.

Current gates:

- strongly typed E67 readiness remains accepted;
- four-GPU Top-5 smoke pass: false;
- Stage A started: false;
- Stage B allowed: false.

--- Round 13 completed: 2026-07-15 03:39:16 CST ---

## Round 14 - isolate pruning/precision loss and close post-repair BOPS admission

Same-physical-model strongly typed FP32 diagnostics used the same fixed
10-frame manifest hash
`c827031ab82bb1925f48ada20dd64d0fa395dbf8d919d04137e95a5d3f35aee4`.
They changed only the quantization-group profile to FP32, disabled diagnostic
BOPS admission, and retained the production FP32 scatter boundary and protected
FP16 functional exceptions.

Collapsed mixed-precision candidate:

- source candidate: `489b3dc112aa637051e64db305f9dff66b0caeda13c5c0ce79a2f359fcce6eab`;
- mixed profile: 25 INT8 / 16 FP16 / 28 FP32, mAP 0.000000;
- all-floating weighted profile: 0 INT8 / 2 FP16 / 67 FP32, mAP 0.114810;
- physical hash remained exactly
  `ee0bb3eedf2b1d6a707b004d46f03dbf958892295694bd5a450154b9cd5d7c67`;
- parameter count remained 4,799,387;
- diagnostic output:
  `outputs/4090_ga_same_physical_fp32_bad_diagnostic_20260714_124339/`.

Accepted mixed-precision control candidate:

- source candidate: `06ee9dd7f6daaced3c9641a9cb1f390562c2a94bb08076a136f76ce051c2a4b4`;
- mixed profile: 21 INT8 / 29 FP16 / 19 FP32, mAP 0.334322;
- all-floating weighted profile: 0 INT8 / 2 FP16 / 67 FP32, mAP 0.516949;
- physical hash remained exactly
  `2e7166f254a5f7a4bdc69943c0eca28f9bb0133f474177abc79dd60f513877b7`;
- parameter count remained 4,787,303;
- diagnostic output:
  `outputs/4090_ga_same_physical_fp32_good_diagnostic_20260714_124353/`.

Both diagnostic engines passed physical, typed precision, merge, production
QDQ boundary and engine-structure audits with 10/10 evaluated and zero skips.
The comparison proves that physical pruning already causes material AP loss,
while the selected mixed-precision profiles add a further 0.114810 and 0.182627
mAP loss respectively. It does not identify a hidden weakly typed fallback or
merge audit failure.

Post-repair admission bug found in the live failure:

- raw candidates were filtered by proxy BOPS before repair;
- candidates were then repaired and rescored, including a fresh
  `bops_feasible` result;
- `select_repaired_stage2_topk` sorted and returned candidates even when the
  repaired rescore explicitly set `bops_feasible: false`;
- live ranks 17-23 already had infeasible F1 values above one million but still
  consumed seven physical deployments and strongly typed engine builds;
- this violates the required repair -> legalized BOPS gate -> Stage-2 order.

Minimal TDD fix:

- `search/stage1/repair_selection.py` now excludes only records whose repaired
  rescore explicitly reports `bops_feasible is False`;
- configurations without that field preserve their previous behavior;
- repair reports now distinguish `legal_repaired_phenotype_count`,
  `post_repair_bops_eligible_count`, and
  `post_repair_bops_ineligible_count`;
- `tests/test_search_final_contract.py` reproduces the prior admission and
  requires the infeasible repaired phenotype to be absent;
- focused final-contract/generation/config/manifest/process-pool regression:
  33 passed;
- modified Python files compile and `git diff --check` passes.

This changes Stage-2 candidate selection semantics. All earlier smoke outputs
remain diagnostic only; the four-GPU smoke must restart from generation 0 with
fresh artifacts after this fix is committed. Stage A remains stopped.

--- Round 14 completed: 2026-07-15 03:52:34 CST ---

## Round 15 - repaired-BOPS fresh smoke completed and Stage A blocked

Fresh run after commit `1f2c6a1`:

`outputs/4090_ga_qdq_stage2_multigpu_smoke_20260714_125402/`

The command intentionally omitted `--gpu-id`; the corrected CLI preserved YAML
GPU5 for Stage-1. Four persistent Stage-2 workers on GPUs 4/5/6/7 all passed
isolation. Their strict FP32 references produced identical mAP 0.802498 on
manifest hash
`c827031ab82bb1925f48ada20dd64d0fa395dbf8d919d04137e95a5d3f35aee4`.

The new repair admission counters were exercised live:

- Stage-1 proxy BOPS eligible: 24;
- legal repaired phenotypes: 24;
- post-repair BOPS eligible: 17;
- post-repair BOPS ineligible: 7;
- genuine compressed deployment candidates: 17.

Only the 17 repair-feasible records were deployed. This confirms the Round 14
fix removed seven known-infeasible engine builds before Stage-2. Final results:

- four failed engine-realized BOPS `[0.205, 0.215]`;
- eleven failed the temporary accuracy hard gate;
- two unique candidates passed all gates;
- accepted mAP/BOPS pairs: 0.334322/0.211216 and 0.101695/0.208798;
- terminal status: `insufficient_unique_deployable_candidates:2<5`.

All workers wrote ready and stop markers, no search/calibration/build/evaluation
process remained, and GPUs 4/5/6/7 returned idle. No serial consistency replay
was run because there were not five accepted candidates. Stage A did not start.

A standalone formal summary is recorded in
`docs/codex_handoffs/4090-ga-multigpu-stage2-smoke-report.md`.

Current gates:

- `STRONGLY_TYPED_E67_READY_FOR_GA = true`;
- `MULTIGPU_TOP5_SMOKE_PASS = false`;
- `STAGE_A_ALLOWED = false`;
- `STAGE_A_STARTED = false`;
- `STAGE_B_ALLOWED = false`.

The failure is closed. It is not permissible to relax the AP gate, widen the
BOPS interval, copy candidates, reuse invalid artifacts or start Stage A from
this state.

--- Round 15 completed: 2026-07-15 04:42:23 CST ---

## Round 16 - deterministic BOPS 0.21 anchor framework and feasibility planning

The branch gate was rechecked before work:

- branch: `feature/heal-compress-h800-sync-4090`;
- local and remote HEAD on entry: `36bf7a03e393db2750c75c9be52dd38e11c0832b`;
- worktree clean on entry;
- H800 fix `b862b3d8ad061bd12580776226c75f564918298d` remains an ancestor.

The production path remains typed ONNX with explicit Q/DQ and FP16/FP32 Cast,
`STRONGLY_TYPED`, and no weak precision fallback flags. PointPillarScatterTRT
is absent from all 69 quantization genes and canonical weighted BOPS rows. Per
the follow-up instruction, the study fixes the already accepted 4090
production scatter boundary to FP32 and does not repeat the FP16/FP32 boundary
comparison.

New deterministic anchor implementation:

- `search/anchors/bops_021.py` defines the A/B/C genotype contracts, exact
  weight-bit x activation-bit theory, requested/repair/realized profile hash
  audit, closed BOPS interval, BOPS transition decomposition, MAC-weighted
  quantization sensitivity and common-manifest audit;
- `search/anchors/runner.py` prepares one real model/Fisher/runtime-shape
  context, invokes no GA operator, performs deterministic legal-width scans,
  uses the existing physical pruner and strongly typed production evaluator,
  and runs the same engine for smoke10 then measured200;
- `search/configs/lidar_pyramid_4090_bops_021_anchors.yaml` fixes GPU4,
  train200 EntropyCalibration2, validation200, warmup20/reset, FP32 scatter,
  BOPS target 0.21 +/- 0.005 and no AP hard gate;
- `tests/test_bops_021_anchors.py` adds 15 focused anchor contract tests.

Two planning-only runs were made. They generated ignored artifacts and did
not export ONNX, calibrate, build engines, evaluate AP, or start GA.

The first run used the current formal GA pruning inventory. Its only dense
root is the 96-unit `pyramid_backbone.resnet.layer1.0.conv3` domain; even the
minimum legal width produced physical BOPS 0.240143. This proves that the
existing exposed domain cannot construct anchor B and explains why random
mixed candidates previously needed substantial INT8 freedom.

The second run used a study-only view of the same formal trace. It selected
only late local roots listed in the anchor config, while preserving early
backbone, detection-head roots and all grouped-conv rules. The trace contained
two matching dense domains, 256 units each:

- `shrink_conv.layers.0.double_conv.0`;
- `shrink_conv.layers.0.double_conv.2`.

This does not modify the tracer or formal Stage A search space. Deterministic
Fisher ordering found anchor B by pruning 72/256 channels in the first shrink
root (retained width 184):

- proxy BOPS retention: 0.2110143453;
- physical BOPS retention: 0.2110143492;
- implied global MAC retention: approximately 0.844057;
- precision genes: all FP16 before and after channel repair;
- the cached shrink gradients and empirical Fisher diagonals are nonzero, so
  no L1 fallback was used.

The no-prune mixed planning found anchor C with one quantization group:

- selected group: `pg_0141` (`pyramid_backbone.deblocks.2.0` in the canonical
  mapping);
- INT8 MAC share: 0.1971422583;
- theoretical/profile BOPS retention: 0.2130358274;
- CUDA proxy BOPS retention: 0.2130358219;
- all pruning genes remain keep.

Focused verification at this checkpoint:

- anchor contracts plus BOPS/realized-BOPS/generation regressions: 27 passed;
- anchor contracts plus BOPS/realized-BOPS regression after late-domain
  planning change: 19 passed;
- modified Python files compile;
- `git diff --check` passes.

The planning result identifies structural and precision profiles that are
budget-feasible, but no AP conclusion exists yet. The next step is a fresh run
from the committed HEAD: strict FP32 reference, then A/B/C smoke10 and common
200-frame measurement. Stage A remains stopped.

Current gates:

- `MULTIGPU_TOP5_SMOKE_PASS = false`;
- `STAGE_A_ALLOWED = false`;
- `STAGE_A_STARTED = false`;
- `STAGE_B_ALLOWED = false`.

--- Round 16 completed: 2026-07-15 05:22:02 CST ---

## Round 17 - BOPS 0.21 anchor measurements and search-region decision

The fresh formal run from committed anchor code `4fb44fe` completed on
physical RTX 4090 GPU4, UUID
`GPU-166702d7-bf30-18e0-83ff-83b316d37c0e`:

`outputs/4090_bops_021_anchor_study_20260714_142336/`

The run used one checkpoint, the fixed train200 EntropyCalibration2 manifest,
one validation200 manifest, the same FP32 scatter plugin boundary, warmup20
followed by latency reset, and the same strongly typed production evaluator.
The calibration tensor-manifest hash is
`eb56308111e20ad7c789b18a8860289fe7357474722dadc43c810868554e0ec5`;
the validation-manifest hash is
`6f601374e573a5ed7da61eeac07259c0eea34fb5ed72d9bf52265d40a02c9f16`.
All three anchors passed smoke10 and measured 200/200 with zero skips.

Same-run strict FP32 reference:

- AP03/AP05/AP07/mAP: 0.810864/0.770644/0.593668/0.725059;
- forward p50/p90/p95: 7.308166/7.568362/7.764058 ms.

Anchor A, all-keep and all-FP16:

- canonical profile 0 INT8 / 70 FP16 / 0 FP32;
- theoretical/proxy/physical/realized BOPS all exactly 0.25;
- AP03/AP05/AP07/mAP: 0.810845/0.770418/0.594003/0.725088;
- p50/p90/p95: 5.721614/5.859551/6.677257 ms.

Anchor B, all-FP16 plus one local physical prune:

- selected `shrink_conv.layers.0.double_conv.0`, retained 184/256 and pruned
  72 channels;
- MAC retention 0.844057, parameters 5,049,999;
- proxy/physical/realized BOPS:
  0.2110143453/0.2110143492/0.2110143492;
- AP03/AP05/AP07/mAP: 0.811078/0.770363/0.594374/0.725272;
- p50/p90/p95: 5.663902/7.020254/8.713158 ms;
- raw, repaired, requested and realized profiles are the same all-FP16 hash.

Anchor C, all-keep plus one INT8 precision group:

- selected `pg_0141`, `pyramid_backbone.deblocks.2.0`;
- INT8 MAC share 0.197142 and no physical pruning;
- canonical profile 1 INT8 / 69 FP16 / 0 FP32;
- proxy/physical/realized BOPS:
  0.2130358219/0.2130358219/0.2130358274;
- AP03/AP05/AP07/mAP: 0.810842/0.769888/0.595260/0.725330;
- p50/p90/p95: 3.918936/4.554928/5.580226 ms;
- raw, repaired, requested and realized profiles share hash
  `03da4b43731c91a5200336063780050f27fe667bd547f0845e54baf37c1fe1a1`.

All anchors have canonical count 70, unresolved count zero, separated pruning
and quantization namespaces, valid semantic QDQ, valid strongly typed profile,
valid per-channel weights, valid merge realization and no unexpected fallback.
`/Concat_9` is Half/Half in every engine. The scatter plugin remains outside
the quantization genes and BOPS rows.

Both B and C are observed low-damage Pareto points relative to the same-run
strict FP32 engine. C is the preferred primary mechanism: it preserves every
channel, has essentially the same measured accuracy as B, and lowers p50 by
1.744966 ms versus B. B remains a useful boundary/control and supplies a
Fisher channel order for future hybrid seeds. The proposed Stage A region is
C-centered with `R_MAC>=0.95`, INT8 MAC share approximately 0.14-0.22, only
late light pruning, no free FP32 gene, and deterministic B/C-derived seeds.
These settings and the proposed AP gate remain pending user approval.

Two report-level correctness gaps were closed with regression-first changes:

- the scatter dtype summary now accepts the actual three Float data inputs
  plus Int32 coordinates and reports the passing Float output as FP32;
- `LOW_DAMAGE_BOPS_021_PATH_IDENTIFIED` is now computed from observed relative
  mAP/AP07/p50 Pareto evidence, exact BOPS admission and 200/0 completion,
  without defining an absolute AP hard gate or unlocking Stage A.

The full evidence, deltas, hashes, BOPS decomposition, search recommendation
and reproduction commands are recorded in
`docs/codex_handoffs/4090-bops-021-anchor-study.md`.

Verification:

- 17 focused anchor tests passed;
- 95 related BOPS, repair, quantization-group, physical-pruning, strongly
  typed graph, merge, manifest and Stage-2 tests passed;
- modified Python files compile and `git diff --check` passes in the final
  delivery verification.

Current gates:

- `ANCHOR_FP16_BASELINE_PASS = true`;
- `ANCHOR_FP16_PRUNING_PASS = true`;
- `ANCHOR_MIXED_NO_PRUNE_PASS = true`;
- `LOW_DAMAGE_BOPS_021_PATH_IDENTIFIED = true`;
- `MULTIGPU_TOP5_SMOKE_PASS = false`;
- `STAGE_A_ALLOWED = false`;
- `STAGE_A_STARTED = false`;
- `STAGE_B_ALLOWED = false`.

--- Round 17 completed: 2026-07-15 05:51:19 CST ---

## Round 18 - constrained generation-0 multi-GPU Top-5 smoke

The user-approved C-centered Stage A search region is now implemented in the
new config `search/configs/lidar_pyramid_4090_ga_stage2_constrained_smoke.yaml`
and the new `search/constrained/` package. The hard region is `R_MAC>=0.95`,
full-canonical INT8 MAC share `[0.14,0.22]`, and BOPS `[0.205,0.215]`, with
FP16/INT8 genes only. Pruning is restricted to at most 20 low-Fisher channels
from the late `shrink_conv.layers.0.double_conv.0` local domain. Existing
pruning closure propagates the physical input change downstream; neither the
tracer nor the core dependency graph was modified.

The initial population is deterministic and constrained rather than fully
random: 358 C-neighborhood, 358 hybrid, 205 restored Anchor-B Fisher, and 103
constrained-fresh candidates. Exact Anchor C appears once. Seed construction
generated 5,117 unique proposals, of which 4,118 passed R_MAC, 1,979 passed
INT8-share, and 1,365 passed legalized BOPS before the required 1,024 unique
genotypes were accepted. Generation 0 processed all 1,024 candidates on GPU5;
all passed post-repair hard gates and represented 35 unique physical pruning
plans. Precision repair identity failures were zero.

The first live run exposed a smoke-manifest context bug and failed closed
before formal admission. Commit `8a938c4` gives smoke10 its own deterministic
manifest while restoring the fixed full200 context for formal evaluation. The
fresh accepted run is:

`outputs/4090_ga_qdq_stage2_constrained_smoke_20260714_164648/`

Persistent workers on physical GPUs 4/5/6/7 evaluated two four-wide batches.
All eight independently materialized, calibrated, built, and evaluated
candidates passed smoke10 and 200/200 with zero skips. Five ranked candidates
were admitted; the remaining three are explicitly recorded as passing
speculative batch deployments. All four workers returned 0, the pool stopped,
and no residual GPU process remained.

Accepted Top-5 summary:

| rank | family | width | R_MAC | INT8 groups | share | BOPS | AP07 | mAP | p50 ms |
|---:|---|---:|---:|---|---:|---:|---:|---:|---:|
| 1 | hybrid | 244/256 | 0.974010 | pg_0141 | 0.197142 | 0.206538 | 0.595563 | 0.725342 | 4.184 |
| 2 | hybrid | 248/256 | 0.982673 | pg_0141 | 0.197142 | 0.208704 | 0.595289 | 0.725149 | 4.412 |
| 3 | hybrid | 252/256 | 0.991337 | pg_0141 | 0.197142 | 0.210870 | 0.595010 | 0.725275 | 5.247 |
| 4 | C-neighborhood | 256/256 | 1.000000 | pg_0141 | 0.197142 | 0.213036 | 0.594496 | 0.725394 | 4.670 |
| 5 | constrained fresh | 248/256 | 0.982673 | pg_0077,pg_0086,pg_0141 | 0.206383 | 0.206971 | 0.597449 | 0.726520 | 5.240 |

All five have unique genotype, physical, deployment, and engine hashes. Raw,
repaired, requested, and realized precision hashes are identical within every
candidate; channel repair and TensorRT made no precision substitution.
Physical, strongly typed, Q/DQ, scale, merge, `/Concat_9`, and realized-BOPS
audits all pass. The common train200 frame-manifest hash is
`eb56308111e20ad7c789b18a8860289fe7357474722dadc43c810868554e0ec5` and
validation200 hash is
`6f601374e573a5ed7da61eeac07259c0eea34fb5ed72d9bf52265d40a02c9f16`.

The real hybrid observations show measured loss below the deliberately
conservative additive proxy prediction. They are saved for between-generation
calibration; no proxy term was changed online. The full source, population,
failure, GPU, metric, hash, delta, and reproduction evidence is recorded in
`docs/codex_handoffs/4090-constrained-top5-smoke-report.md`.

Final verification for the implementation and manifest fix is 184 related
search/constrained tests passed, all changed Python files compiled, and
`git diff --check` passed.

Current gates:

- `STRONGLY_TYPED_E67_READY_FOR_GA = true`;
- `ANCHOR_LOW_DAMAGE_PATH_IDENTIFIED = true`;
- `CONSTRAINED_SEARCH_SPACE_APPLIED = true`;
- `MULTIGPU_TOP5_SMOKE_PASS = true`;
- `STAGE_A_ALLOWED = true`;
- `STAGE_A_STARTED = false`;
- `STAGE_B_ALLOWED = false`.

Stage A was deliberately not launched. The next step requires explicit user
confirmation, then a fresh generation-0 start of the formal five-generation
Stage A run using this constrained space.

--- Round 18 completed: 2026-07-15 08:41:40 CST ---

## Round 19 - pruning-rate reachability contradiction audit

This round did not continue Taylor proxy development and did not launch GA,
Stage A, Stage B, QDQ, or TensorRT. It isolated the discrepancy between the
joint-Taylor anchor planner's observed `0.217360920` ceiling and the historical
first-order Taylor pruning sweep.

Historical 0.50/0.60/0.70 physical plans were replayed directly without
current repair. Their parameter counts exactly match the saved model objects:
2,723,599, 2,181,995, and 1,632,503 out of 5,464,791. Thus 0.60 is physically
proven at `0.600717575`. The 0.70 model is physically proven at `0.701268905`,
completed 500-frame evaluation, and collapsed to mAP `0.067819`; its failure is
accuracy, not structure.

Current parameter coverage contains 6,912 selected atoms in 24 scopes. The
per-unit dependency raw sum is 9,366,208 because closures overlap; the global
parameter-element union is 5,463,232. There are no missing selected parameter
slices. An independent legal-width solver and physical replay prove the same
current inventory and constraints can reach 586,919 parameters, or
`0.892599918` full-model pruning, while respecting `domain_cap=0.8` (observed
maximum domain rate `0.796875`). Synthetic forward and fixed real-data 10/10
smoke pass, although accuracy at this extreme collapses as expected.

Two implementation defects were closed with regression-first fixes. Grouped
raw requests now project to one conservative common legal width, and the anchor
planner explicitly records the maximum legal state so importance-order skew
cannot define reachability. Grouped virtual parameter counting now treats
axis-1 indices as physical-group-local and matches the materialized model.
Post-fix planner, independent solver, slice union, and physical count all agree
at `0.892599918` with zero parameter-count error.

Historical masks are not automatically current-rule legal: eight layer-2
grouped domains retain 12 channels per group, while the current allowlist omits
12. This is a real constraint-set difference, but it is not the cause of the
old `0.217361` ceiling. The complete evidence and limitations are in
`docs/codex_handoffs/4090-prune-rate-reachability-audit.md` and
`outputs/20260715_150636_prune_rate_reachability_audit/`.

Verification at this checkpoint: 11 new assertions and 68 related tests pass;
historical one-shot replay, current physical replay, and 10-frame GPU4 smoke
all pass. Full 1,789-frame validation was intentionally not run for the
extreme reachability masks.

Current execution gates remain unchanged:

- `GA_STARTED = false`;
- `STAGE_A_STARTED = false`;
- `STAGE_B_STARTED = false`;
- `TENSORRT_STARTED = false`.

--- Round 19 completed: 2026-07-16 06:00:51 CST ---

## Round 20 - GPU evaluation protocol and work-conserving Stage-2 dispatch

The legal-width anchor deployment matrix now uses a two-level evaluation
protocol. All precision/structure variants first run a common 50-frame GPU
screening manifest; only the strict baselines plus selected accuracy-boundary
and screening non-dominated rows proceed to the 1,789-frame full validation.
The old behavior of running the entire initial 45-engine matrix on full
validation is no longer the formal protocol.

Formal AP matching was moved from the NumPy/Shapely CPU implementation to the
OpenCOOD CUDA BEV-IoU extension. Production evaluation now requires
`ap_iou_backend=gpu`, rejects a silent CPU fallback, performs a CUDA self-IoU
smoke check, and records the backend in every result. Data loading uses eight
persistent workers with prefetching. A real strict-FP32 10-frame check passed
10/10 with zero skips; fixed-box CPU/GPU AP parity passed at IoU 0.3, 0.5, and
0.7.

During the new anchor screening run, the first seven completed deployments all
passed 50/50 with zero skips and recorded `gpu` AP matching plus eight loader
workers. The run directory is
`outputs/4090_legal_width_joint_taylor_anchor_sweep_20260716_034830/`.
This experiment remains in the screening/build stage; no screening metric has
been promoted to a full-validation or formal Pareto result.

The persistent four-GPU pool exposed a separate utilization defect: tasks were
submitted in fixed waves, so one slow INT8/build task prevented idle workers
from receiving the next candidate. A regression test reproduced the barrier.
`PersistentStage2ProcessPool.map_tasks` now dispatches one task per worker and
immediately replenishes whichever worker finishes, while retaining ordered
results, per-task timeouts, cache identity, and one persistent process per
physical GPU. The focused process-pool suite passes 3/3. The already-running
anchor controller loaded the earlier module implementation and therefore uses
the old wave behavior for both its current screening and any full-validation
pool created later by that same controller. Only a newly started controller
process, including the later GA run, loads the work-conserving scheduler.

No candidate encoding, legal-width decoder, Taylor ranking, physical pruning,
Q/DQ, precision realization, AP formula, or cache signature was relaxed by
this scheduling update.

--- Round 20 completed: 2026-07-16 19:16:42 CST ---

## Round 21 - approved greedy-first six-budget search specification

This round finalized the design contract for the next implementation and did
not start greedy search, GA, Stage A, Stage B, engine construction, or model
evaluation. The approved design is recorded in
`docs/superpowers/specs/2026-07-16-greedy-first-six-budget-joint-search-design.md`.

The formal Stage-1 objective removes the underresolved exponential Taylor
mapping and its `tau`. It will maximize the linear score
`J1 = -0.8 * (L_joint / L_scale) + 0.2 * R_prune`, where `L_joint` contains
weight pruning and retained-weight quantization Taylor perturbations but no
activation Taylor or SQNR term. `L_scale` will be the deterministic positive
nearest-rank P90 over unique accepted greedy-path states and legal anchors and
will remain immutable throughout GA.

The execution order is greedy first, followed by GA. Greedy searches the six
BOPS targets `0.05/0.10/0.15/0.20/0.25/0.30` independently using legal-width
and deployable precision actions. GA then uses three independent populations,
64 individuals and offspring per population, and 20 generations per budget.
The three populations merge their ranking each generation and deploy at most
one global unique Top-5 set.

BOPS is the only performance admission gate. The primary tolerance remains
`+/-0.005`; only when every primary candidate is exhausted may the nearest
legal candidate enter under an explicitly reported `+/-0.0075` tolerance, and
it may not belong to an adjacent budget's primary interval. No mAP or AP hard
gate is applied.

Generation candidate-count semantics are now explicit. Zero deployable
candidates fails the generation with a complete BOPS/admission funnel. One
candidate is audited and smoke-tested, skips the 500-frame comparison, and is
queued directly as the generation winner for common 1,789-frame validation.
Two through five candidates all run the fixed 500-frame evaluation and select
one winner using `F2 = mAP - 0.10 * (p50 / strict_FP32_p50)`. This score encodes
the accepted exchange that a 10% p50 reduction can compensate an absolute
0.01 mAP decrease.

Stage-2 will dynamically use RTX 4090 devices below 50% memory occupancy,
preferring the lowest-utilization devices. One signed strict-FP32 reference is
shared across workers. Official latency remains a later serial replay on one
quiet GPU, and official Pareto fronts require full-validation mAP plus that
formal latency.

Current state:

- `DESIGN_APPROVED = true`;
- `SPEC_WRITTEN = true`;
- `IMPLEMENTATION_STARTED = false`;
- `GREEDY_SEARCH_STARTED = false`;
- `GA_SEARCH_STARTED = false`;
- `STAGE_A_STARTED = false`;
- `STAGE_B_ALLOWED = false`.

--- Round 21 completed: 2026-07-17 01:11:38 CST ---

## Round 22 - implementation and experiment plans

The approved greedy-first six-budget design has been decomposed into three
ordered, independently reviewable execution plans:

1. `docs/superpowers/plans/2026-07-16-linear-joint-greedy-search.md` covers the
   signed fixed P90 scale, scalar/CUDA linear joint objective, two-level BOPS
   admission, legal action enumeration, bounded greedy search, and the
   six-budget proxy orchestration.
2. `docs/superpowers/plans/2026-07-16-three-seed-stage2-orchestration.md` covers
   the direct mAP/p50 F2 score, one shared strict-FP32 reference, deployment and
   evaluation protocol identities, low-occupancy GPU scheduling, variable
   per-generation Stage-2 candidate supply, three-seed generation merging,
   full validation, and formal latency.
3. `docs/superpowers/plans/2026-07-16-six-budget-real-experiment-pareto.md`
   covers the real greedy and GA runs, endpoint and generation-winner full
   validation, isolated latency replay, official Pareto artifacts, reports,
   and final branch verification.

The plans explicitly resolve the pre-calibration dependency: greedy uses a
`raw_joint_loss` diagnostic mapping because the greedy paths are themselves
the source of `L_scale`. After those paths and legal anchors produce the
read-only nearest-rank P90 artifact, formal GA switches to
`linear_fixed_scale` and maximizes
`J1 = -0.8 * (L_joint/L_scale) + 0.2 * R_prune`.

Stage-2 is split into build/audit/smoke and evaluation-only protocols. This is
required so a generation with exactly one deployable candidate can skip the
500-frame comparison while still joining the common 1,789-frame validation
queue. Protocol-specific cache keys prevent smoke, 500-frame, full-validation,
and formal-latency rows from being confused.

No implementation or experiment was started in this round. The existing
uncommitted coarse-boundary tau test remains excluded from the plan commit and
is explicitly scheduled for removal as obsolete WIP before the first RED test.

Current state:

- `DESIGN_APPROVED = true`;
- `IMPLEMENTATION_PLANS_COMPLETE = true`;
- `IMPLEMENTATION_STARTED = false`;
- `GREEDY_SEARCH_STARTED = false`;
- `GA_SEARCH_STARTED = false`;
- `STAGE_A_STARTED = false`;
- `STAGE_B_ALLOWED = false`.

--- Round 22 completed: 2026-07-17 01:45:40 CST ---

## Round 23 - fixed-scale linear proxy and six-budget greedy implementation

The first approved implementation plan is complete. This round implemented
the Stage-1 and proxy-only greedy framework; it did not run a real greedy
experiment, build an engine, evaluate mAP, start GA, or start Stage A/B.

The formal joint objective is now available as
`J1 = -0.8 * (L_joint / L_scale) + 0.2 * R_prune`. `L_joint` remains the
weight-pruning plus retained-weight-quantization first/empirical-Fisher
second-order Taylor perturbation. Activation Taylor and SQNR contribute zero
to the formal objective. Generic GA code continues to minimize `F1=-J1`.
Scalar and torch-batched implementations use float64 accumulation and have
matching output fields. Linear rows contain `L_scale` and
`normalized_joint_loss` and do not contain legacy `tau`, exponent, `S_task`,
or saturation fields.

`search/proxy/joint_loss_scale.py` calibrates `L_scale` as deterministic
nearest-rank P90 over unique positive finite phenotype losses, signs the full
member set, writes atomically, chmods the artifact read-only, and rejects
mapping, value, member, write-mode, or hash violations. Pre-calibration greedy
uses the explicit `raw_joint_loss` mapping; that mapping is now forbidden from
entering GA directly. Historical exponential artifacts remain available only
through an explicit `task_score_mapping: exponential` declaration.

The new BOPS admission package implements primary `+/-0.005` admission and
conditional `+/-0.0075` admission. Expanded candidates remain unpassed at the
single-value classifier level and can be admitted only after the set-level
selector proves primary supply is empty. Candidates in an adjacent budget's
primary interval are excluded. Every result preserves signed error, effective
tolerance, reason, complete funnel counts, and nearest misses.

The greedy implementation uses only adjacent legal compression actions:
one domain-width index decrement or one actually deployable precision-level
decrement. It never invokes normal repair. The bounded search minimizes
`delta_L_joint / delta_R_BOPS_saved`, then prefers larger structural pruning
gain and stable action ID. It retains eight frontier alternatives, evaluates
at most 4,096 unique states per budget, never evaluates one genotype twice,
never expands one state twice, and searches the complete bounded frontier for
a primary endpoint before considering expanded endpoints.

`search/orchestration/legal_width_greedy.py` runs targets
`0.05/0.10/0.15/0.20/0.25/0.30`, shares a proxy memo across budgets, writes
per-budget trace and endpoint artifacts, creates deployable
`ProxyCandidateRecord` endpoints, and freezes `joint_loss_scale.json`. Its
successor layers use the existing `Stage1ProxyEvaluator.evaluate_batch`, so a
CUDA-batched proxy is not degraded into thousands of scalar evaluations. The
new CLI flags are `--greedy-only` and `--joint-loss-scale`; greedy-only returns
before constructing the real evaluator or any Stage-2 worker.

Implementation commits pushed to the 4090 branch:

- `d40ed31` fixed linear joint-loss scale artifact;
- `7781f04` scalar and batched linear joint objective;
- `8cf16a9` auditable two-level BOPS admission;
- `834e405` legal greedy compression actions;
- `4b13111` bounded target-aware greedy search;
- `1f9c560` six-budget orchestration, CLI routing, and formal config.

The complete Plan-1 regression gate passed 72 tests in 2.98 seconds. All
modified Python files passed `py_compile`; `git diff --check` passed. Local and
remote HEAD were both `1f9c560` before this documentation commit, the worktree
was clean, and the required H800 ancestor check returned zero.

Current state:

- `LINEAR_JOINT_SCORE_IMPLEMENTED = true`;
- `TWO_LEVEL_BOPS_ADMISSION_IMPLEMENTED = true`;
- `GREEDY_PROXY_SEARCH_IMPLEMENTED = true`;
- `GREEDY_REAL_DEPLOYMENT_STARTED = false`;
- `GA_SEARCH_STARTED = false`;
- `STAGE_A_STARTED = false`;
- `STAGE_B_ALLOWED = false`.

--- Round 23 completed: 2026-07-17 02:29:18 CST ---

## Round 24 - three-seed six-budget Stage-2 orchestration implementation

The second approved implementation plan is complete and pushed. This round
implemented the real-experiment orchestration, cache identities, GPU
scheduling, variable candidate-count semantics, generation-winner full
validation, isolated formal latency, and Pareto plumbing. It did not execute
the six-budget greedy experiment, the three-seed GA, candidate engine builds,
500-frame evaluation, 1,789-frame validation, Stage A, or Stage B.

Formal Stage-2 now maximizes the direct measured score
`F2 = mAP - 0.10 * (candidate_p50 / strict_FP32_p50)`. It has no mAP, AP@0.7,
or max-drop hard gate. Frame-count, zero-skip, finite-output, BOPS, physical
structure, strongly typed precision identity, Q/DQ, and merge audits remain
fail-closed. One signed strict-FP32 engine/mAP/p50 reference is built once and
shared by all candidate workers; candidate workers are forbidden from
building private references.

Stage-2 work is split into explicit `reference_strict_fp32`, `build_smoke`,
`evaluate_500`, `full_validation`, and `formal_latency` protocols. Pool reuse
is keyed by mandatory protocol-specific `task_cache_key`, while
`candidate_hash` is lineage only. `DeploymentRegistry` separates engine
identity from evaluation-protocol identity and appends every
budget/seed/generation reference instead of overwriting it. Build/smoke and
evaluation-only evaluator methods ensure 500-frame and full-validation tasks
reuse an existing engine rather than rebuilding it.

The Stage-2 scheduler samples all configured RTX 4090 devices, excludes cards
above 50% memory occupancy, includes exactly-50% cards, orders eligible cards
by utilization and memory fraction, and records foreign processes without
terminating them. An empty configured GPU allowlist means all eight cards are
considered. No eligible device produces an explicit pending failure instead
of selecting an overloaded card. Formal latency uses a stricter independent
selection: no compute process, low utilization, one worker, and one GPU UUID.

Per-generation Stage-2 now has the approved count semantics. Zero BOPS-
admissible candidates write the complete primary/expanded funnel without
building. A sole deployable candidate passes build/smoke/audits, skips the
500-frame comparison, and enters the common full-validation winner queue.
Two through five candidates require 500/500, zero skips, and finite formal F2;
the maximum F2 wins. Primary `+/-0.005` candidates are exhausted before an
expanded-only `+/-0.0075` candidate may be used. Physical and realized BOPS
are checked again against the currently active interval. No candidate is
copied to fill Top-5.

The formal GA configuration is
`search/configs/lidar_pyramid_4090_joint_six_budget_ga.yaml`. It fixes targets
`0.05/0.10/0.15/0.20/0.25/0.30`, three independent seeds, population and
offspring 64, and 20 generations. The three populations evolve independently;
their records are merged once per generation, deduplicated by phenotype, and
ranked by descending J1 before one global Stage-2 decision. Initial
populations retain original/anchor/greedy endpoint seeds and legal adjacent
width neighbors, while seed-specific RNG ordering prevents three identical
initial populations. The configuration requires a CLI-provided read-only
`--joint-loss-scale`; activation Taylor and SQNR have zero formal objective
weight, and no AP hard gate is present.

Full validation operates only on unique greedy terminal endpoints and unique
generation winners. Repeated deployments retain all budget/generation
lineage but run 1,789 frames once. After parallel workers close, formal latency
replays strict FP32 first and then every unique full-validation candidate on
one isolated GPU. Budget winners maximize full-validation
`mAP - 0.10 * formal_p50/strict_FP32_formal_p50`. Official mAP-BOPS,
mAP-parameter-retention, and mAP-formal-p50 fronts accept only successful
full-validation rows joined to successful formal latency; screening metrics
and latency proxies cannot enter the official fronts. Greedy points have a
separate plot marker.

Implementation commits pushed in this round:

- `5d5d954` direct measured mAP/latency Stage-2 score;
- `3a9e489` deployment/evaluation protocol split and cache registry;
- `22eff11` low-occupancy multi-GPU scheduler;
- `c32bb0c` zero/one/two-to-five generation semantics;
- `da452a4` three-seed six-budget GA orchestration;
- `24c1bcb` generation-winner full validation and formal latency;
- `28e9dae` formal evaluator test-fixture initialization.

The complete Plan-2 regression gate passed 83 tests in 2.93 seconds. All
changed Python files passed `py_compile`; `git diff --check` passed. The one
initial failure was an `object.__new__` test fixture missing the already
mandatory `evaluation_num_workers=8` and `ap_iou_backend=gpu`; the fixture was
corrected without weakening production constraints. Local and remote HEAD
were both `28e9dae` before this documentation commit, and the required H800
ancestor check returned zero.

Current state:

- `THREE_SEED_GA_ORCHESTRATION_IMPLEMENTED = true`;
- `VARIABLE_STAGE2_SUPPLY_IMPLEMENTED = true`;
- `SINGLE_CANDIDATE_500_SKIP_IMPLEMENTED = true`;
- `SHARED_FP32_REFERENCE_IMPLEMENTED = true`;
- `FORMAL_FULL_VALIDATION_ORCHESTRATION_IMPLEMENTED = true`;
- `FORMAL_LATENCY_ORCHESTRATION_IMPLEMENTED = true`;
- `PARETO_REPORTING_IMPLEMENTED = true`;
- `REAL_SEARCH_STARTED = false`;
- `GREEDY_SEARCH_STARTED = false`;
- `GA_SEARCH_STARTED = false`;
- `STAGE_A_ALLOWED = false`;
- `STAGE_A_STARTED = false`;
- `STAGE_B_ALLOWED = false`.

--- Round 24 completed: 2026-07-17 03:28:18 CST ---

## Round 25 - six-budget greedy entry gate and real-run blocker fixes

Plan-3 execution started inline on the only permitted 4090 branch. The entry
audit is preserved at
`outputs/20260717_033025_six_budget_entry_audit/entry_audit.json`. Entry HEAD
was `c1ec729`; branch, remote divergence, clean worktree, and required H800
ancestor checks passed. PyTorch is `2.0.1+cu118`, TensorRT Python is
`10.9.0.34` when loaded with the formal TensorRT library path, and the local
SM89 PointPillarScatterTRT plugin SHA256 is
`91aec743dca383151b995a60d004cd254ce115ed16152873c1f46754bb15022d`.
The complete pre-experiment gate initially passed 111 tests.

The first fresh proxy run,
`outputs/4090_legal_width_greedy_six_budget_20260716_123446`, failed before
candidate scoring because the precision legalizer treated explanatory
protection reasons on `pg_0000` and `pg_0146` as deployment fallbacks even
though both requested and legalized precisions were FP16. A RED test proved
the behavior. Commit `5b182e7` now records no fallback when a protected group
requests its fixed default precision, while an external request for another
precision still produces a fail-closed fallback. The focused regression
passed 15 tests and the expanded gate passed 118 tests before the commit was
pushed.

The second fresh proxy run,
`outputs/4090_legal_width_greedy_six_budget_20260716_124119`, passed Fisher,
legal-width inventory, and deterministic Taylor-ranking construction, then
evaluated the full configured 4,096-state bound for every budget. All six
targets were correctly reported infeasible rather than admitted out of band,
and scale calibration failed closed because there was no accepted positive
path. Direct trace audit showed that beam width eight exhausted the state
budget at depths `{0: 1, 1: 101, 2: 1137, 3: 2857}`; the minimum proxy BOPS
retention reached only `0.5788060426712036`. This is a search-depth supply
failure, not evidence that the requested BOPS budgets are structurally
unreachable.

The approved iterative `delta L / delta BOPS` comparator is therefore set to
one deterministic path (`frontier_size: 1`) while retaining the 4,096 unique
state hard bound. This lets the same bound cover deep compression actions
instead of spending it on shallow beam combinations. Greedy metrics also drop
the fully expanded phenotype payload because genotype plus fixed decoder
reconstruct it exactly; candidate and genotype hashes remain in every state.
This removes the cause of approximately 3.4 GB per-budget trace files without
changing proxy values, candidate identity, BOPS admission, or endpoint
deployment. RED-to-GREEN tests for the compact trace contract and formal
single-path configuration passed, with the related suite at 22 tests.

Current state:

- `ENTRY_AUDIT_PASS = true`;
- `PRE_EXPERIMENT_TEST_GATE_PASS = true`;
- `PROTECTED_FIXED_PRECISION_LEGALIZER_FIXED = true`;
- `GREEDY_DEPTH_SUPPLY_FIX_APPLIED = true`;
- `GREEDY_ENDPOINT_DEPLOYMENT_STARTED = false`;
- `GA_SEARCH_STARTED = false`;
- `STAGE_A_STARTED = false`;
- `STAGE_B_ALLOWED = false`.

--- Round 25 completed: 2026-07-17 04:05:13 CST ---

## Round 26 - bounded greedy state-limit correction

The third fresh greedy run,
`outputs/4090_legal_width_greedy_six_budget_20260716_130654`, verified that
the single-path comparator and compact trace changes work as intended. The
first budget trace fell from approximately 3.4 GB to 38 MB while retaining
all scalar proxy metrics, genotype lineage, parent hashes, and action IDs.
No deployment or evaluation worker was started.

The run also provided a direct bound diagnosis. With 4,096 unique evaluated
states, the deterministic path executed 46 compression actions and reached
`R_BOPS=0.5263378620147705`, `R_prune=0.1585839238865676`, and
`L_joint_raw=9.00602388724472e-05`. It then exhausted the state limit before
any requested budget interval. Therefore all six endpoint records remained
explicitly infeasible and scale calibration failed closed. The observed
roughly 100 remaining legal actions per step means the original limit cannot
cover the approximately one hundred or more sequential precision/width
actions needed for the lowest budgets.

The formal greedy configuration now uses `max_expansions: 16384` with
`frontier_size: 1`. This is still a deterministic bounded search and does not
change the primary `+/-0.005`, conditional `+/-0.0075`, legal-width,
precision-legality, joint Taylor, or BOPS definitions. A RED-to-GREEN config
contract test fixes this state limit so a future edit cannot silently return
to the empirically insufficient 4,096-state bound. The focused greedy suite
passed 18 tests before the full gate and fresh rerun.

Current state:

- `GREEDY_TRACE_COMPACTION_VERIFIED = true`;
- `GREEDY_4096_STATE_LIMIT_SUFFICIENT = false`;
- `GREEDY_16384_STATE_LIMIT_APPLIED = true`;
- `GREEDY_REAL_ENDPOINTS_AVAILABLE = false`;
- `GREEDY_ENDPOINT_DEPLOYMENT_STARTED = false`;
- `GA_SEARCH_STARTED = false`;
- `STAGE_A_STARTED = false`;
- `STAGE_B_ALLOWED = false`.

--- Round 26 completed: 2026-07-17 04:14:35 CST ---

## Round 27 - successful six-budget proxy endpoints and deployment resume route

The fourth fresh proxy run,
`outputs/4090_legal_width_greedy_six_budget_20260716_131554`, completed on
entry commit `49770d6`. All six budgets produced primary-tolerance endpoints;
none used the conditional expanded interval. Actual proxy BOPS retentions for
targets `0.05/0.10/0.15/0.20/0.25/0.30` are respectively
`0.0549569651`, `0.1048461348`, `0.1540474743`, `0.2048732042`,
`0.2548551559`, and `0.3042219281`. The corresponding structural parameter
prune rates are `0.3618260973`, `0.2134800764`, `0.2134800764`,
`0.2134800764`, `0.2101328303`, and `0.2096431501`. All six genotype,
precision, phenotype, and candidate identities are unique; they contain four
unique physical structure hashes, and normal repair was never invoked.

The frozen linear scale is read-only mode `0444`, has value
`0.013404028918218002`, uses nearest-rank P90 over 206 unique positive greedy
path states, and has semantic hash
`fa9243cca1b09d42d92c8c23ce7e1e4e76ce0f09392e0f8ac9bb766a890915fd`.
Its file SHA256 is
`c0b273837408ff19650f707916140909007668a938493b72bc9c5260f3c1cefb`.
The artifact contains no tau field. Commands, resolved config, Fisher
manifest, checkpoint/context identity, legal-width inventory, width-space
hash, and canonical ranking manifest are present in the run directory.

A missing real-experiment route was found before deployment: generic
`--stage2-only` required explicit candidate files and would have recomputed
Fisher instead of reading greedy terminal endpoints. The runner now handles
`--resume RUN --stage2-only` for a greedy-enabled config before context or
Fisher construction. It loads only `greedy/budget_*_endpoint.json`, selects
all GPUs satisfying the 50-percent memory policy, builds one shared strict
FP32 reference, deduplicates terminal deployments, runs build/smoke followed
by GPU-IoU 1,789-frame validation with eight DataLoader workers, and closes
all workers in a `finally` path. The greedy config now carries explicit
Stage-2, full-validation, and multi-GPU protocol sections. Focused runner,
process-pool, physical, strongly typed, Q/DQ, merge, and realized-BOPS
regressions passed 46 tests.

Current state:

- `GREEDY_PROXY_SEARCH_COMPLETE = true`;
- `GREEDY_PRIMARY_ENDPOINT_COUNT = 6`;
- `GREEDY_EXPANDED_ENDPOINT_COUNT = 0`;
- `GREEDY_UNIQUE_PHENOTYPE_COUNT = 6`;
- `GREEDY_UNIQUE_PHYSICAL_STRUCTURE_COUNT = 4`;
- `LINEAR_SCALE_FROZEN = true`;
- `GREEDY_STAGE2_RESUME_ROUTE_IMPLEMENTED = true`;
- `GREEDY_ENDPOINT_DEPLOYMENT_STARTED = false`;
- `GA_SEARCH_STARTED = false`;
- `STAGE_A_STARTED = false`;
- `STAGE_B_ALLOWED = false`.

--- Round 27 completed: 2026-07-17 04:40:07 CST ---

## Round 28 - greedy endpoint real deployment and full validation

The six primary greedy endpoints were built and evaluated through the real
production path. One shared strict-FP32 reference was built on GPU 6. Endpoint
build-smoke tasks ran on GPUs 6/7/1/5/0/4, and full-validation tasks ran on
GPUs 2/3/6/7/1/5. All six endpoints passed 10/10 smoke and 1,789/1,789 full
validation with zero skips. Evaluation used the common manifest hash
`f53bc4717da33ddde0e1d4d8078e1ed5cc71f0ed063d000f50cee8f016d264cb`,
GPU IoU postprocessing, eight DataLoader workers, and warmup 20 followed by
latency reset.

The strongly typed build, physical structure, frozen pruning plan, semantic
QDQ, per-channel weight scale, EntropyCalibration2 lineage, merge realization,
`/Concat_9`, precision identity, engine deserialize, and realized BOPS audits
passed for every endpoint. No nonterminal greedy candidate was built. Build
count, unique endpoint count, and unique deployment count are all six. All
candidate workers, calibration workers, and `trtexec` processes exited.

Full-validation mAP for budgets `0.05/0.10/0.15/0.20/0.25/0.30` is
`0.7041011985`, `0.7366048020`, `0.7367924802`, `0.7367551673`,
`0.7367133415`, and `0.7367757633`. The shared strict-FP32 reference mAP is
`0.7365732236`. The parallel p50 values are screening-only and range from
`4.3749` to `7.4552` ms; they are not eligible for the official latency
Pareto. Complete identities, AP values, realized BOPS, parameter retention,
precision counts, and hashes are recorded in
`docs/codex_handoffs/4090-greedy-six-budget-search-report.md`.

Current state:

- `GREEDY_PROXY_SEARCH_COMPLETE = true`;
- `GREEDY_ENGINE_BUILD_SUCCESS_COUNT = 6`;
- `GREEDY_FULL_VALIDATION_SUCCESS_COUNT = 6`;
- `GREEDY_FORMAL_LATENCY_COMPLETE = false`;
- `GA_SEARCH_STARTED = false`;
- `STAGE_A_STARTED = false`;
- `STAGE_B_ALLOWED = false`.

--- Round 28 completed: 2026-07-17 05:02:13 CST ---

## Round 29 - immutable greedy evidence binding for the six-budget GA

The formal GA entry can now bind the already completed greedy run with
`--greedy-run`. The CLI resolves and hashes both
`greedy/greedy_summary.json` and
`greedy_full_validation/greedy_full_validation.json`, writes their paths and
SHA256 values into runtime provenance, and passes separate endpoint and
full-validation manifests to the runner. Missing files fail closed before
search starts.

The full-validation loader independently rechecks endpoint coverage,
successful-count consistency, 1,789 evaluated frames, zero skips, finite mAP,
requested/realized precision identity, and the existence of every serialized
engine. For the accepted run
`outputs/4090_legal_width_greedy_six_budget_20260716_131554`, all six endpoint
rows passed. Its full-validation manifest SHA256 is
`022f507477c7fd1d69f2aeefa10a893889a34a2b02d16f6281e119b714e40dac`.
The new GA run will therefore schedule zero greedy build tasks and zero greedy
full-validation tasks while still using those engines in the final formal
latency and Pareto comparison. Generation winners remain subject to the
original fresh full-validation protocol.

Modified implementation and contract files:

- `search/cli.py`: `--greedy-run` binding and immutable provenance;
- `search/orchestration/legal_width_stage2.py`: strict external evidence
  validation and normalized frame fields;
- `search/orchestration/lidar_pyramid_search.py`: reuse branch before
  generation-winner full validation;
- `search/configs/lidar_pyramid_4090_joint_six_budget_ga.yaml`: explicit
  external full-validation manifest field;
- `tests/test_generation_winner_full_validation.py` and
  `tests/test_legal_width_greedy_orchestration.py`: RED-to-GREEN reuse,
  zero-current-task, CLI, and provenance contracts.

The focused tests passed 12/12. The combined greedy, legal-width, three-seed,
Stage-2, strongly typed, QDQ, merge, realized-BOPS, full-validation, formal
latency, and Pareto regression gate passed 163/163 in 4.60 seconds. The three
modified Python files passed `py_compile`, and `git diff --check` passed.

Current state:

- `GREEDY_EXTERNAL_EVIDENCE_BOUND = true`;
- `GREEDY_CURRENT_RUN_BUILD_TASK_COUNT = 0`;
- `GREEDY_CURRENT_RUN_FULL_VALIDATION_TASK_COUNT = 0`;
- `GREEDY_FULL_VALIDATION_SUCCESS_COUNT = 6`;
- `GA_SEARCH_STARTED = false`;
- `STAGE_A_STARTED = false`;
- `STAGE_B_ALLOWED = false`.

--- Round 29 completed: 2026-07-17 05:10:32 CST ---

## Round 30 - fail-closed Stage-2 artifact retention before the GA restart

The first formal GA attempt is preserved at
`outputs/4090_joint_six_budget_ga_20260716_141230`. It completed Fisher
collection, legal-width inventory, fixed first/second-order rankings, one
shared strict-FP32 reference, and all 60 Stage-1 generations for the 0.05
budget (three seeds times 20 generations). The first generation then started
five real Stage-2 candidates on the persistent multi-GPU pool. No candidate
engine or Stage-2 result completed before the run was stopped, so none of
this attempt is accepted as formal search evidence.

The stop was deliberate and fail closed. The filesystem had approximately
123 GiB free. The five in-progress candidate directories had already reached
approximately 962 MiB; each candidate carried about 109 MiB in reproducible
ONNX and PTH intermediates before its engine was available. Retaining those
intermediates for up to 600 per-generation candidates would consume about 65
GiB by itself, in addition to engines, proxy archives, reports, and final
evaluation artifacts. The run was stopped before a predictable out-of-space
failure. All controller, worker, calibration, evaluation, and TensorRT
processes exited, and the 3.2 GiB incomplete output remains intact.

The formal GA now applies a strict post-generation retention policy. Only
regular files ending in `.onnx` or `.pth` below that generation's `stage2/`
directory may be removed, and only after build/smoke and any required
500-frame evaluation have returned. Serialized engines, calibration caches,
typed/QDQ/merge/physical/precision audits, JSON/CSV identities, metrics, logs,
and failure records are retained. Symlinks are never followed. The code
records every removed relative path and byte count in
`stage2_artifact_retention.json`, verifies that the engine set is unchanged,
and rejects any configured suffix outside the `.onnx/.pth` allowlist.

Changed files:

- `search/orchestration/legal_width_six_budget_ga.py`: bounded retention and
  per-generation invocation;
- `search/configs/lidar_pyramid_4090_joint_six_budget_ga.yaml`: explicit
  enabled suffix allowlist and engine-preservation contract;
- `tests/test_three_seed_generation_merge.py`: RED-to-GREEN deletion and
  preservation contract plus formal-config assertions.

The focused test module passed 5/5. The complete current regression gate
passed 164/164 in 4.50 seconds; `py_compile` and `git diff --check` passed.

Current state:

- `FIRST_GA_ATTEMPT_ACCEPTED = false`;
- `FIRST_GA_ATTEMPT_STOP_REASON = predictable_disk_exhaustion`;
- `STAGE2_REPRODUCIBLE_ARTIFACT_RETENTION_IMPLEMENTED = true`;
- `STAGE2_ENGINE_RETENTION_REQUIRED = true`;
- `FORMAL_GA_RESTART_REQUIRED = true`;
- `STAGE_A_STARTED = false`;
- `STAGE_B_ALLOWED = false`.

--- Round 30 completed: 2026-07-17 05:30:21 CST ---
