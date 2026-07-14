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
