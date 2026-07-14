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
