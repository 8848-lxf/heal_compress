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
