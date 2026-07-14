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
