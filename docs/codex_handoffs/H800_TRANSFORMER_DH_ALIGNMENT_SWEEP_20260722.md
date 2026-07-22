# H800 Transformer unified Q/K/V d_h alignment sweep

## Round 1 — implementation and first real nonaligned gates

- Branch: `feature/h800-transformer-dh-alignment-sweep`
- Base commit: `6db0e483d61e6ae845a9eeb83632efe83616627d`
- Output: `/data/lxf/heal_data/outputs/h800_transformer_dh_alignment_20260722_021715`
- Toolchain: `/home/lixingfeng/miniconda3/envs/modelopt`, TensorRT 10.9.0.34, Conda CUDA 11.8/GCC 11.2, H800 SM90.
- Minimal SM90 extension gate compiled and ran successfully; no CPU fallback.

Implemented production code:

- `search/model_families/transformer/dh_candidate_grid.py`: actual-D0 dense grids, alignment classes, adjacent microbenchmark pairs.
- `search/model_families/transformer/dh_pruning_contract.py`: fail-closed unified Q/K/V head-local masks and nested-mask audit.
- `search/model_families/transformer/dh_physical_rewrite.py`: physical Q/K/V output-row and W_O input-column slicing, V2XViT HGT relation-axis slicing, arbitrary integer d_h, updated scale, and multi-family Phase-B rewrite.
- `search/model_families/transformer/dh_precision_profiles.py`: P32, F3-P16 and SQ1-P8 contracts; QK fixed to F32A32O32.
- `search/model_families/transformer/dh_alignment_audit.py`: EngineInspector tactic, Tensor Core, padding and fallback evidence. Generic CNN padding and valid cuBLAS GEMV tactics are not misclassified as d_h padding/fallback.
- `search/orchestration/lidar_transformer_dh_sweep.py`: true model inventory and one frozen train200 first-order Taylor ranking shared by every width.
- `search/orchestration/lidar_transformer_dh_build.py`: physical model, fresh ONNX, strongly typed/noTF32 TensorRT and fresh per-width calibration200 P8 builds.
- `search/orchestration/lidar_transformer_dh_evaluate.py`: manifest-locked smoke10/fixed50/fixed500, workers=8, GPU AP/IoU, warmup reset, zero-skip gate.
- `search/orchestration/lidar_transformer_dh_run_matrix.py`: resumable family-scoped serial build/evaluation queues safe for multi-GPU execution.
- `search/orchestration/lidar_transformer_dh_microbenchmark.py`: non-additive primitive micro-engines for projection/QK/AV/Out tactic diagnosis.
- `search/orchestration/lidar_transformer_dh_latency.py`: five-minute-isolated, baseline-replay full-engine latency protocol.
- `search/orchestration/lidar_transformer_dh_identity_parity.py`: D0 physical decomposition versus original PyTorch numerical parity.
- `search/reporting/transformer_dh_alignment.py`: 330-row evidence matrix, requested/realized aggregation, precision conflicts, Q/DQ inventory, deployment contract and evidence-gated root conclusion.
- `tests/test_transformer_dh_alignment.py`: 28 targeted tests currently pass.

Frozen inventory and ranking:

- CoBEVT: window H8/d32 and grid H8/d32.
- V2XViT: HGT agent-relation H8/d32; spatial window w16 H4/d64; w8 H8/d32; w4 H16/d16.
- CoBEVT Taylor train200 mean task loss: 0.6775579439; ranking hash `646e80ea...a5d`.
- V2XViT Taylor train200 mean task loss: 0.4503874976; ranking hash `f2e3e9e...ad8bad9`.
- Nested-mask audits passed. The w4/d16 family additionally schedules optional d15…8 only after its main d16 gate.

First real results:

- CoBEVT D0 P32 fixed500: window 0.6445782461; grid 0.6451126281.
- CoBEVT D0 F3-P16 fixed500: window 0.6448073020; grid 0.6448988187.
- V2XViT D0 P32 fixed500: agent 0.6581883210; window-w8 0.6576902055.
- V2XViT D0 F3-P16 fixed500: agent 0.6582439140; window-w8 0.6580112953.
- V2XViT D0 SQ1-P8 fixed500: agent 0.6578398228; window-w8 0.6577372179, 120 Q/DQ and zero precision conflicts.
- V2XViT window-w8 d31: physical rows/columns are truly 248 wide; P32 mAP 0.6581355699 and P16 mAP 0.6584579282 on fixed500.
- Existing d31 P32/P16/P8 engines re-audited as `EXACT_NONALIGNED`; no materialized padding and no precision fallback.

Active/resumable queues use GPU 0/2/4/7 respectively for CoBEVT window/grid and V2XViT agent/window-w8. Each card performs one candidate build/evaluation chain serially. GPU 6 is not used by this run. External processes are not killed. The queues can be resumed with `python -m search.orchestration.lidar_transformer_dh_run_matrix` and will verify completed structure/build/evaluation artifacts before reuse.

Not yet complete at this checkpoint:

- Remaining Phase-A widths, V2XViT w16/w4 families, Phase-B selected joint-family candidates, primitive microbenchmarks, D0 original-forward parity, and isolated formal latency.
- Formal latency remains evidence-gated: low-occupancy GPUs are valid for build/AP validation, but the formal label still requires a five-minute external-process-free window.
- No GA/Greedy/full1789 was run and no Pyramid path was modified.

---

Timestamp: 2026-07-22 18:19:30 CST — Round 1

---

## Round 2 — resumed persistent Phase-A queues

- Resumed from Codex session `019f578e-5ae0-7c40-b419-7cb79b5b3668` without replaying or deleting accepted artifacts.
- Re-ran the dedicated d_h suite: 30/30 passed.
- Re-ran the existing CoBEVT/V2XViT/Transformer regression suites: 190/190 passed.
- `compileall` and `git diff --check` passed.
- Restarted family-scoped, persistent serial queues on low-occupancy GPU 0/2/4/7. No external process was killed or modified.
- Main queues resume CoBEVT window/grid and V2XViT agent-relation/window-w8 from their accepted artifacts.
- Persistent disjoint-width follow-up queues cover V2XViT window-w16 d64...32 and window-w4 d16...8.
- After all Phase-A queues exit, persistent follow-up tasks run both D0 identity-parity checks and all six family primitive microbenchmarks.
- Formal latency remains intentionally unscheduled until an H800 satisfies the required five-minute external-process-free isolation gate.

At this checkpoint the resumed progress journals had advanced beyond the old interruption, including CoBEVT window d30 and V2XViT window-w8 d28. The background queues and their logs are rooted under the same output directory and do not depend on a live Codex terminal session.

---
