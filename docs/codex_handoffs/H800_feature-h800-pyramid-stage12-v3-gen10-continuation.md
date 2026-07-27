# H800 Pyramid Stage-1/Stage-2 GA generation-10 continuation

Branch: `feature/h800-pyramid-stage12-v3-gen10-continuation`  
Worktree: `/home/lixingfeng/UniAD_examine/heal_compress_h800_pyramid_stage12_v3_gen10_continuation`  
Base: `5f77bc38d718a4053e92f73ba7c453e0e912d9b3`

---

Timestamp: 2026-07-28 04:08 CST

## Development progress

- Extended the existing strict CNN formal-GA entrypoint to accept the original
  five-generation contract and a Pyramid-only ten-generation continuation.
- The original run did not serialize all 64 survivor genotypes or Python RNG
  state. The continuation therefore deterministically replays generations
  1--5 from the same seed, immutable search space and cached real Stage-2
  results, then executes new generations 6--10.
- Added a compact frozen generation-5 snapshot containing reports, budget
  summaries and generation 00--05 summaries. Engine, ONNX and calibration
  artifacts are not duplicated.
- Added SHA-256 fail-closed verification requiring every replayed generation
  00--05 summary to match its frozen snapshot before the ten-generation result
  is accepted.
- Made the strict GA generation contract follow `formal_gen5` or
  `formal_gen10`; seed 0, population/offspring/survivor 64, Stage-2 quota 5,
  no-repair semantics and the `J_struct_gate + J_WQ + J_AQ` proxy are unchanged.
- Targeted regression: 32 tests passed; compileall, py_compile and
  `git diff --check` passed.

## Runtime status

- The existing five-generation Pyramid root remains
  `/data/lxf/heal_data/outputs/h800_pyramid_sixbudget_ga_stage2_300_genwinner500_20260726_171505`.
- GPU launch is deferred until one permitted device is sufficiently available.
  CoBEVT currently owns GPU2 and GPU3 is occupied by an external training job.
- The continuation must use `--resume --generations 10` and preserve the
  original six-budget root; the new frozen snapshot records the prior result
  before final reports are updated.

---

Timestamp: 2026-07-28 04:21 CST

## Verification, cleanup and launch queue

- Moved the generation-00--05 replay verification ahead of final result and
  acceptance serialization. A divergent replay now fails closed before a
  ten-generation result can be accepted.
- Targeted GA/proxy regressions: 38 passed. `compileall`, `py_compile`, and
  `git diff --check` passed.
- Committed and pushed `bd13fd62`; local/remote divergence is `0 0`.
- Queued a GPU2-only launcher (`PID 3988956`) behind the active CoBEVT task.
  It performs read-only process polling and requires five consecutive GPU2
  compute-process-free samples before launch. It does not probe or occupy any
  other GPU.
- Permanently removed 260,117,397,504 bytes (242.25 GiB) of exact, audited
  failed/redundant artifacts. The output tree fell from 721 GiB to 479 GiB.
  Retained all reports, manifests, current winners, reusable caches and the
  224 GiB Pyramid generation-5 root required for deterministic continuation.
