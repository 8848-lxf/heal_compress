# H800 CNN GA two-tier evaluation handoff

Branch: `feature/h800-cnn-stage2-300-genwinner500`

Base: `ebce76cda66b22530655418111b845b5e318098f`

## 2026-07-26 12:05 PDT

- Preserved the previously requested five-generation Pyramid/DiscoNet/F-Cooper scheduler and the legal frontier/beam Greedy recovery.
- Added the two-tier real-evaluation contract for remaining/restarted budgets:
  - Stage-2 Top-5 screening: 300 validation frames, 100 warmup frames;
  - one winner per generation: 500 validation frames, 200 warmup frames;
  - final budget winner: fixed500 generation-winner pool versus the fixed500 Greedy anchor.
- Fixed500 validation reuses the screened engine through the model-family `reevaluate_existing_candidate_engine` path. It verifies source deployment and precision acceptance and does not rebuild the engine.
- Added ordered target-subset support so completed budgets remain untouched while only early/unstarted budgets are rerun in a new output root.
- Tests: 29 targeted tests passed; compileall, py_compile, and `git diff --check` passed.

--- 2026-07-26T12:05:00-07:00 ---
