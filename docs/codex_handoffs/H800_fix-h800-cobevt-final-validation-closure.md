# H800 CoBEVT final-validation closure

## 2026-07-30T09:47:14+08:00

- Branch: `fix/h800-cobevt-final-validation-closure`.
- Isolated worktree: `/home/lixingfeng/UniAD_examine/heal_compress_h800_cobevt_final_validation_fix`.
- Base commit: `2d2aef4cab4311819a4308eff894311c40cb7905` from the active CoBEVT J_AQ=0 gen5 campaign branch.
- The active campaign process and its source worktree were treated as read-only. No process signal was sent and no active output was overwritten.

### Root cause

- CoBEVT Stage-2 deployment itself is closed: the inspected 0.30 Greedy artifact has physical, engine, Q/DQ, precision, merge, Transformer FP32-island and functional-precision acceptance, with requested INT8 count equal to realized INT8 count and zero conflict/unmapped/fallback.
- The fixed500 evaluator deliberately returns metrics only. The old generation-winner validator expected deployment acceptance fields in that metrics-only payload and interpreted absent fields as `False`, producing a false `requested_realized_exact=false` failure after a successful engine evaluation.
- A separate metadata bug serialized all Greedy winners with the evaluator bound to the lowest target (0.05). This made the 0.30 anchor report a 0.254978 BOPS deviation even though its actual retention is 0.304978 and it is inside its own budget band.

### Code changes

- `search/ga/cnn_stage12_v3.py`
  - Added a versioned generation-winner validation schema so stale cached validation results are re-audited.
  - Added CoBEVT static Stage-2 acceptance reconstruction bound to the complete phenotype hash and the actual engine SHA256.
  - Required all immutable physical/QDQ/precision/merge/Transformer acceptance flags, exact requested/realized INT8 counts, zero conflict/unmapped/fallback, an engine stored inside the candidate artifact, and a matching on-disk SHA256.
  - Combined those immutable Stage-2 facts with fixed500 metrics rather than expecting the metrics worker to duplicate deployment metadata.
  - Preserved fail-closed behavior for missing evidence or any identity/hash/precision mismatch.
  - Recomputed serialized Greedy winner metrics with an evaluator bound to each winner's own target budget.
- `tests/test_cnn_stage12_v3_gen5.py`
  - Added target-specific Greedy anchor regression checks.
  - Added a metrics-only CoBEVT fixed500 closure test.
  - Added an engine-hash mismatch fail-closed test.

### Verification

- Actual 0.30 artifact replay, without rebuilding or using a GPU:
  - candidate `3fac89473cabb5cef9e9a8d27bdadd7fbf9ec0bfd33e0bfab6f0102ec52fe44f`;
  - actual `R_BOPS=0.30497841142578874`;
  - requested/realized INT8 `1/1`;
  - conflict/unmapped/fallback `0/0/0`;
  - static closure audit passed with no issue;
  - fixed500 B0/candidate mAP `0.6451518256/0.6450590798` and p50 `7.281743/5.037996 ms` already exist in the artifact.
- Targeted tests: 96 passed.
- Full pytest: 1081 passed, 0 failed.
- `python -m compileall -q search scripts tests`: passed.
- `python -m py_compile` for changed source and test: passed.
- `git diff --check`: passed.

### Safe continuation

- The currently running GPU7 campaign still imports the old source from `/home/lixingfeng/UniAD_examine/heal_compress_h800_cobevt_jaq0_sixbudget_gen5`; changing it in place would violate process isolation.
- After that process exits naturally, resume/revalidate the existing campaign root using this fix branch. Existing hash-matched engines and evaluations can be reused; no Greedy/GA search rerun is required solely for this schema repair.
- Do not accept validation artifacts written with the old schema as final; the patched validator automatically re-audits them.

---

## 2026-07-30T14:43:00+08:00

- Reconfirmed the active CoBEVT campaign remains on GPU7 at PID `508877` and
  still imports the pre-fix worktree.  No signal was sent and no active
  artifact was changed.
- Added `scripts/resume_cobevt_validation_after_active_search.sh`.  It waits
  read-only for both the active search and its previously queued legacy
  follow-on, verifies PID identities while they exist, then resumes the same
  output root with the fixed validation schema.
- The continuation requires six completed budgets and zero formal failures
  before starting the existing single-GPU repeat-3 P/Q evaluation on GPU7.
  It performs no search rerun when hash-matched Stage-2 artifacts are valid.

---
