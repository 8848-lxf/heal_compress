# 4090 Stage-2 Engine Build Cap Handoff

## Status

- Branch: `feature/heal-compress-h800-sync-4090`
- Entry HEAD: `6d184ac0dc2c459cf713e33a3dae5698863c32ee`
- Implementation HEAD: `9bcec4c003145d731ba4e3ca17163cfa18625b59`
- Required H800 ancestor: `b862b3d8ad061bd12580776226c75f564918298d`
- H800 ancestor check: passed
- Old run active: false
- Corrected fresh run started: false at this documentation checkpoint

## Direct Finding

The stopped run implemented `topk_stage2: 5` as five successful deployment
results rather than five engine-build slots. `run_generation_stage2()` kept
submitting ranked candidates after engine-build, smoke, physical-BOPS, or
realized-BOPS failures. The BOPS=0.10 budget therefore built 120 engines:
100 nominal slots plus 17 replacements after smoke collapse and 3 replacements
after physical-BOPS rejection.

That behavior contradicted the accepted experiment contract. The run was
stopped without deleting artifacts at
`/data/lxf/heal_data/outputs/4090_joint_six_budget_ga_20260717_180315`.

## Stopped-Run Evidence

| budget | generations present | winners | production engines | calibration engines | recorded build/smoke failures |
|---:|---:|---:|---:|---:|---|
| 0.05 | 20 | 8 | 25 | 25 | 10 smoke collapse, 1 physical BOPS |
| 0.10 | 20 | 20 | 120 | 120 | 17 smoke collapse, 3 physical BOPS |
| 0.15 | 17 | 16 | 91 | 91 | 4 physical BOPS; generation 17 incomplete |
| 0.20 | 0 | 0 | 0 | 0 | not started |
| 0.25 | 0 | 0 | 0 | 0 | not started |
| 0.30 | 0 | 0 | 0 | 0 | not started |

The 0.10 count is the direct reproduction of the bug:

```text
20 generations * 5 build slots = 100
+ 17 smoke-collapse backfills
+ 3 physical-BOPS backfills
= 120 production engines
```

## Corrected Data Flow

The corrected generation path is:

```text
ranked Stage-1 phenotypes
-> proxy BOPS hard gate
-> physical materialization and planned physical-BOPS preflight
-> ranked preflight replacement without engine construction
-> select at most five physical-BOPS-admissible candidates
-> one capped calibration/TRT build wave, size <= 5
-> smoke/precision/QDQ/merge/realized-BOPS audits
-> no post-build backfill
-> 0/1/2-5 generation result semantics
```

Physical preflight may inspect more than five ranked phenotypes, but it does
not export ONNX, insert QDQ, calibrate, or build an engine. Once up to five
preflight candidates are chosen, the worker pool receives exactly one
`build_smoke` wave. Any failure consumes its build slot.

Generation decisions are now:

- zero successful deployments: skip the generation and record
  `generation_skip_reason`, `failure_stage_histogram`, and
  `failure_reason_histogram`;
- one successful deployment: skip the 500-frame comparison, make it the
  generation winner, and retain it for final full validation;
- two through five successful deployments: evaluate all on the fixed
  500-frame manifest and select the existing F2 winner;
- zero candidates at proxy BOPS admission are distinguished from zero at
  physical BOPS preflight and zero after capped engine build/smoke.

## Code Changes

- `search/orchestration/generation_stage2.py`
  - adds ranked physical preflight;
  - replaces success-count backfill with one capped build wave;
  - adds engine/preflight counters and explicit generation skip evidence.
- `search/orchestration/legal_width_six_budget_ga.py`
  - submits `physical_preflight` worker tasks before `build_smoke`;
  - keeps candidate artifact directories stable across both phases.
- `search/stage2/candidate_worker.py`
  - routes the new physical-only protocol without requiring an engine
    realized-precision hash.
- `search/stage2/lidar_pyramid_real_evaluator.py`
  - materializes the physical model and computes planned BOPS from physical
    runtime shapes and the legalized precision profile;
  - records `engine_build_started=false` in the preflight evidence.
- `tests/test_generation_stage2_candidate_counts.py`
  - proves build/smoke failures cannot trigger candidate 6 or 7;
  - proves physical preflight can scan ranked supply without exceeding five
    engine builds;
  - proves zero-candidate skip evidence is complete.
- `tests/test_search_stage2_candidate_worker.py`
  - proves physical preflight does not route through engine build.

## Verification

RED evidence reproduced four failures before implementation:

- existing code called build waves `[a,b,c,d,e]` and `[f,g]`;
- `physical_preflight_batch_fn` did not exist;
- `physical_preflight` was an unknown worker protocol.

GREEN evidence:

- focused generation/worker tests: `16 passed`;
- complete selected search/deployment regression: `245 passed`;
- warnings: 6 existing ModelOpt/dependency-version warnings;
- modified Python `py_compile`: passed;
- `git diff --check`: passed.

## Storage And Restart

No old outputs were deleted. At the checkpoint, `/data` had approximately
14 GiB available while `/` had approximately 305 GiB available. The corrected
fresh run must not reuse the stopped run directory. Its large artifact root is
therefore planned under `/var/tmp/lxf/heal_data/outputs`, while code and light
reports remain in the repository. Existing lossless completed-generation
hardlink compaction remains available for known byte-identical audit aliases.

## State

```text
PROXY_BOPS_GATE_BEFORE_ENGINE=true
PHYSICAL_BOPS_GATE_BEFORE_ENGINE=true
MAX_ENGINE_BUILD_ATTEMPTS_PER_GENERATION=5
POST_BUILD_BACKFILL_ENABLED=false
ZERO_CANDIDATE_GENERATION_SKIPPED=true
SINGLE_CANDIDATE_500FRAME_SKIPPED=true
TWO_TO_FIVE_CANDIDATES_EVALUATE_500=true
OLD_RUN_ACCEPTED=false
CORRECTED_RUN_STARTED=false
STAGE_A_STARTED=false
STAGE_B_ALLOWED=false
```

--- Round completed: 2026-07-18 21:57:18 CST ---
