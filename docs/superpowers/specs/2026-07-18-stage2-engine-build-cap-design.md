# Stage-2 Engine Build Cap Design

## Decision

For every GA budget and generation, `topk_stage2: 5` is a hard cap on
calibration and TensorRT engine build attempts. It is not a target number of
successful engines. A failed build, smoke check, realized-precision audit, or
realized-BOPS audit consumes one of the five build slots and must not trigger a
replacement engine build.

## Admission Flow

1. Rank unique Stage-1 phenotypes by the configured joint proxy.
2. Apply the proxy BOPS interval before any physical work.
3. Materialize ranked candidates and compute physical BOPS before ONNX, QDQ,
   calibration, and TensorRT. Candidates that fail this preflight may be
   replaced from the ranked supply because they have not consumed an engine
   build slot.
4. Stop preflight selection after at most five candidates pass.
5. Build at most those five candidates once, concurrently where GPUs permit.
6. Keep post-build realized BOPS and precision checks as fail-closed audits,
   without build backfill.

## Generation Result Semantics

- Zero successful candidates: skip the generation. Record whether supply was
  exhausted at proxy BOPS admission, physical BOPS preflight, engine build,
  smoke, precision/merge/QDQ audit, or another explicit stage.
- One successful candidate: designate it as the generation winner, skip the
  500-frame comparison, and include it in final full-validation selection.
- Two to five successful candidates: evaluate every successful candidate on
  the fixed 500-frame manifest and choose the generation winner with the
  existing F2 rule.
- A failed 500-frame evaluation is recorded and never causes another engine
  build.

## Invariants

- `engine_build_attempt_count <= topk_stage2` for every generation.
- `calibration_attempt_count <= topk_stage2` for every generation.
- Physical-BOPS rejection occurs before calibration and engine build.
- Realized-BOPS mismatch remains fail closed but cannot cause build backfill.
- The BOPS interval and expanded-tolerance policy are unchanged.
- Candidate genotype, deterministic structure decoder, physical pruning,
  strongly typed QDQ deployment, and Stage-2 F2 scoring are unchanged.

## Recovery

The run `4090_joint_six_budget_ga_20260717_180315` was stopped during
`budget_015/generation_017` because it used success-count backfill. Its evidence
is retained as diagnostic history. Since Stage-2 selection semantics change,
the corrected six-budget search starts in a fresh timestamped directory.
