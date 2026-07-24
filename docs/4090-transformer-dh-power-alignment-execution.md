# 4090 Transformer d_h power-alignment execution record

This document records additive RTX 4090 evidence for the H800 Transformer
head-dimension alignment experiment. It does not replace or modify H800
latency conclusions.

---

Completed: `2026-07-23T15:13:16Z`

## Priority P16/F3 matrix

- Platform: RTX 4090, SM89, TensorRT 10.9, CUDA 11.8.
- Precision contract: P16/F3 with FP16 projections, explicit FP32 Q/K
  recovery, F32A32O32 QK, and accepted FP16 AV/output boundaries.
- Accuracy: fixed500, 500 evaluated, zero skipped, GPU AP/IoU, workers=8.
- Formal latency: device-resident full-engine forward, CUDA event, 200 warmup,
  500 timed iterations per repeat, 5 repeats, baseline replay on the same GPU.
- GA, Greedy, full1789, H800 execution: not run by this experiment.

The priority queue completed 22 P16 structures. The exact `d_h` 8/16/32
subset contains 14 physical structures including one baseline per model.
Compact evidence is in
`docs/codex_handoffs/4090-transformer-dh-priority-p16-results.csv`.

## Initial conclusion

CoBEVT window Attention `d_h 32 -> 16` is the first candidate that satisfies
both fixed500 accuracy and same-profile full-engine latency requirements:

- baseline mAP: `0.644457442`;
- candidate mAP: `0.642115537`;
- delta mAP: `-0.002341905` (`SAFE` under the 0.003 threshold);
- initial baseline p50: `3.857312 ms`;
- initial candidate p50: `3.752768 ms`;
- initial p50 reduction: `2.710%`;
- initial speedup: `1.027858x`;
- baseline replay drift: `0.719%`;
- candidate repeat p50 CV: `0.0102%`.

Three independent fresh builds were then measured. Their candidate p50
reductions were `3.176%`, `3.086%`, and `3.399%`. All three exceed the formal
1% gate and none slows down. Engine hashes and tactic signatures differ across
builds, so the result is classified `BUILD_REPEAT_STABLE`. Detailed evidence is
in `docs/codex_handoffs/4090-transformer-dh-window16-build-stability.csv`.

The evidence therefore supports a platform-scoped statement: on RTX 4090,
SM89, and TensorRT 10.9, CoBEVT window `d_h=16` produces a stable full-engine
speedup under the same P16/F3 mixed-precision contract. This is not yet an H800
latency conclusion.

## Rejected or unresolved priority points

- CoBEVT grid `d_h=8` and window `d_h=8` are faster but accuracy-unsafe.
- CoBEVT grid `d_h=16` is latency-beneficial but accuracy-borderline.
- V2XViT agent relation `d_h=8/16` is much faster but accuracy-unsafe.
- V2XViT spatial-window candidates are mostly below the 1% latency gate; the
  W16/H4 `d_h=8` point barely exceeds it but is accuracy-unsafe.
- Joint structures remain pending after this priority conclusion.

---

Completed: `2026-07-23T20:09:00Z`

## P32/P16/P8 cross-profile and neighbor-control evidence

The 22 priority physical structures now have complete P32, P16, and P8
fixed500 and formal full-engine latency evidence: 66 structure/profile rows,
all with 500 evaluated frames and zero skipped frames. P8 used fresh
per-structure SQ1 calibration; no structure reused another structure's scale
or calibration artifact.

V2XViT agent-relation `d_h=24` retained accuracy under every contract and
showed a stable same-profile speedup:

- P32: mAP `0.656629011`, p50 `18.848640 ms`, speedup `1.344552x`;
- P16: mAP `0.656534773`, p50 `15.627264 ms`, speedup `1.404157x`;
- P8: mAP `0.656466355`, p50 `14.947328 ms`, speedup `1.415907x`.

Three independent fresh builds per profile confirmed p50 reductions of
`25.91%`-`26.00%` for P32, `28.79%`-`28.86%` for P16, and
`29.43%`-`29.56%` for P8. The engine and tactic hashes differ, so this is
build-stable performance rather than reuse of one favorable engine.

The mandatory nearby 4-aligned controls disprove a stronger alignment-only
claim. Across P32/P16/P8, `d_h=20` is faster than `d_h=24`, while `d_h=24`
is faster than `d_h=28`. Therefore `d_h=24` has a substantial physical
structure speedup but does not satisfy the rule that an 8-aligned point must
beat both `target-4` and `target+4` controls. The observed ordering is
monotonic with width and does not isolate an extra 8-alignment advantage.
Compact evidence is in
`docs/codex_handoffs/4090-transformer-dh-agent24-alignment-controls.csv`.

## Preserved failures

Two repeat-build attempts were intentionally retained as invalid environment
evidence. A direct worker invocation first leaked to the system compiler and
failed closed; a second manually assembled environment omitted the official
ModelOpt runtime-library path and caused `trtexec` SIGSEGV after ONNX parsing.
Using the repository's `build_engine_modelopt` wrapper resolved the issue.
These failures are infrastructure provenance, not structure failures.

---

Completed: `2026-07-24T16:11:23Z`

## Complete P32/P16/P8 matrix

The RTX 4090 collaboration run completed all 63 unique physical structures and
all three requested precision contracts. Every one of the 189 unique
structure/precision phenotypes passed engine build, realized-precision audit,
and fixed500 evaluation with 500 evaluated frames and zero skipped frames. The
195 report rows include the C0/V0 aliases of the two baseline structures; they
must not be interpreted as additional physical structures or engines.

Formal latency was measured on one RTX 4090 (`GPU-166702d7-bf30-18e0-83ff-83b316d37c0e`)
using device-resident inputs, CUDA events, 200 warmup iterations, five repeats
of 500 timed iterations, and a baseline replay around every candidate batch.
All 24 batches passed isolation and replay-drift checks. The maximum baseline
replay drift was `0.640769%`, below the 1% floor used by the latency gate.

Baseline fixed500 mAP and p50 were:

| model | profile | mAP | p50 ms | precision-only speedup vs P32 |
|---|---:|---:|---:|---:|
| CoBEVT | P32 | 0.644891152 | 5.288376 | 1.000000x |
| CoBEVT | P16 | 0.644457442 | 3.888080 | 1.360151x |
| CoBEVT | P8 | 0.643917808 | 3.212632 | 1.646119x |
| V2XViT | P32 | 0.658200598 | 25.381288 | 1.000000x |
| V2XViT | P16 | 0.658097274 | 21.943824 | 1.156648x |
| V2XViT | P8 | 0.658448156 | 21.162040 | 1.199378x |

P8 is the validated SQ1 SmoothQuant INT8 Q/K projection contract. It is not
FP8 and does not use native INT8 QK MatMul.

## Final alignment conclusion

Physical `d_h` pruning can produce stable full-engine speedup on this
RTX4090/SM89/TensorRT-10.9 platform, but higher-order alignment alone is not a
general speedup rule. Most exact-power and 8/16/32-aligned points either fail
the fixed500 accuracy threshold, fail the 1% same-profile latency gate, or do
not beat both nearby 4-aligned controls.

The only row that passes all five formal admission gates is CoBEVT window
Attention `d_h 32 -> 16` under P16/F3:

- fixed500 mAP `0.642115537`, delta `-0.002341905` versus the P16 baseline;
- formal p50 `3.761152 ms`, same-profile speedup `1.033747x`;
- faster than the available `d_h=12` and `d_h=20` controls;
- three fresh engine builds reduced p50 by `3.1758%`, `3.0863%`, and
  `3.3992%`;
- a joint structure containing window `d_h=16` remained latency-beneficial.

CoBEVT joint C1 (`grid/window=24/24`) remained accuracy-safe and
build-repeat-stable under P16 and P8, with same-profile speedups of `1.027836x`
and `1.027961x`. It remains experimental because the required single-family
neighbor-control evidence does not establish an independent alignment
advantage.

V2XViT agent `d_h=24` is accuracy-safe and build-repeat-stable under all three
profiles, with formal speedups of `1.345187x`, `1.404109x`, and `1.416939x`.
It remains experimental because `d_h=20` is faster than `d_h=24`, which is in
turn faster than `d_h=28`; that ordering supports width reduction, not an
independent 8-alignment advantage. No V2XViT row passed all search-admission
gates.

The full machine-readable evidence is under
`/data/lxf/heal_data/outputs/h800_transformer_dh_power_alignment_4090_20260723_051931/`.
The primary files are `power_alignment_contract.json`,
`power_alignment_single_family_fixed500.csv`,
`power_alignment_joint_fixed500.csv`,
`power_alignment_same_profile_speedup.csv`,
`power_alignment_neighbor_controls.csv`, and
`power_alignment_build_repeat_stability.csv`.

## Branch provenance

The experiment started with `origin/feature/heal-unified-search-h800` at
`3293f4ef953d0ecb5e58c8a92a2a94c7e45b2e7f`. At completion the remote branch
was `460764fdecf9ea6298667586c033a51aa4b71c7d`, indicating an external update
during the experiment. This worktree did not checkout, merge, cherry-pick, or
push the formal-search branch. All experiment changes remain on
`feature/h800-transformer-dh-alignment-sweep` and are marked as RTX 4090
evidence.
