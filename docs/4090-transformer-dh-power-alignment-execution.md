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
