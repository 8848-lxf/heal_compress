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
- Neighbor controls, P32, P8, and joint structures remain pending after this
  priority conclusion.

## Preserved failures

Two repeat-build attempts were intentionally retained as invalid environment
evidence. A direct worker invocation first leaked to the system compiler and
failed closed; a second manually assembled environment omitted the official
ModelOpt runtime-library path and caused `trtexec` SIGSEGV after ONNX parsing.
Using the repository's `build_engine_modelopt` wrapper resolved the issue.
These failures are infrastructure provenance, not structure failures.

