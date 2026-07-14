# 4090 explicit-QDQ GA readiness report

## Decision

`READY_FOR_GA = false`

Stage A has not started. Stage B is not allowed. This is a fail-closed interim
readiness result: the fresh 4090 plugin lifecycle gate passes, but every local
4090 currently has a foreign long-running compute process. The strict FP16 and
matched E67-ENT 10-frame/200-frame evaluations therefore have not been run and
no latency or AP result has been accepted.

## Branch and source gate

- branch: `feature/heal-compress-h800-sync-4090`;
- required source commit: `b862b3d8ad061bd12580776226c75f564918298d`;
- required commit is an ancestor of this branch: yes, exit code 0;
- H800 binary, ONNX, engine, calibration cache, timing cache and latency result reused: no;
- H800 branch modified or pushed: no.

## Fresh 4090 plugin and engine lifecycle

The production plugin was rebuilt with a clean CMake build using the local
`modelopt` CUDA 11.8 toolchain, TensorRT 10.9, and `CMAKE_CUDA_ARCHITECTURES=89`.

- plugin: `quantization/plugins/pointpillar_scatter_trt/build/libpointpillar_scatter_trt.so`;
- plugin SHA256: `f5fd5b17cfe5f560452f1cf6f37b695c825fd02b263061ed04140895d24f24c1`;
- plugin size: 89,168 bytes;
- TensorRT registry creator: `PointPillarScatterTRT`, version `1`, namespace empty;
- TensorRT: `10.9.0.34`;
- CUDA runtime used by modelopt: `11.8`;
- GPU architecture: NVIDIA GeForce RTX 4090, SM 8.9.

A fresh minimal PointPillarScatter ONNX was built on physical GPU 1 only to
test serialization and deserialization. It was not benchmarked and supplies no
latency evidence.

- ignored artifact directory: `outputs/4090_ga_qdq_readiness_20260714_014405/plugin_gate/`;
- serialized engine size: 3,588 bytes;
- engine SHA256: `2851187fa2096c8d33d2f7f042e96fd2d5d1a5bb7ed09c3e0295cf2eb9aa35d6`;
- explicit plugin-before-deserialize: passed;
- deserialized I/O tensors: 3 inputs and 1 output;
- `trtexec --skipInference` build/serialize/deserialize: passed.

The production Stage-2 build worker now repeats this check for every formal
engine. A missing engine, missing plugin, or a `deserialize_cuda_engine` null
result closes the candidate with `engine_deserialize_failure` before precision,
BOPS, or AP scoring.

## GPU isolation gate

Fresh telemetry at the readiness attempt found one foreign compute process on
each physical GPU 0-7. The processes belong to user `guohongze`, had been alive
for approximately 13,400-13,600 seconds, and occupied approximately 3,896-7,376
MiB each. GPU 3 was observed at 29 percent utilization and GPU 6 at 89 percent;
instantaneous zero utilization on the other GPUs does not make their persistent
foreign contexts acceptable for latency collection.

The formal runner now records, before and after each 300/500-frame evaluation:

- physical index, UUID and device name;
- driver version, utilization, temperature, power and memory;
- compute PID, user, command, used memory and elapsed time;
- the allowed current search PID and all foreign PIDs.

Any foreign compute PID or unexplained utilization closes the run with
`gpu_competition_detected`. No external process was killed or modified.

## Production recipe evidence available before the runtime gate

Source and CPU regression tests still enforce the H800 production contract:

- 70 canonical compute entries, consisting of 69 parameterized weighted entries
  plus the protected functional affine-grid MatMul;
- matched E67 coverage of 67 INT8 and 3 FP16 canonical entries;
- the FP16 exceptions are pillar-VFE linear,
  `pyramid_backbone.single_head_2`, and
  `pyramid_backbone.functional_affine_grid_matmul`;
- the functional MatMul remains `mapped_but_protected_fp16`;
- semantic post-ReLU/post-merge QDQ boundaries, FP16 merge output constraints,
  per-output-channel symmetric weight quantization, scalar symmetric activation
  scales, and EntropyCalibration2 train200 lineage are covered by formal tests.

These are source/test facts, not substitutes for the pending fresh 4090 E67
engine and 200-frame realization audits. The all-keep topology hash has not yet
been measured locally and therefore has not been compared with
`2cb4cabc8d939a730e474f48c2bbacf001f0093ee81c21a6249fe3c4f9cf1c3c`.

## Tests

- Correct formal CPU environment (`univ2x-opt`): `272 passed`, 7 known
  ModelOpt/PyTorch compatibility/deprecation warnings.
- TensorRT-focused environment (`modelopt`): plugin registry load, minimal
  engine build and explicit deserialize passed; 32 focused orchestration/runtime
  tests passed.
- A diagnostic broad run inside `modelopt` produced `271 passed, 1 failed`
  because that environment intentionally has TensorRT but no Python
  `modelopt.torch.quantization` package. The identical suite was rerun in
  `univ2x-opt`, where the missing package is installed, and passed 272/272.
- Modified Python files: `python -m py_compile` passed.
- `git diff --check`: passed before documentation update and must be repeated at
  the commit gate.

## Pending readiness gates

The following required items remain incomplete and prevent GA admission:

1. wait for at least one isolated RTX 4090 and record a passing isolation report;
2. build strict FP16 all-keep and matched E67-ENT all-keep engines fresh;
3. run engine deserialize and fixed 10-frame smoke on both engines;
4. run both engines on the same deterministic 200-frame manifest with 200/200
   evaluated and zero skips;
5. emit the actual pruning/quantization namespace/crosswalk counters from the
   fresh formal export;
6. emit the actual 70-entry mapping, E67 67/3 realization, semantic boundary,
   per-channel scale, EntropyCalibration2, merge and fallback audits;
7. verify the local all-keep QDQ topology hash and explain any difference from
   the accepted H800 static topology hash;
8. rule out near-zero AP and shrink collapse on the fresh 4090 results.

Until all eight items pass, `READY_FOR_GA` remains false and no generation-0
population may be started.

--- Readiness attempt recorded: 2026-07-14 16:49:04 CST ---

## Shared-GPU authorization update

At 2026-07-14 17:34 CST the user explicitly authorized continuing on GPUs
which still have sufficient free memory. This supersedes only the requirement
that no foreign compute PID may exist. It does not waive telemetry or allow a
busy card:

- production defaults remain fail-closed for every non-4090 configuration;
- the 4090 readiness and Stage-A configs explicitly select physical GPU 1;
- `allow_foreign_gpu_processes: true` preserves and reports the foreign PID;
- `max_gpu_utilization_pct: 20` rejects the card whenever sampled utilization
  is above 20 percent;
- the same policy is applied before/after readiness, every 300-frame Stage-2
  evaluation, and every 500-frame budget-final evaluation.

GPU 1 passed the new authorization gate at the update point with 20,175 MiB
free and 0 percent sampled utilization. The known foreign server used 3,896
MiB and remained fully visible in the report. `READY_FOR_GA` remains false
until the fresh strict-FP16/E67 runtime gates themselves pass.

--- Shared-GPU policy recorded: 2026-07-14 17:34:28 CST ---

## Isolated GPU selection update

Before the shared-policy commit was pushed, the user reported that GPUs 4-7
had become free. A fresh process query confirmed that all four had zero compute
PIDs. The formal readiness and Stage-A configs were therefore changed to
physical GPU 5 with `allow_foreign_gpu_processes: false`; the shared mode remains
available in code but is not enabled for this experiment.

- GPU: physical 5;
- UUID: `GPU-d4b8342a-7038-f567-ce40-a4705b23854b`;
- free memory: 24,080 MiB;
- sampled utilization: 0 percent;
- compute PID count: 0;
- isolation mode: strict.

--- Isolated GPU 5 selected: 2026-07-14 17:36 CST ---

## First full readiness attempt and merge-inspector fix

Run directory:
`outputs/4090_ga_explicit_qdq_readiness_20260714_024246/`.

The first complete attempt remained fail-closed because strict FP16 reported
`concat_merge_not_compatible:/Concat_9:not_yet_verified`. The engine itself
passed build, serialization/deserialization, structure and precision checks.
TensorRT 10.9's compiler backend had lowered the concat into three
`__myl_Move` layers whose layer names and metadata did not contain `/Concat_9`;
their output tensors were all named `/Concat_9_output_0` and were all Half.
The merge audit searched only engine layer names, so this was a reproducible
inspector matching defect rather than an FP32 realization.

The E67 diagnostic completed before the run returned:

- evaluated/skipped: 200/0;
- AP@0.30: 0.758102;
- AP@0.50: 0.706666;
- AP@0.70: 0.484700;
- mAP: 0.649823;
- canonical precision: 67 INT8 / 3 FP16 / 0 FP32 / 0 unresolved;
- raw fused engine weighted precision: 65 INT8 / 3 FP16;
- Q/DQ topology hash:
  `2cb4cabc8d939a730e474f48c2bbacf001f0093ee81c21a6249fe3c4f9cf1c3c`;
- E67 merge audit: passed;
- fresh EntropyCalibration2 cache: generated on GPU 5.

The fix adds an exact match against TensorRT engine output tensor names when a
merge layer name is unavailable. It accepts the compiler-backend representation
only because all matched input/output formats are Half; FP32/mixed output still
fails. A focused test reproduces the three real `__myl_Move` layers. Applying
the fix to the preserved strict-FP16 layer-info changes only the inspector
verdict to FP16 and clears all merge issues.

Because formal audit code changed, no engine/evaluation from this attempt is
promoted. A fresh run from a new directory and new commit is required.

`READY_FOR_GA = false`

--- First full attempt analyzed: 2026-07-14 17:56:09 CST ---
