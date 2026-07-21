# CoBEVT Native F16A32 Tactic Pinning Design

## Goal

Audit and, where the installed TensorRT 10.9.0.34 runtime proves it possible, pin native FP16-operand TensorRT tactics with FP32 accumulation for the six real CoBEVT QK and six real AV shapes on RTX 4090 SM89.

## Evidence boundary

The implementation distinguishes F32A32O32, F16A32O32, F16A32O16, F16A16O16, and UNKNOWN_ACCUM. A Cast that materializes Float operands is classified as `F32A32_AFTER_MATERIALIZED_CAST`. A custom cuBLASLt plugin is retained only as historical reference and never counts as native TensorRT.

Only direct Level-A accumulator evidence can make a profile eligible for a formal search gene. Numeric fingerprints and opaque tactic names may produce Level-B or Level-C records, but cannot be upgraded by inference.

## Architecture

`sampleEditableTimingCache` is the source of truth for timing-cache key/value text formats. A small C++ helper, compiled with the modelopt Conda toolchain, builds strongly typed primitive QK/AV micrographs with `kEDITABLE_TIMING_CACHE`, emits profiling tables and serialized cache files, and updates one key to a selected tactic hash. Python orchestration binds each record to the exact graph signature, shape, GPU, TensorRT, builder flags, ONNX, and cache hashes.

The orchestration has four gates: (1) native micro-engine build and tactic evidence; (2) cache edit and three fresh deterministic rebuilds; (3) full-engine tactic/key preservation; and (4) smoke10/fixed50/fixed500 plus isolated formal latency. Missing or opaque evidence fails closed and produces a report without fabricating a native profile.

## Data flow

Existing six-block captures and primitive graph recipes are reused without changing checkpoint, masks, shapes, fixedK, manifests, or F3 non-target precision. Every candidate writes provenance, profiling log, layer inspector, cache hashes, tactic map, numerical fingerprint, and failure reason. Full-engine candidates reuse the accepted F3 exporter and only enter evaluation after exact native tactic and cache gates pass.

## Cache contract

An edited timing cache is valid only for the exact GPU architecture, TensorRT/CUDA versions, graph signature, shape/layout, builder flags, tactic source, workspace, and cache key. Rebuilds must select the requested tactic ID; this proves pinning, not accumulator semantics. A full-engine layer rename, fusion, key change, or tactic mismatch is `micro_pinning_not_preserved_in_full_engine`.

## Failure handling

If all tactics are F16A16 or opaque, the shape is `NO_F16A32_TACTIC` or `UNKNOWN`. Partial 6/6 coverage is never promoted to full native. Plugin fallback, F32 fallback, stale cache reuse, system nvcc, and changed shape/provenance are explicit failures.

## Verification

Tests cover contract separation, cache compatibility, Level-A/B/C policy, partial-profile rejection, output precision, and deterministic pinning. Existing CoBEVT tests remain a regression gate. No Pyramid process, GA, Stage-A/B, or 1789-frame validation is touched.
