# 4090 CoBEVT Native F16A32 Tactic Pinning

Experiment output:

`/data/lxf/heal_data/outputs/cobevt_native_f16a32_tactic_pinning_20260721_121057`

Platform: RTX 4090 (SM89), TensorRT 10.9.0.34, CUDA 11.8, strongly typed, TF32 disabled, fixedK29696.

## Result

TensorRT 10.9 has no public matrix-multiply accumulator dtype setter. Editable Timing Cache stores a tactic hash for an exact cache key; it does not store an independent dtype request.

The editable profiling table reported 60 candidates for each of six captured QK and six captured AV primitive shapes. Every shape included a kernel whose specialization names `f16f16_f16f32`; this is direct Level-A evidence of FP16 operands with FP32 accumulation. Inspector reports Half output, so the realized phenotype is `F16A32O16`, not `F16A32O32`. All 12 micro-engines retained the requested tactic across three fresh rebuilds.

One QK micrograph (`fusion_net.layers.1.window_attention.fn`) rounds a small nonzero FP32 score entirely to zero at O16. This is an output precision boundary, not an accumulator pin failure.

## Full Model

An early A3/A5 diagnostic replay showed that primitive QK and AV keys can be pinned in full diagnostic graphs. Those graphs do not preserve the rest of F3 and are excluded from the formal profile result.

The exact F3-derived results are:

| Profile | Full-graph realization | fixed500 mAP | Delta vs F3 | Formal p50 |
|---|---|---:|---:|---:|
| Fresh F3 | QK F32A32O32, AV default FP16 | 0.648687490 | 0 | 4.303872 ms |
| QK native F16A32 request | six complete `_gemm_mha_v2` layers; primitive keys disappear | not evaluated as native pin | n/a | n/a |
| AV native F16A32 pin | 6/6 primitive `F16A32O16`, three rebuilds stable | 0.648573277 | -0.000114213 | 4.321280 ms |

The AV profile is accuracy-safe but 0.40% slower (`0.995972x` speedup) than fresh F3. It is rejected by the no-latency-gain gate. The QK profile is rejected because micro pinning is not preserved in the exact full graph. No new precision gene is enabled.

## Portability

The exact AV cache preserved all six targets on another RTX 4090 with the same 4 GiB workspace. Rebuilding with a 1 GiB workspace failed closed under `ERROR_ON_TIMING_CACHE_MISS`. The cache therefore requires exact TensorRT, CUDA, SM89, graph signature, shapes, BuilderConfig, tactic sources, and cache hash.

## Evidence

- Capability audit: `capability/`
- Per-shape availability: `native_shape_pinning_matrix.csv`
- Direct accumulator evidence: `tactic_accumulator_evidence.csv`
- Cache edits: `cache_edit_manifest.csv`
- Full-engine preservation: `full_engine_tactic_preservation.csv`
- Accuracy: `native_f16a32_accuracy.csv`
- Formal latency: `native_f16a32_latency.csv`
- Search contract: `native_f16a32_tactic_search_contract.json`
- Full conclusion: `root_conclusion.md`

Enumeration is marked `partial`: the public API exposes all candidates in the editable profiling table but does not guarantee that internally filtered implementations are enumerable. Complete fused MHA accumulator precision remains unknown.
