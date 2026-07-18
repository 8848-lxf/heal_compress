# CoBEVT Attention FP16 Boundary Audit Design

## Goal

Locate the smallest CoBEVT Attention node set that must remain FP32, using
fresh strongly typed TensorRT engines, fixed smoke10/fixed50/fixed500 AP, and
intermediate tensor parity. This diagnostic does not alter search, pruning,
quantization-gene, or training behavior.

## Constraints

- Work only on `feature/cobevt-attention-dim-pruning-audit`.
- Use one `fixedK=29696` engine per profile; no buckets or chunking.
- Reuse the existing smoke10/fixed500 frame identities and GPU AP backend.
- Build and evaluate with TensorRT 10.9 in `modelopt` on physical GPU 2.
- Use strongly typed ONNX with explicit Cast; no precision constraints or
  weakly typed fallback.
- Keep all model weights unchanged and use the unpruned d32 baseline.
- Do not run INT8, GA, Greedy, pruning search, training, or distillation.

## Chosen Architecture

The generic canonical typed contract remains responsible for weighted
projection dtypes. A new CoBEVT-only boundary rewriter runs after canonical
typing and before the existing auxiliary dtype closure. It inserts explicit
input/output Cast nodes around LayerNorm, projection outputs, QK Einsum,
Softmax, AV Einsum, output projection, scale, and residual Add according to one
central profile manifest.

This ordering is required. Applying the rewriter after auxiliary closure could
leave a residual branch already downcast by propagated FP16, invalidating
single-boundary isolation. Applying it before canonical typing would allow the
generic pass to overwrite the intended functional boundaries.

## Profile Model

Each profile declares the compute dtype for these roles:

```text
layernorm
q_projection
k_projection
v_projection
qk_scale
qk_matmul
softmax
av_matmul
output_projection
residual_add
```

All unspecified roles are rejected. A profile also declares which operation
outputs return immediately to FP32. A0 through A7 are immutable built-ins; A8
is enabled only if scale remains separable in the actual exported graph. M1 to
M3 are built-in combinations. M4/M5 are materialized from the single-boundary
evidence and stored as resolved profile JSON rather than hidden code defaults.

## Graph Ownership

The rewriter discovers six blocks from the canonical origin map and validates
exactly one node per role per block:

- `/layers.{0,1,2}/{window,grid}_attention/norm/LayerNormalization`;
- canonical Q/K/V/Out MatMul nodes from the origin map;
- `fn/Mul_6` for the pre-QK scale;
- `fn/Einsum` for QK;
- `fn/attend/Softmax`;
- `fn/Einsum_1` for AV;
- `{window,grid}_attention/Add` for residual merge.

Missing, duplicate, or ambiguous nodes fail closed. The RPE Add and mask Where
are recorded in provenance and follow the logits/Softmax boundary implied by
the profile; they are never silently omitted.

## Provenance and Realization

Every rewrite emits a profile manifest and per-node records containing source
and destination tensors, ONNX element types, inserted Cast nodes, canonical
module ownership, and file hashes. Static inventory joins these records with
EngineInspector rows. Requested, ONNX, and realized dtype are reported
separately. A fused TensorRT layer is accepted only when the relevant ONNX node
appears in its metadata and exposed floating input/output formats satisfy the
requested boundary; otherwise realization is unresolved and the profile fails.

## Evaluation Flow

1. Build A0 and A1-A7 fresh; validate identity hashes and EngineInspector.
2. Evaluate smoke10 in order A1 through A7.
3. Classify delta mAP against A0 as safe (`>= -0.01`), ambiguous
   (`-0.05 < delta < -0.01`), or catastrophic (`<= -0.05`).
4. Run fixed50 only for ambiguous or parity-suspicious profiles and selected
   combinations.
5. Resolve M4/M5 from evidence; do not hardcode a root-cause conclusion.
6. Run fixed500 only for the best two or three combinations.
7. Measure formal latency only when GPU 2 has no competing process; otherwise
   retain AP evidence and mark latency blocked.

## Tensor Parity

Diagnostic ONNX copies append selected internal tensors as outputs. They are
built separately and never used for latency. The generic TensorRT runner reads
all outputs. A0 and each profile use identical frame inputs. Generic statistics
cover finite counts, moments, cosine, relative L2, maximum/mean absolute error.
QK, Softmax, and residual roles add the requested rank, distribution, entropy,
divergence, and update-retention metrics. The three worst frames are selected
from the full smoke10 parity error after all ten are evaluated.

## Failure Handling

- A profile with missing mapping, non-finite output, incomplete evaluation,
  mismatched engine identity, unresolved realized precision, or unexpected
  propagation is failed and preserved.
- Existing engines are not reused unless all requested source/config/plugin/
  manifest/build hashes match. The new output root starts empty, so normal
  operation is fresh.
- Diagnostic-output engines are explicitly labeled and excluded from latency.
- No threshold is relaxed and no model weight is changed to make a profile pass.

## Outputs

The timestamped output contains run/config/profile manifests, static and
realized precision inventories, A0-A7 results, combination results, diagnostic
tensor parity, failure frames, root-cause report, and the final recommended
precision contract. Repository handoff documentation records code changes,
commands, evidence hashes, failures, and remaining uncertainty.
