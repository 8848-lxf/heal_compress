# CoBEVT Minimal Structure × Quantization × Latency Plan

## Scope

Implement and execute the approved S0–S4 Attention structure cross, the S0-only
SmoothQuant projection experiment, and the 54-record F3 primitive TensorRT LUT.
The implementation reuses the existing Transformer semantic materializer, task
Taylor rankings, corrected requested/realized parser, and fixedK29696 strongly
typed export/build/evaluation path.

## Implementation order

1. Add failing tests for the five mandatory structures, P0/F3 role contracts,
   structure/precision interaction accounting, SmoothQuant profile dependencies,
   and the complete F3 LUT identity/gating rules.
2. Add a small model-family contract module and the Attention-FP32-with-rest-FP16
   explicit precision boundary.
3. Add a resumable orchestration entry point. Every phase writes immutable,
   hash-bound evidence under a fresh timestamped output directory.
4. Execute A in stages: fresh build, smoke10, fixed50, then the required six-way
   fixed500 selection.
5. Execute B using ModelOpt 0.29 SmoothQuant. Fail closed if the full-model export
   cannot preserve explicit INT8 projection realization.
6. Build all 54 fresh F3 deployment-unit engines. Only publish a formal LUT after
   five minutes of GPU isolation; otherwise publish screening evidence.
7. Generate the joint contract/report, run regressions and compile/diff checks,
   then commit and push only the independent branch.

## Invariants

- Structure masks are shared by P0 and F3 for a given S profile.
- Q/K indices are paired per head; V and output-projection input indices are
  paired independently.
- Requested precision never substitutes for EngineInspector realization.
- A requested/realized mismatch or complete fused MHA is excluded from the F3
  primitive LUT.
- No 1789-frame validation, Pyramid process mutation, stale engine reuse, or
  shared-GPU latency presented as formal evidence.
