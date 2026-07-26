# H800 V2X-ViT GA two-tier evaluation handoff

Branch: `feature/h800-v2xvit-stage2-300-genwinner500`

Base: `836a49a6e23156c55ef6e9fe2028670b810b38de`

## 2026-07-26 12:05 PDT

- Added the two-tier real-evaluation contract without changing the 10-generation GA scheduler:
  - every generation's Stage-2 Top-5 uses 300 validation frames after a 100-frame warmup;
  - the unique winner selected in each generation is re-evaluated on 500 validation frames after a 200-frame warmup;
  - the final budget winner is selected only from the fixed500 generation winners and the fixed500 Greedy anchor.
- Generation-winner validation reuses the exact screened TensorRT engine after phenotype hash, engine SHA256, and requested/realized precision checks. It does not repeat physical pruning, calibration, ONNX export, Q/DQ insertion, or engine build.
- Screening and winner-validation results use separate cache namespaces and carry explicit frame/warmup/protocol metadata.
- Audited the formal Taylor call chain. Structure gate, weight quantization, and activation quantization all use first plus empirical-Fisher second order:
  `abs(g * delta) + 0.5 * abs(E[g^2] * delta^2)`.
- Corrected the functional-gate diagnostic decomposition so first- and second-order totals are reported separately; the fitness formula itself was already correct.
- Tests: 46 targeted tests passed; compileall, py_compile, and `git diff --check` passed.

--- 2026-07-26T12:05:00-07:00 ---
