# H800 Pyramid R_BOPS=0.05 J_AQ=0 controlled GA ablation

Branch: `exp/h800-pyramid-r005-ga-jaq0-gen10`  
Worktree: `/home/lixingfeng/UniAD_examine/heal_compress_h800_pyramid_r005_ga_jaq0`

---

Timestamp: 2026-07-28 05:10 CST

## Development progress

- Added a strict `activation_taylor_fitness_weight` switch to the shared
  Stage-1 evaluator and CNN Greedy action scoring. Only values `0` and `1` are
  accepted.
- The `J_AQ=0` mode retains raw activation Taylor statistics as diagnostics,
  but uses `J_total = J_struct_gate + J_WQ + 0 * J_AQ` for Greedy and GA.
- Physical W8A8 activation quantization, calibration, ONNX/QDQ, TensorRT
  realization checks, fixed300 Stage-2 screening and fixed500 generation-winner
  validation remain unchanged.
- Restricted direct ten-generation ablation execution to Pyramid, seed 0 and
  the single R_BOPS=0.05 budget. This is the strongest matched diagnostic for
  the observed low-budget pruning bias while bounding disk and engine cost.
- The experiment must start only after the main Pyramid J_AQ=1 continuation
  finishes. It writes to a new output root and cannot reuse the main run's
  candidate fitness or final winner.
- Targeted regression: 33 passed; compileall, py_compile and git diff check
  passed.

