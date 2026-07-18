# CoBEVT Attention Dimension Pruning Implementation Plan

1. Add failing tests for unequal Q/K/V projection widths, per-head QK/VO mask
   coupling, independent QK/VO positions, physical tensor shapes, scale and
   baseline conversion parity.
2. Implement the explicit-projection Attention replacement, deterministic mask
   codec, shape audits and B1 physical materialization.
3. Add real-task mean-gradient collection and first-order Taylor ranking for
   QK and VO head-internal units.
4. Add B2 global embedding closure for an eight-head `E=192`, `d_h=24`
   candidate and validate predicted versus physical parameter shapes.
5. Add an experiment runner that writes immutable candidate identities,
   manifests, masks, shape/parameter audits and reproduction commands.
6. Generate one deterministic 500-frame manifest, run zero-shot PyTorch smoke,
   export formal candidates to ONNX, and verify ONNX parity.
7. Build strongly typed FP16 full-model TensorRT engines in `modelopt`, run GPU
   smoke and fixed-500 evaluation on GPU 2.
8. Export and benchmark FP16/explicit-QDQ INT8 Attention subgraphs for the
   requested dimension sweep using captured CoBEVT activations.
9. Record failures without fallback, compact large artifacts, run regression,
   compile and diff checks, then commit and push only source/tests/lightweight
   evidence/report files.

