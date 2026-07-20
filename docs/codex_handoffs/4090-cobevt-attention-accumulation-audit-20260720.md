# 4090 CoBEVT Attention Multiplication/Accumulation Audit

## Round 1: 2026-07-20

### Scope

This round audits the existing CoBEVT Attention precision boundary. It does not
modify Pyramid GA, Greedy search, pruning genes, the tracer, ChannelResolver,
or the QDQ deployment contract. Work was performed on
`feature/cobevt-attention-accumulation-audit` in the isolated CoBEVT worktree.

Entry was checked against the 4090 collaboration branch. The main branch was
at `06b82882d6318a0eeb372b386e758e0cf2466703`, remote sync was clean, and the
4090 branch dry-run push reported `Everything up-to-date`. `.bashrc` was not
modified. GPU 5 was used for the new Q/K capture and numerical screening; no
Pyramid process was stopped.

### Evidence

- CoBEVT has six actual Attention modules: three `window_attention` and three
  `grid_attention` modules. The requested `HGTCavAttention`, 8x8 and 16x16
  attention types are not present in this model and are recorded as unavailable.
- Captures use ten fixed validation frames, six Attention modules, and 60
  capture records. Capture manifest hash:
  `21cb7f0a41e97acbb9b8a...` (full hash is in the output manifest).
- The accepted F3 engine's QK layers are separate primitive Einsum/GEMM layers
  with tactic `sm80_xmma_gemm_f32f32_f32f32_f32...`; its Softmax and AV Einsum
  remain separate layers. This is FP32 QK operands and FP32 QK output, not a
  proven FP16-operand/FP32-accumulator path and not complete fused MHA.
- A fresh R1 full-model build was attempted with TensorRT 10.9, strongly typed,
  `--noTF32`, fixedK=29696 and the production Scatter plugin. The parser failed
  with `IEinsumLayer must have all inputs of same type. Input 1 has type Float
  and input 0 has type Half.` The failure is retained verbatim.
- R2 is deliberately not labeled as realized: TensorRT 10.9 exposes no
  separate strongly-typed MatrixMultiply accumulator control. A Cast pattern
  cannot prove FP16 operands with FP32 accumulation.
- The accepted F3 fixed500 result is 500/500, 0 skip, mAP 0.648738697
  (AP30 0.779363519, AP50 0.683141925, AP70 0.483710647). Its measured
  p50 5.187 ms is screening latency because formal isolated latency was blocked
  by shared GPU processes; it is not used as a formal speed ranking.

### Numerical interpretation

The captured-Q/K numerical matrix compares R0 through R5 using the same real
Q/K tensors. R2 is a local numerical reference implemented as FP16 products
summed in FP32; it is not a TensorRT engine result. In normal layer0/layer2
blocks R2 is closer to R3 than R1, indicating that low-precision accumulation
can contribute. The layer1 window block has Q/K magnitudes near 1e-6 and FP16
products underflow; R2 remains collapsed there while R3 remains close to the
FP32 reference. Therefore the current evidence supports a combination of
projection/input rounding and low-precision product/accumulation effects, with
input underflow capable of dominating. It does not support the claim that
accumulation alone is the unique cause.

### API conclusions

- TensorRT 10.9.0.34 has `IDynamicQuantizeLayer` and
  `INetworkDefinition.add_dynamic_quantize`, but the installed header/API audit
  exposes FP4/FP8 dynamic block quantization (block 16/32 semantics), not INT8
  dynamic Q/K quantization. There is no `IAttention` class and no separate
  MatrixMultiply accumulator property.
- ModelOpt 0.29.0 has a callable `INT8_SMOOTHQUANT_CFG` and
  `modelopt.torch.quantization.model_calib.smoothquant`. A toy Linear call with
  a calibration loop created `pre_quant_scale` and finite output. This is a
  ModelOpt SmoothQuant recipe, not SageAttention dynamic per-block quantization.
- Native SageAttention dynamic INT8/per-block semantics are not expressible by
  the current TensorRT 10.9 API audit. Any future test of that contract must be
  isolated in a newer TensorRT environment (at least the requested >=10.15.1
  comparison environment), without replacing the current 10.9 installation.

### Files

Implementation:

- `search/model_families/lidar_cobevt/attention_accumulation.py`
- `search/orchestration/lidar_cobevt_attention_accumulation_audit.py`
- `search/model_families/lidar_cobevt/attention_precision_boundaries.py`
- `search/orchestration/lidar_cobevt_attention_pruning.py`
- `tests/test_lidar_cobevt_attention_accumulation.py`

Output evidence:

`/data/lxf/heal_data/outputs/cobevt_attention_accumulation_audit_20260720_052632/`

Important files include `fp16_accumulation_matrix.csv`,
`int8_qk_accumulation_matrix.csv`, `requested_realized_precision.csv`,
`fusion_tactic_inventory.csv`, `full_model_accumulation_profiles.json`, the
TensorRT and ModelOpt API audits, and the preserved R1 parser failure log.

### Verification

The CoBEVT precision-boundary regression suite plus the new audit tests passed:
`54 passed`. Python compilation and `git diff --check` are run before commit.

### Status

`R2_EXACT_FP16_OPERANDS_FP32_ACCUM_SUPPORTED=false`

`R1_FULL_MODEL_FP16_DEFAULT_BUILD=false`

`R3_F3_PROVENANCE_REUSED=true`

`FORMAL_LATENCY_ISOLATED=false` (shared GPU processes blocked isolation)

`PYRAMID_GA_MODIFIED=false`

---

`2026-07-20T09:30:00+08:00 | round=1 | accumulation audit evidence and API conclusions recorded`
