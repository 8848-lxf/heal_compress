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

## Round 2: 2026-07-20

### Scope

This follow-up only reconciles the historical low-precision Attention build
evidence with the failed R1 graph and corrects requested/realized precision
provenance. It does not add a search profile, run fixed50/fixed500, or touch
Pyramid GA.

### R1 root cause and fix

Historical `attention_fp16`, `M2_projection_qk_av_fp16_softmax_fp32`, and
`M4_projection_boundary_qk_av_fp16` engines were all TensorRT 10.9 strongly
typed builds with symmetric FP16/FP16 QK operands in all six Attention blocks.
They realized six complete `_gemm_mha_v2` layers and produced smoke10 mAP in
the 0.11605-0.11686 range.

The failed R1 graph was not equivalent: its first QK Einsum received Half on
input 0 and Float on input 1. A recovered FP32 K projection passed through
Reshape/Transpose nodes while stale pre-rewrite value-info still labeled the
branch FP16, so only the Q operand received the required Cast. TensorRT failed
at parser validation before tactic selection.

The boundary rewriter now propagates updated types through pass-through nodes
and validates both QK data operand dtypes after ONNX inference. Mixed or
unresolved QK operands fail during export. Unit tests cover stale type-info,
window/grid symmetry, one-sided rewrite failure, F3 preservation, and AV
isolation.

Fresh fixed R1 evidence:

- profile: `R1_fp16_operands_default_accum_fixed`
- ONNX SHA256: `1c6388520ce0c3d519b663a920bd5544822d42ed6dbed3aeb11e5cb6d84f37f0`
- engine SHA256: `d19bbfbea966a550699fe9644dc89cb5687b93808ba23acc310cef2a77094a92`
- all six QK nodes: FP16 / FP16, validated before build
- build/runtime: success, 10/10, 0 skip
- AP30/AP50/AP70/mAP: 0.167243587 / 0.141195565 / 0.039307246 / 0.115915466
- realization: six complete `_gemm_mha_v2` layers
- accumulator precision: `unknown` because the fused tactic does not expose
  reliable accumulator metadata
- search policy: `not_allowed_for_formal_search`

### Precision parser correction

Requested precision is now taken from the F3 role contract, never from a
coarse profile-name label. Realized precision uses EngineInspector dtype and
tactic first; explicit typed-ONNX boundaries are accepted only as secondary
evidence when a functional role is absorbed into a local fusion. Conflicting
evidence returns `unknown`.

The corrected F3 inventory has 72 rows and zero conflicts. Across all six
blocks, Q/K/V projections and output projections are FP16 `h16816gemm`, QK is
FP32 `f32f32...f32` with FP32 accumulator evidence, and AV is FP16
`h16816gemm`. Q/K recovery Casts explicitly produce FP32. F3 remains separate
primitive QK/AV GEMMs, not complete fused MHA.

### Files

Code and tests:

- `search/model_families/lidar_cobevt/attention_precision_boundaries.py`
- `search/model_families/lidar_cobevt/attention_accumulation.py`
- `search/orchestration/lidar_cobevt_attention_pruning.py`
- `tests/test_lidar_cobevt_attention_precision_boundaries.py`
- `tests/test_lidar_cobevt_attention_accumulation.py`

Evidence:

`/data/lxf/heal_data/outputs/cobevt_attention_accumulation_followup_20260720_112146/`

Key files are `historical_fp16_profile_inventory.csv/json`,
`strict_fp16_vs_r1_qk_dtype_matrix.csv/json`,
`requested_realized_precision_corrected.csv/json`,
`r1_fixed_build_report.json`, `r1_fixed_fusion_tactic_inventory.csv`,
`strict_fp16_vs_r1_provenance_diff.md`, and
`r1_reconciliation_conclusion.md`.

---

`2026-07-20T11:48:57-07:00 | round=2 | R1 provenance reconciled and F3 precision parser corrected`
