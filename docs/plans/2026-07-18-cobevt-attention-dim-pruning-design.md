# CoBEVT Attention Dimension Pruning Design

Date: 2026-07-18
Branch: `feature/cobevt-attention-dim-pruning-audit`

## Scope

This experiment adds physical head-internal dimension pruning for the six
Attention modules in CoBEVT without changing the external HEAL sources or the
existing family-wide embedding pruning recipe.

The formal candidates are:

- baseline: `d_qk=32`, `d_v=32`, `E=256`, `H=8`;
- QK-only: `(d_qk, d_v)=(24,32)` and `(16,32)`;
- B1 internal bottleneck: `(24,24)` and `(16,16)` with `E=256`;
- B2 global embedding: at least `E=192`, `H=8`, `d_qk=d_v=24`.

## Physical Representation

The stock Attention uses one `[3E, E]` fused projection and `chunk(3)`, which
cannot represent unequal Q/K and V widths. The experiment replaces each loaded
Attention instance with a repository-owned module exposing independent
`q_proj`, `k_proj`, `v_proj`, and `out_proj` layers.

Conversion from the checkpoint is lossless at width 32:

- Q/K/V rows are copied from the three fused QKV regions;
- Q and K use identical per-head keep indices;
- V projection rows and output-projection input columns use identical per-head
  keep indices;
- QK and VO keep indices may differ;
- relative-position bias, mask semantics, dropout, residual and FFN behavior are
  unchanged;
- attention scaling is always `1/sqrt(d_qk)`.

No candidate uses zero masks as a substitute for physical pruning. Linear
metadata and weight tensors are physically resized.

## Taylor Ranking

The original FP32 checkpoint is evaluated on a fixed real-data calibration
manifest. Mean task-loss gradients are accumulated for each parameter. For an
Attention module `m`, head `h`, local dimension `r`:

`I_QK = sum(abs(WQ_row * mean_grad_WQ_row)) + sum(abs(WK_row * mean_grad_WK_row))`

`I_VO = sum(abs(WV_row * mean_grad_WV_row)) + sum(abs(WO_column * mean_grad_WO_column))`

Bias terms are included when present. Rankings are local to each module and
head, with stable local-dimension tie breaking. Quantization, Fisher, SQNR and
latency are not part of these structural rankings.

## B2 Closure

B2 uses an independent global embedding-axis keep set. It synchronously resizes
the shrinker output, fusion LayerNorms, Attention inputs/outputs, FFN boundary
dimensions, fusion MLP head and detection-head inputs. The Attention head count
remains eight; this is distinct from the existing recipe that prunes whole
heads while retaining `dim_head=32`.

## Validation

Every formal candidate must pass physical-shape, split, reshape, scale,
residual, finite-output, ONNX, TensorRT deserialize and GPU smoke audits. All AP
comparisons use one fixed 500-frame manifest and GPU AP/IoU postprocessing with
eight DataLoader workers. Full-engine FP16 latency is measured on GPU 2 only
when it is isolated; concurrent measurements are retained as screening data and
not presented as formal latency.

The INT8 study is limited to an Attention-compatible subgraph. Explicit Q/DQ
realization and TensorRT layer information determine whether a shape is called
INT8; engine build success alone is insufficient.

