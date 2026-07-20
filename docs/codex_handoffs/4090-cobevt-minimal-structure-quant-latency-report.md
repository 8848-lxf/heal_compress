# 4090 CoBEVT minimal structure × quantization × latency study

Branch: `feature/cobevt-minimal-structure-quant-latency`

Base: `b4cae960111a13eee4c578601948fb4f6c1261bd`

Output: `/data/lxf/heal_data/outputs/cobevt_minimal_structure_quant_latency_20260720_132209`

## Scope and implementation

This bounded experiment adds five deterministic Attention structures (S0--S4),
an Attention-FP32 control contract, the accepted F3 contract, selective
ModelOpt 0.29 SmoothQuant screening, and a real TensorRT F3 deployment-unit
latency LUT. It does not run Pyramid GA, 1789-frame validation, or a new search.

Key implementation files:

- `search/model_families/lidar_cobevt/minimal_structure_quant_latency.py`:
  schemas, contracts, interaction and LUT validation.
- `search/model_families/lidar_cobevt/f3_attention_unit.py`: real-weight F3
  deployment unit and strict realized-phenotype parser.
- `search/orchestration/lidar_cobevt_minimal_structure_quant_latency.py`:
  S0--S4 materialization, build and evaluation orchestration.
- `search/orchestration/lidar_cobevt_smoothquant_projection.py`: real-activation
  ModelOpt alpha sweep and fail-closed full-model attempts.
- `search/orchestration/lidar_cobevt_f3_latency_lut.py`: 54-engine build,
  isolation audit, CUDA-event timing and full-engine validation.
- `search/reporting/cobevt_minimal_structure_quant_latency.py`: machine-readable
  candidate contract and final report writer.

## Structure × precision evidence

All five structures passed physical forward checks. All ten P0/F3 engines were
freshly exported and built strongly typed with zero unresolved or realization
mismatches. All fixed50 evaluations completed 50/50 with zero skip. The six
selected fixed500 results were:

| structure | d_qk/d_v | profile | AP30 | AP50 | AP70 | mAP | screening p50 ms |
|---|---:|---|---:|---:|---:|---:|---:|
| S0 | 32/32 | Attention FP32 | 0.779366 | 0.683304 | 0.483874 | 0.648848 | 5.907 |
| S0 | 32/32 | F3 | 0.779880 | 0.683404 | 0.483234 | 0.648839 | 5.248 |
| S1 | 24/24 | Attention FP32 | 0.779062 | 0.680796 | 0.477184 | 0.645681 | 5.709 |
| S1 | 24/24 | F3 | 0.778408 | 0.679789 | 0.478816 | 0.645671 | 5.176 |
| S3 | 24/32 | Attention FP32 | 0.779550 | 0.682149 | 0.478587 | 0.646762 | 6.273 |
| S3 | 24/32 | F3 | 0.779522 | 0.682229 | 0.478061 | 0.646604 | 5.161 |

The F3 interaction term is `-0.000001103` for S1 and `-0.000149597` for
S3. At these widths, structure loss dominates and the additional F3 loss is
nearly independent.

The accepted historical mixed baseline is read from the accepted artifact,
not hard-coded as a replacement result: mAP `0.6489856644822475`.

## SmoothQuant evidence

The alpha sweep uses inputs captured from the same fixed smoke10 manifest for
all six real Attention blocks. ModelOpt 0.29 created `pre_quant_scale` for every
selected module. Alpha 0.7 was selected for all three profiles:

| profile | selected roles | alpha | projection rel-L2 | QK rel-L2 | Softmax JS |
|---|---|---:|---:|---:|---:|
| SQ1 | Q/K | 0.7 | 0.004949 | 0.006851 | 0.000004090 |
| SQ2 | Q/K/V | 0.7 | 0.006091 | 0.006851 | 0.000004090 |
| SQ3 | Q/K/V/out | 0.7 | 0.005978 | 0.006851 | 0.000004090 |

SQ1 and SQ2 full-model runs both completed ModelOpt calibration and smoothing,
then exited with SIGSEGV during fresh ONNX tracing before an ONNX or TensorRT
engine was created. The logs also show that `/usr/bin/nvcc` rejects SM89, so
the ModelOpt CUDA extension falls back to its CPU implementation. SQ3 was not
run because its explicit SQ2-success prerequisite failed. Consequently no
requested INT8 projection is reported as realized and there is no SmoothQuant
AP result.

## F3 primitive LUT

All 54 mandatory `(6 blocks) × (3 d_qk) × (3 d_v)` engines built and passed the
strict F3 realized contract. TensorRT fuses Q/K/V projections locally, while QK,
Softmax, AV and output projection remain primitive execution layers. Complete
fused MHA is false for 54/54.

Formal timing used GPU UUID
`GPU-a4b5f0d9-c77c-2c99-b4c3-1b3811b835e3`, after a five-minute no-process
isolation audit, with 200 warmups, 2000 measured executions and five repeats.
Across blocks, unit p50 is approximately 0.161--0.209 ms. Typical values are:

| d_qk/d_v | per-block p50 range ms |
|---:|---:|
| 16/16 | 0.162--0.164 |
| 24/24 | 0.190--0.192 |
| 32/32 | 0.207--0.209 |

The full-engine delta validation and final LUT classification are recorded in
`latency_lut/f3_lut_full_engine_validation.csv` and
`latency_lut/f3_lut_validation_report.md`.

| structure | d_qk/d_v | formal full-engine p50 ms | p95 ms | delta vs S0 ms |
|---|---:|---:|---:|---:|
| S0 | 32/32 | 4.237600 | 4.245440 | 0.000000 |
| S1 | 24/24 | 4.154368 | 4.160770 | -0.083232 |
| S2 | 16/16 | 3.935232 | 3.941376 | -0.302368 |
| S3 | 24/32 | 4.165424 | 4.171776 | -0.072176 |
| S4 | 32/24 | 4.266944 | 4.272128 | +0.029344 |

The mean relative error of LUT-predicted full-engine deltas is `1.020176`.
S1 and S2 individual errors are 18.45% and 10.52%, but S3 is 51.12% and S4
has the wrong speed ordering. Therefore LUT v1 is `not_additive`, not
search-ready; it may be retained only as local primitive evidence.

## Candidate contract

- Allowed: S0 and S3; Attention-FP32 and F3.
- Experimental: S1; S2/S4 remain fixed50-only and require fixed500 before
  promotion.
- SmoothQuant: no allowed full-model profile; SQ1/SQ2 remain unresolved at
  export, and SQ3 is prerequisite-blocked.
- Forbidden: full-Attention-FP16 R1 and any requested/realized mismatch.
- No 1789-frame conclusion is claimed.

---

Round completed at `2026-07-20 Asia/Shanghai`; final commit and verification
evidence:

- implementation commit: `60ba3cff6fa12978e38a07fcd56f9508b47c1d2d`;
- CoBEVT Attention regression set: `186 passed`, 24 warnings;
- modified Python `py_compile`: passed;
- `git diff --check`: passed;
- output size: approximately 4.8 GiB, containing ONNX/engine evidence outside Git;
- Pyramid GA was neither modified nor interrupted.
