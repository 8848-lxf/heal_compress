# 4090 CoBEVT SmoothQuant Conda toolchain deployment audit

Date: 2026-07-20/21 (Asia/Shanghai)

Branch: `feature/cobevt-smoothquant-conda-toolchain`

Base commit: `1a818373bd690206e0ee981fce461a5cc6a03328`

Output: `/data/lxf/heal_data/outputs/cobevt_smoothquant_conda_toolchain_20260720_222809`

## Conclusion

The previous run did invoke `/usr/bin/nvcc`, failed to compile the SM89 ModelOpt extension, fell back away from the CUDA extension, and later exited full-model SQ1/SQ2 export with `SIGSEGV`.  The bounded rerun uses the modelopt Conda Python, Conda CUDA 11.8 `nvcc`, Conda `g++`, an isolated SM89 extension cache, and the vendored pinned ModelOpt 0.29 source.  Both the minimal extension and the real ModelOpt tensor-quant extension execute on CUDA without CPU fallback.  E0--E5 now export successfully and SQ1/SQ2/SQ3 all build strongly typed TensorRT engines with exact INT8 projection realization and FP32 QK.

This is a strong provenance reconciliation, but not a compiler-only causal A/B: the current branch also fixes stale ONNX dtype propagation and restores projection Q/DQ adjacency after the F3 boundary rewrite.  The defensible conclusion is that the old toolchain was invalid and the historical SIGSEGV is not reproducible after both the toolchain and graph-contract fixes; it is not defensible to attribute the signal to `nvcc` alone.

## Toolchain

| item | value |
|---|---|
| Conda prefix | `/home/lixingfeng/anaconda3/envs/modelopt` |
| Python | `/home/lixingfeng/anaconda3/envs/modelopt/bin/python` |
| ModelOpt source | `/home/lixingfeng/UniAD_examine/HEAL/prune_model/Model-Optimizer-0.29.0` |
| ModelOpt | 0.29.0 |
| nvcc | `/home/lixingfeng/anaconda3/envs/modelopt/bin/nvcc` |
| nvcc version | CUDA 11.8, V11.8.89 |
| nvcc SHA256 | `be61debaa9878c5b221f5943c5ff1b468adc77c3093ef6b243a765569c23f9fc` |
| host C++ | `/home/lixingfeng/anaconda3/envs/modelopt/bin/x86_64-conda-linux-gnu-g++` |
| CUDA_HOME | `/home/lixingfeng/anaconda3/envs/modelopt` |
| CUDACXX | `/home/lixingfeng/anaconda3/envs/modelopt/bin/nvcc` |
| arch | SM89 (`TORCH_CUDA_ARCH_LIST=8.9`) |
| TensorRT | 10.9.0.34, strongly typed, no TF32 |
| fixedK | 29696 |

The minimal extension output is bit-exact to its PyTorch reference (`max_abs=0`, `relative_l2=0`), contains an `sm_89` cubin, and links Conda `libcudart.so.11.0`.  The real ModelOpt extension was compiled by the same Conda compiler and loaded from the run-specific cache; `runtime_backend=cuda`, `cpu_fallback=false`.

## Alpha sweep

All profiles use static per-tensor symmetric INT8 activations and per-output-channel symmetric INT8 weights.  The selection score is the sum of independently min-max-normalized projection relative L2, QK relative L2, and Softmax JS.  All three profiles select `alpha=0.7`.

| profile | INT8 projections | selected alpha | projection rel-L2 | QK rel-L2 | Softmax JS |
|---|---|---:|---:|---:|---:|
| SQ1 | Q,K | 0.7 | 0.004924913 | 0.006761849 | 4.1484e-6 |
| SQ2 | Q,K,V | 0.7 | 0.006063769 | 0.006761849 | 4.1484e-6 |
| SQ3 | Q,K,V,Out | 0.7 | 0.005975759 | 0.006798147 | 4.0767e-6 |

Alpha 0.75 slightly lowers QK error but raises projection error; 0.8 lowers Softmax JS but degrades both projection and QK reconstruction.  Under the declared joint metric 0.7 remains the stable choice.

## Export and TensorRT realization

| stage/profile | export | build | requested INT8 projections | realized INT8 | QK | fused MHA |
|---|---|---|---:|---:|---|---|
| E0 toy | pass | n/a | Q,K recipe | n/a | n/a | n/a |
| E1 real Q projection | pass | n/a | Q | n/a | n/a | n/a |
| E2 one real Attention QK | pass | n/a | Q,K | n/a | FP32 comparison graph | no |
| E3 / SQ1 | pass | pass | 12 | 12 | 6/6 FP32 primitive, FP32 accumulator | no |
| E4 / SQ2 | pass | pass | 18 | 18 | 6/6 FP32 primitive, FP32 accumulator | no |
| E5 / SQ3 | pass | pass | 24 | 24 | 6/6 FP32 primitive, FP32 accumulator | no |

Engine SHA256:

- SQ1: `79245ae3e7f763dd93778a79e0fef7389321e01ea3453ea1ab6062dc0fff17ae`
- SQ2: `47f20b4f3acb59d0a823c9b388ed0db24d04e0d4c2a89ef7fa6faabe44334111`
- SQ3: `481cd08d39620846debdb027fd9166d9eceb7e86e8647c6ecee374a0def1c8b1`

The realized projection tactics contain INT8 tensor-core GEMM metadata.  Q/K projection outputs recover to Float and all six QK tactics are Float/Float to Float.  This is INT8 projection plus FP32 QK, not native INT8 Attention.  TensorRT exposes no complete fused MHA in these engines.  No requested projection falls back to FP16/FP32.  The pre-quant scale is present in ONNX; no standalone engine layer with that node identity remains, so TensorRT absorbed it into fused execution.  This proves absence of a standalone runtime Mul, not necessarily compile-time weight folding.

## Accuracy

All evaluations use eight DataLoader workers, the CUDA AP/IoU backend, fixedK29696, the same manifests, and zero skipped frames.

| profile | scope | AP30 | AP50 | AP70 | mAP | delta mAP vs fresh S0 F3 |
|---|---|---:|---:|---:|---:|---:|
| S0 F3 | smoke10 | 0.747942 | 0.556926 | 0.257266 | 0.520711 | 0 |
| SQ1 | smoke10 | 0.746262 | 0.559157 | 0.250034 | 0.518484 | -0.002227 |
| SQ2 | smoke10 | 0.746019 | 0.558247 | 0.245110 | 0.516459 | -0.004252 |
| SQ3 | smoke10 | 0.746208 | 0.558482 | 0.244936 | 0.516542 | -0.004169 |
| S0 F3 | fixed50 | 0.742440 | 0.619598 | 0.410264 | 0.590767 | 0 |
| SQ1 | fixed50 | 0.743363 | 0.621252 | 0.412048 | 0.592221 | +0.001454 |
| SQ2 | fixed50 | 0.743663 | 0.620611 | 0.406196 | 0.590157 | -0.000610 |
| SQ3 | fixed50 | 0.741865 | 0.621690 | 0.406961 | 0.590172 | -0.000595 |
| accepted historical S0 F3 | fixed500 | 0.779364 | 0.683142 | 0.483711 | 0.648739 | reference |
| SQ1 | fixed500 | 0.779252 | 0.682966 | 0.482608 | 0.648275 | -0.000463 |

SQ1 was the only SmoothQuant profile promoted to fixed500 because it was the fixed50 winner.  Its fixed500 change is within the earlier ±0.003 accuracy-safe band.  The small fixed50 gains are treated as sampling noise, not an accuracy improvement claim.

## S2 structure follow-up

The retained indices and F3 contract were reused unchanged.  S2 (`d_qk=d_v=16`) completed 500/500 with zero skip:

- AP30 0.774428
- AP50 0.672662
- AP70 0.470454
- mAP 0.639181
- accepted formal p50 3.935232 ms

S2 is mAP-latency non-dominated in the bounded structure set, but remains experimental because its mAP is 0.009658 below S0 F3.  The previously rejected additive unit LUT remains `not_additive`; only full-engine action anchors may be used.

## Formal latency

GPU 0 (`GPU-8c4f469e-d367-4cd9-52f0-b8645cd67a23`) was observed for five minutes before creating the CUDA context.  No external process appeared.  The four engines were then replayed serially with device-resident inputs, CUDA events, 200 warmups, 2000 timed iterations and five repeats.  Process audits between candidates found no foreign process.

| profile | p50 ms | p90 ms | p95 ms | p99 ms | speedup vs S0 F3 | p50 reduction |
|---|---:|---:|---:|---:|---:|---:|
| S0 F3 | 4.240144 | 4.246528 | 4.248576 | 4.252682 | 1.0000x | 0% |
| SQ1 | 3.649536 | 3.659776 | 3.661861 | 3.666944 | 1.1618x | 13.93% |
| SQ2 | 3.644416 | 3.652576 | 3.654656 | 3.658689 | 1.1635x | 14.05% |
| SQ3 | 3.609600 | 3.617792 | 3.619840 | 3.623936 | 1.1747x | 14.87% |

SQ3 is the fastest engine, but it has only fixed50 accuracy.  SQ1 is the fastest profile that also has fixed500 evidence, and is the accuracy/latency recommendation for the next bounded search.

## Search contract

- Structure: retain S3 24/32 as allowed; keep S1 24/24 and S2 16/16 experimental; S2 is not promoted solely because it is Pareto non-dominated.
- Precision: ATTN_FP32 and F3 remain allowed.
- SmoothQuant: SQ1 is allowed for a bounded next-stage search because exact realization, fixed500 accuracy, and isolated latency all pass. SQ2/SQ3 remain experimental because they do not improve fixed50 accuracy over SQ1 and lack fixed500 evidence.
- R1 full FP16 QK remains rejected.
- Native INT8 QK remains forbidden.
- Unit latency LUT remains `not_additive`; full-engine anchors are authoritative.
- Next experiment: start with a bounded Greedy smoke using only S0/S3 plus SQ1 and full-engine action anchors.  Do not expand GA precision genes until SQ1 has passed the later 1789-frame production validation.

## Validation

- CoBEVT regression, including all new toolchain/precision/reporting tests: 277 passed.
- Modified Python files compile successfully; `git diff --check` passes.
- No Pyramid GA process was modified, stopped, or restarted.

## Key evidence

- `environment/` and `modelopt_extension/`: compiler/runtime provenance.
- `smoothquant/alpha_sweep.csv`: 15-point alpha sweep.
- `export_debug/staged_export_matrix.csv`: E0--E5 status.
- `tensorrt/build_matrix.csv`: requested/realized gate.
- `evaluation/smoothquant_results.csv`: AP results.
- `structure_followup/s2_fixed500.json`: S2 result.
- `latency/formal_latency.json`: isolated latency replay, when complete.
