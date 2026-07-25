# 4090 V2X-ViT Greedy 0.30 deployment-closed run

Completed on the RTX 4090 collaboration branch `4090-transformer-unified-search`, starting from `3ffcd93021a90743a71f677152e0757e7b2bc1b8`. The formal H800 branches were read-only throughout this run.

Run root:

`/data/lxf/heal_data/outputs/h800_v2xvit_greedy030_joint_taylor_deployment_closed_v2_20260724_224500`

## 2026-07-25 02:30 CST

Implemented a deployment-closed V2X-ViT precision space and strongly typed export/build path:

- `search/quantization_space/v2xvit_deployment_closed.py`: legal weighted and functional precision loci.
- `quantization/precision/typed_graph.py`: explicit typed-graph closure and fail-closed Q/DQ boundaries.
- `search/stage2/v2xvit_deployment_closed.py`: fresh physical export, candidate-bound train200 calibration and exact realization checks.
- `search/model_family/deployment.py`: deterministic EntropyCalibration2 collection, repeated-module observation audit, and finite positive per-channel weight scales.
- `search/integration/runtime_environment.py`: explicit ModelOpt 0.29 vendored source and Conda environment resolution.

Original-structure deployment controls all built successfully and were requested/realized exact. The theoretical deployment-closed BOPS retention values are P32 `0.973352610128241`, P16 `0.42960487035801975`, and P8 `0.29366793541546443`. P8 fixed50 mAP was `0.4085919105939901`, so the buildable precision floor is not an accuracy-safe operating point.

The historical 0.05 JMIX failure was reproduced as a strongly typed functional-boundary problem. On the same extreme physical structure, the repaired maximal mixed profile built fresh with 68 requested and 68 realized INT8 weighted loci, train200 `200/200`, and zero skipped frames. This isolates the historical CUDA 716/Myelin error to stale functional activation/Q/DQ typing rather than an unavoidable physical-shape failure.

## 2026-07-25 04:20 CST

Implemented conservative fixed-statistics proxies:

- `search/proxy/conservative_action_taylor.py`: elementwise absolute first/second-order weight and activation transition costs.
- Activation precision action:
  `sum(abs(g_A * delta_A) + 0.5 * abs(h_A * delta_A^2))`.
- Weight precision action:
  `sum(abs(g_W * delta_W) + 0.5 * abs(h_W * delta_W^2))`.
- Structure action:
  `sum(abs(g_u * u) + 0.5 * abs(h_u * u^2))` at semantic functional gates.
- `search/greedy/conservative_joint.py`: adjacent legal actions, nonnegative risk/BOPS utility, zero repair, Taylor-first winner selection and physical-only Stage-2 deduplication.

Eight fixed training samples were streamed for weight Fisher, activation transitions and structural gates. Activation tensors were released after each sample; only scalar transition costs were retained. The Greedy loop performed zero forward, backward, physical export, ONNX export and TensorRT build calls.

## 2026-07-25 05:35 CST

The first single-sample proxy run selected an aggressive structure at R_BOPS `0.2965086`; its physical S32 fixed50 mAP was `0.3953647761615597`, confirming structural collapse. It is retained as failed evidence and was not reused.

The eight-sample run selected five physical-unique Stage-2 candidates. Their S32 fixed50 mAP values were:

| Candidate | R_BOPS | S32 fixed50 mAP |
|---|---:|---:|
| `55c2560d` | 0.304278995 | 0.574405027 |
| `f1921e00` | 0.304476687 | 0.575637158 |
| `90a7e5ed` | 0.303611348 | 0.573727287 |
| `d40cc33` | 0.304226140 | 0.575153434 |
| `5bd288fb` | 0.304163452 | 0.574807202 |

The B0 fixed50 mAP was `0.5764098935206716`; all five passed the `B0 - 0.01` structural gate. Candidate `f1921e008a0949a9341dcf36a4f07dcc9920226ffe3838367020ac1e6140ed62` was selected by highest physical S32 mAP.

Winner summary:

- R_BOPS: `0.30447668683887086`.
- Parameter retention: `0.7380607746991291`.
- Precision loci: 99 FP16, 14 FP32, 0 INT8.
- Shrinker retained width: 120.
- Attention and FFN widths remain at their native widths.
- No structure or precision repair.
- JMIX engine SHA256: `871ef3aa82502939984043549b346560b6616140f251dbad8a9097159e09c06a`.
- JMIX requested/realized exact, zero fallback.

## 2026-07-25 06:15 CST

Fresh fixed500 evaluation used workers=8 and CUDA IoU/NMS, with 500 evaluated and zero skipped frames for every control:

| Control | AP30 | AP50 | AP70 | mAP |
|---|---:|---:|---:|---:|
| B0 | 0.770197629 | 0.692212014 | 0.512265809 | 0.658225151 |
| S32 | 0.770940251 | 0.692996610 | 0.511682390 | 0.658539750 |
| JMIX-FRESH | 0.770861674 | 0.693339145 | 0.511729071 | 0.658643297 |

The sub-millipoint positive differences are treated as evaluation noise, not accuracy gains. There is no structural collapse at R_BOPS 0.30.

Formal latency used GPU UUID `GPU-3f1b6c50-00bc-f851-01f0-6c9ff3471dad`, 200 warmup executions, 500 timed executions, five matched rounds, and baseline replay before and after each candidate:

| Control | p50 ms | Speedup vs matched B0 |
|---|---:|---:|
| B0 | 27.689072 | 1.0000x |
| S32 | 26.962432 | 1.0270x |
| JMIX-FRESH | 22.923088 | 1.2079x |

Compression relative to B0:

- BOPS: `3.284323704x`.
- Parameters: `1.354901973x`.
- Theoretical mixed weight storage: `2.323422879x`.
- JMIX engine file size is reported separately and is not used as theoretical weight storage.

Final machine-readable acceptance is at `reports/final_acceptance.json`. Formal GA and full1789 remain disabled; this run validates the repaired Greedy/deployment closure at 0.30 only.

Reproduction entry points:

```bash
/home/lixingfeng/anaconda3/envs/modelopt/bin/python scripts/run_v2xvit_greedy030_conservative.py ...
/home/lixingfeng/anaconda3/envs/modelopt/bin/python scripts/run_v2xvit_deployment_closed_profile.py ...
/home/lixingfeng/anaconda3/envs/modelopt/bin/python scripts/evaluate_v2xvit_deployment_closed_engine.py ...
/home/lixingfeng/anaconda3/envs/modelopt/bin/python scripts/run_v2xvit_weight_only_latency.py ...
/home/lixingfeng/anaconda3/envs/modelopt/bin/python scripts/finalize_v2xvit_greedy030_deployment_closed.py
```
