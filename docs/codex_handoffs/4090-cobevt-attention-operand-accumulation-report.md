# CoBEVT Attention Operand/Accumulator Audit

日期：2026-07-21
分支：`feature/cobevt-attention-operand-accumulation-search`
硬件：RTX 4090，SM89
TensorRT：10.9.0.34，CUDA 11.8，ModelOpt 0.29，strongly typed，noTF32

## 结论摘要

本轮没有把 FP16 storage、FP32 Cast 或 TensorRT opaque tactic 误报为独立的 F16A32 native TensorRT phenotype。TensorRT 10.9 的普通 strongly-typed `Einsum/MatMul` 没有可用的 accumulator API，EngineInspector 也不直接暴露 accumulator 类型。

原生矩阵共 72 条记录（6 个真实 Attention block × 12 个 QK/AV profile）：60 条成功构建并运行，12 条 native INT8 QK/AV 构建失败。`M2/N2` 的 Cast 请求实际是 materialized FP32 operands + FP32 GEMM，不能称为 F16A32。默认 FP16 tactic 在不同 block 上混合得到 F16A16、F16A32 或 unknown，不能作为可控搜索基因。

自定义 `QKMixedAccumPlugin` 使用 Conda CUDA 11.8 nvcc、cuBLASLt `CUBLAS_COMPUTE_32F`，对全部 6 个 block 的 QK/AV 通过了 Level-A plugin-oracle 证据；它明确标记为 `plugin_oracle`，不是 native TensorRT fused MHA。

## Full-model 结果

所有 profile 使用同一 checkpoint、fixedK=29696、同一 fixed500 manifest、GPU AP 后处理、8 个 DataLoader workers，并完成 500/500、0 skip。

| profile | QK phenotype | AV phenotype | fixed500 mAP | ΔmAP vs A0 | formal forward p50 (ms) | p95 (ms) | p99 (ms) | status |
|---|---|---|---:|---:|---:|---:|---:|---|
| A0_F3_REFERENCE | F32A32 native primitive | accepted F3 native FP16 path | 0.648863688 | 0 | 5.096993 | 5.716156 | 6.214573 | reference |
| A2_QK_F16A32_PLUGIN | plugin F16 operands/FP32 accumulation | accepted F3 native FP16 path | 0.648884910 | +0.000021 | 5.213299 | 5.872958 | 6.763183 | rejected: no latency gain |
| B1_AV_F16A32_PLUGIN | F32A32 native primitive | plugin F16 operands/FP32 accumulation | 0.648722092 | -0.000142 | 5.177538 | 5.976610 | 6.835268 | rejected: no latency gain |
| C1_QK_AV_F16A32_PLUGIN | plugin F16 operands/FP32 accumulation | plugin F16 operands/FP32 accumulation | 0.648477697 | -0.000386 | 5.059340 | 5.736802 | 6.128528 | experimental plugin oracle |

Formal latency was measured serially on GPU 7 after a five-minute, 31-sample process audit. It used the same fixed500 manifest, 20 warmup frames with ten execution rounds (200 warmup executions), and five execution rounds over 500 evaluation frames (2500 timed executions). Screening p50 values from the parallel smoke/fixed50 runs are not used as formal values.

## Native TensorRT matrix

`trt_realized_phenotype.csv` contains the complete matrix. The important aggregate is:

- `M0_QK_F32A32`: 6/6 F32A32, Level A.
- `M1_QK_F16_DEFAULT`: 4/6 F16A16, 1/6 F16A32, 1/6 unknown. This is tactic-dependent, not an independently requested accumulator contract.
- `M2_QK_F16A32_REQUEST`: 6/6 `F32A32_after_materialized_cast`.
- `M3_QK_F16A16_REQUEST`: 3/6 F16A16 and 3/6 F16A32; TensorRT 10.9 does not expose a separate accumulator request here.
- `M4_QK_I8A32I`: 0/6 build success.
- `M5_QK_BF16A32`: 6/6 BF16A32.
- `N0_AV_F32A32`: 6/6 F32A32.
- `N1_AV_F16_DEFAULT`: 6/6 F16A16.
- `N2_AV_F16A32_REQUEST`: 6/6 materialized FP32 GEMM.
- `N3_AV_F16A16_REQUEST`: 5/6 F16A16 and 1/6 F16A32.
- `N4_AV_I8A32I`: 0/6 build success.
- `N5_AV_BF16A32`: 6/6 BF16A32.

The native INT8 failures are build failures, not accuracy failures. No native INT8 QK/AV result is admitted to the search contract.

## Plugin oracle

The plugin matrix contains 12/12 build and runtime successes. Every QK and AV plugin input is Half and every output/accumulator contract is explicitly Float through cuBLASLt compute type `CUBLAS_COMPUTE_32F`. The plugin is serialized with an explicit 64 MiB workspace and context-attached cuBLASLt handle. The initial non-empty namespace and zero-workspace failures remain preserved under `plugins/plugin_oracle_matrix_namespace_failure.*` and the corresponding failure records.

The plugin is not a complete fused MHA implementation. It replaces the selected QK/AV Einsum with a custom TensorRT `PluginV2` layer and therefore remains an experimental deployment oracle, not a native TensorRT precision gene.

## Numerical boundary

The exact cuBLASLt oracle on captured tensors established the numerical separation:

- QK F16A32 mean relative L2: `0.0002075`; QK F16A16: `0.1671`.
- AV F16A32 mean relative L2: `0.0001082`; AV F16A16: `0.0003232`.
- QK BF16A32 mean relative L2: `0.0015606`; QK INT8A32I mean relative L2: `0.0051400`.
- AV BF16A32 mean relative L2: `0.0009334`; AV INT8A32I mean relative L2: `0.0189292`.

These are oracle fingerprints, not a claim that native TensorRT implemented every combination. The full-model results show that the accepted F3 path remains safe; the plugin F16A32 path also stays within the fixed500 safety threshold, but its custom-kernel status prevents formal search admission.

## Search contract

`attention_operand_accumulation_search_contract.json` records:

- accepted reference: `A0_F3_REFERENCE`;
- experimental: `C1_QK_AV_F16A32_PLUGIN`;
- rejected: `A2_QK_F16A32_PLUGIN`, `B1_AV_F16A32_PLUGIN` because their isolated p50 did not improve over A0;
- unknown accumulator phenotypes are not searchable;
- plugin-oracle results are not native TensorRT results.

The existing additive F3 unit latency LUT remains `not_additive`; this audit does not change that status or use unit latency sums for full-model ranking.

## Failure and scope records

- The first A2 rewrite failure (`Cast → Transpose → Einsum` producer path) is preserved. The rewrite was corrected to clone only pure layout nodes and recover the FP16 projection source.
- The second A2 audit failure (six dead `Mul_6` scale nodes) is preserved. The final inventory explicitly records those nodes as absorbed by the plugin instead of treating them as unresolved execution layers.
- The first formal replay path typo is preserved under `latency/formal_replay`; retry1 is retained, and the strict 200-warmup/2500-timed result is under `latency/formal_replay_retry2`.
- No 1789-frame validation, GA, Stage-A, Stage-B, or Pyramid process was started or modified.
- Native F16A32 accumulator control remains unresolved in TensorRT 10.9. The plugin provides a Level-A oracle only.

## Reproduction

Use the modelopt Python at `/home/lixingfeng/anaconda3/envs/modelopt/bin/python`, set `PYTHONPATH` to this worktree, and run the full-model entry point for `prepare`, `build`, and `evaluate` with the plugin path recorded in each run manifest. The native matrix and plugin matrix commands are recorded in their respective run manifests and build reports.
