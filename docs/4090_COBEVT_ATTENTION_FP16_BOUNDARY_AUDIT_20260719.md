# 4090 CoBEVT Attention FP16 Boundary Audit

## 本轮状态

- 分支：`feature/cobevt-attention-dim-pruning-audit`
- 物理 GPU：2，RTX 4090，fixedK=29696
- TensorRT：10.9，strongly typed，`--noTF32`
- 输出目录：`/data/lxf/heal_data/outputs/cobevt_attention_fp16_boundary_audit_20260719_030621`
- 本轮没有修改 GA、Greedy、tracer、ChannelResolver 或结构化剪枝器。
- 本轮没有执行 INT8、结构剪枝或训练。

## 直接结论

CoBEVT 的 Attention FP16 崩塌不是 Softmax 单点问题，也不是 Q/K/V
projection 或 QK MatMul 各自单独使用 FP16 的问题。可复现的最小破坏组合是：

```text
FP16 projection values -> FP16 QK scale/MatMul
```

证据如下：

- A1（Q/K/V projection FP16，输出立即 Cast FP32）安全；
- A3（仅 QK MatMul FP16）安全；
- A4（仅 Softmax FP16）安全；
- M2（projections + QK + AV FP16，Softmax/Add FP32）灾难性崩塌；
- M4（projection 输出先 Cast FP32，再 Cast FP16 进入 QK）仍灾难性崩塌；
- M3/M5（projections/AV/Out 等 FP16，但 QK 保持 FP32）安全。

因此 Softmax 不是唯一根因。它可以放大上游 logits/rank 误差，但在 QK
保持 FP32 时，Softmax 自身使用 FP16 已被 A4、M5 和 F3 验证为安全。

当前最小已验证 FP32 precision island：

```text
LayerNorm FP32
Q projection FP16 -> explicit Cast FP32
K projection FP16 -> explicit Cast FP32
QK scale FP32
QK MatMul FP32
V projection FP16
Softmax FP16
AV MatMul FP16
output projection FP16
residual Add FP16
```

LayerNorm 保留 FP32 是保守的部署契约：历史崩塌图中的 LayerNorm 本身仍为
FP32，未被实验证明为根因；A2 的严格 FP16 LayerNorm 在 TensorRT 10.9 中被
融合为 Float/Float，无法满足 requested/realized identity，故 fail closed，
不用于安全性结论。

## A0-A7 Smoke10

| Profile | evaluated/skipped | AP30 | AP50 | AP70 | mAP | screening p50 ms | 结论 |
|---|---:|---:|---:|---:|---:|---:|---|
| A0 strict FP32 | 10/0 | 0.746086 | 0.558954 | 0.253747 | 0.519596 | 7.796 | reference |
| A1 QKV projections FP16 | 10/0 | 0.746780 | 0.556453 | 0.255797 | 0.519677 | 8.889 | safe |
| A2 LayerNorm FP16 | N/A | N/A | N/A | N/A | N/A | N/A | exact realization unavailable |
| A3 QK MatMul FP16 | 10/0 | 0.746878 | 0.559916 | 0.255906 | 0.520900 | 8.658 | safe |
| A4 Softmax FP16 | 10/0 | 0.747421 | 0.557203 | 0.258797 | 0.521140 | 8.771 | safe |
| A5 AV MatMul FP16 | 10/0 | 0.747453 | 0.557738 | 0.256356 | 0.520516 | 8.122 | safe |
| A6 output projection FP16 | 10/0 | 0.746858 | 0.559866 | 0.251729 | 0.519485 | 11.332 | safe |
| A7 residual Add FP16 | 10/0 | 0.746842 | 0.560099 | 0.255246 | 0.520729 | 7.755 | safe |

Smoke10 只用于灾难性配置排除，以上 p50 均不是正式隔离 latency。

## 组合实验

| Profile | evaluated/skipped | AP30 | AP50 | AP70 | mAP | 结论 |
|---|---:|---:|---:|---:|---:|---|
| M1 projections FP16 / core FP32 | 10/0 | 0.747284 | 0.560485 | 0.255081 | 0.520950 | safe |
| M2 projections+QK+AV FP16 / Softmax FP32 | 10/0 | 0.166951 | 0.140752 | 0.042732 | 0.116812 | catastrophic |
| M3 projections+AV FP16 / QK+Softmax FP32 | 10/0 | 0.746711 | 0.556479 | 0.257072 | 0.520087 | safe |
| M4 projection FP32 round-trip + QK FP16 | 10/0 | 0.166859 | 0.140935 | 0.040358 | 0.116051 | catastrophic |
| M5 projections+Softmax+AV+Out+Add FP16 / QK FP32 | 10/0 | 0.746778 | 0.559935 | 0.256074 | 0.520929 | safe |

M2 的 full diagnostic graph 中，LayerNorm 与 Q/K/V projection 误差仍较小，
QK MatMul 是首个出现 row-rank/cosine 明显不稳定的张量边界。M5 全 74 输出
diagnostic graph 会触发 TensorRT 10.9 optimizer `SIGSEGV`，因此使用固定的
`pre/qk/post` 三分片诊断；三片均完成 10/10、0 skip。分片图无效 padding
token 的 raw max error 较大，但 fused-BEV/head-input relative L2 约为
`6.6e-4/4.8e-4`，并且正式 fixed500 AP 保持稳定，故不把 padding max error
误判为模型失败。

## Fixed500 Production Candidates

参考 `attention_fp32_rest_fp16`：

```text
AP30/AP50/AP70/mAP = 0.779746/0.683858/0.483353/0.648986
screening p50/p90/p99 = 6.388/7.591/9.058 ms
```

| Profile | evaluated/skipped | AP30 | AP50 | AP70 | mAP | delta mAP | p50/p90/p99 ms | screening speedup |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| F1 core FP32 | 500/0 | 0.779063 | 0.683260 | 0.483769 | 0.648697 | -0.000288 | 5.244/8.076/16.072 | 1.218x |
| F2 LN/QK/Softmax/Add FP32 | 500/0 | 0.779740 | 0.683368 | 0.482884 | 0.648664 | -0.000322 | 5.165/9.166/16.625 | 1.237x |
| F3 LN/QK FP32 minimal island | 500/0 | 0.779364 | 0.683142 | 0.483711 | 0.648739 | -0.000247 | 5.187/9.814/18.787 | 1.232x |

三个候选均满足 fixed500 `delta mAP >= -0.003`。共享负载下 F2 的 p50 最低，
但 F2/F3 差异小于 1%，不能作为正式 latency 排序。F3 是最小 FP32 island，
因此作为自动搜索空间的推荐 contract；F2 保留为观测到的 screening-fastest
contract。

正式隔离 latency 未执行：GPU 2 上仍有 Pyramid persistent worker PID
`2346954` 和外部 Python PID `1396839`。本轮没有终止或干扰这些进程。证据见
`formal_latency_isolation.json`。

## 关键实现

- `search/model_families/lidar_cobevt/attention_precision_boundaries.py`
  - 集中定义 A0-A7、M1-M5、F1-F3；
  - 显式 role-to-node precision contract；
  - 强制 FP16 Softmax 输入具有显式 Cast，避免 RPE/mask 元数据漂移。
- `search/reporting/cobevt_attention_precision_inventory.py`
  - requested ONNX dtype 与 EngineInspector realized dtype 对账；
  - 处理融合层、reformat 和显式 Cast provenance。
- `search/model_families/lidar_cobevt/attention_tensor_parity.py`
  - diagnostic tensor 列表和 parity 指标；
  - QK rank/top-k、Softmax entropy/KL/JS、residual update retention；
  - deterministic pre/qk/post diagnostic shards。
- `search/integration/lidar_cobevt_attention_parity_worker.py`
  - fixedK29696、smoke10、8 DataLoader workers；
  - CUDA float32 vectorized parity metrics；
  - reference-only residual inputs，避免候选图输出 alias 崩溃。
- `search/orchestration/lidar_cobevt_attention_precision_audit.py`
  - prepare/build/evaluate/parity 全流程；
  - diagnostic engine 绑定 production TensorRT runtime；
  - full diagnostic 失败时自动分片并合并结果。
- `search/orchestration/lidar_cobevt_attention_pruning.py`
  - F1-F3 的外部 weighted graph FP16 基础 profile 接入。

关键提交：

```text
4642744 feat: diagnose CoBEVT attention FP16 boundaries
c6a7795 feat: add CoBEVT attention boundary interaction profile
75f524c feat: add CoBEVT rest-FP16 attention contracts
431a84e feat: add minimal CoBEVT FP32 attention island
b21cf48 fix: enforce explicit FP16 softmax boundaries
57bed5e fix: bind diagnostic engines to production TensorRT
a1ae6eb perf: keep attention parity metrics on GPU
635f6bf feat: shard CoBEVT attention tensor parity
```

## 机器可读证据

输出根目录包含：

```text
attention_precision_inventory.json
attention_precision_inventory.md
precision_boundary_matrix.json
precision_boundary_matrix.csv
tensor_parity_summary.json
tensor_parity_summary.csv
attention_failure_frames.json
formal_latency_isolation.json
root_cause_report.md
final_recommended_precision_contract.json
```

大型 ONNX、plan 和 diagnostic engines 只保存在 `/data/lxf/heal_data/outputs`，
不提交到 Git。

## 未完成项

1. A2 严格 FP16 LayerNorm 在 TensorRT 10.9 中无法精确 realization；当前保持
   FP32，不能声称已验证 FP16 安全或不安全。
2. GPU 2 未隔离，F1/F2/F3 的正式同 GPU 串行 latency 尚未完成。
3. 本轮只提出自动搜索空间 precision contract，不启动 GA/Greedy 搜索。

--- 2026-07-19 06:05:24 +0800 | Round 1: Attention FP16 boundary audit ---
