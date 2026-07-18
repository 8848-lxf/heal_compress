# 4090 CoBEVT Attention Dimension Pruning Report

## 结论摘要

```text
QK_ONLY_D_QK_STRUCTURE_LEGAL=true
FIXED_K=29696
FIXED_K_VALIDATED_SCOPE=full_validation
FULL_VALIDATION_MAX_K=29164
FULL_VALIDATION_K_OVERFLOW_COUNT=0
DEPLOYMENT_TOPOLOGY=single_engine_fixed_k
BUCKETED_ENGINES=false
OVERLIMIT_CHUNKING=false
OVERFLOW_POLICY=fail_closed
AP_PROTOCOL=fixed500_preliminary
FULL_VALIDATION_AP_EXECUTED=false
FORMAL_ISOLATED_LATENCY_AVAILABLE=false
FULL_MODEL_MIXED_INT8_EXECUTED=false
ATTENTION_SUBGRAPH_EXPLICIT_QDQ_INT8_EXECUTED=true
```

CoBEVT 正式实验固定为每个候选一个 `fixedK=29696` 单引擎，与 Pyramid
已经接受的单引擎 K 保持一致。完整 1789 帧 validation 点云计数扫描的最大
K 为 `29164`，在 K=29696 下超限数为 0。正式链路不实现分桶 engine、动态
多引擎路由、超限分片或静默截断；未来若出现超限，必须 fail closed 并重新
审核新的单引擎 K。

Q/K-only `d_qk=24/16` 和 B1 uniform `d_h=24/16` 均完成真实物理剪枝、
固定 500 帧 PyTorch FP32 评估、strongly typed TensorRT 构建和固定 500 帧
真实 GPU 推理。B2 全局 embedding `E=192,d_h=24` 结构合法，但未经恢复训练
的精度明显崩塌，不进入后续合法结构动作。

## 实验身份

- 分支：`feature/cobevt-attention-dim-pruning-audit`
- 入口代码提交：`0a308544e8c77539d08219955cc149e99aa506d7`
- checkpoint：`net_epoch_bestval_at19.pth`
- checkpoint SHA256：`67b0f2f00d74fea4912b4fdb902c1150146c733201e5dc36d3358f5ba605cfd4`
- config SHA256：`a0ee9d64fd1b01af95b1c997937ab07e0e810440236accfb598d5462ba086c33`
- fixed500 manifest SHA256：`5a2a81c05e635f71f277e9bd151dbb6d5dd0a8f9b08475d60511f67399824c19`
- Taylor calibration manifest SHA256：`535f4096168df2985229843bc4e08833ed2a032968ca101f1e8222be46df3a56`
- plugin SHA256：`91aec743dca383151b995a60d004cd254ce115ed16152873c1f46754bb15022d`
- TensorRT：10.9，CUDA：11.8，目标 GPU：RTX 4090 SM89
- 输出目录：`/data/lxf/heal_data/outputs/cobevt_attention_dim_pruning_20260718_085644`

fixed500 文件包含 20 个 warmup frame 和 500 个正式评估 frame，因此合约
中记录数为 520；正式结果只统计 warmup reset 后的 500 帧。

## Attention 结构

模型包含 6 个 Attention 模块：3 个 Window Attention 和 3 个 Grid
Attention。原始结构均为：

```text
E = 256
H = 8
d_qk = 32
d_v = 32
D_qk = 256
D_v = 256
to_qkv.weight = [768, 256]
to_out.weight = [256, 256]
```

物理剪枝 recipe 将原始 `chunk(3)` 替换为显式 Q/K/V split，从而允许
`d_qk != d_v`。Q/K-only 只裁剪 Q 和 K 投影对应输出行；B1 同时裁剪 Q/K/V
输出行和 `to_out` 输入列。Q 与 K 共用同一局部坐标，V 与 Out 共用另一组
局部坐标；QK 和 VO 不要求使用相同原始位置。

B2 会同步闭合 embedding 轴，包括上游 shrinker、LayerNorm、QKV、Out、
residual、FFN、后续 Transformer block、fusion head 和检测头。当前 B2 d24
虽通过 shape/forward/export/engine 审计，但零恢复训练 AP 不可接受。

## 一阶 Taylor 排名

排名来自原始 FP32 checkpoint 上的 50 个真实任务样本，micro-batch 为 1，
任务损失为 `L_cls + L_reg + L_dir + L_obj`。每个参数元素累计平均梯度
`g_i=mean_n(dL_n/dw_i)`，元素重要性为 `|w_i g_i|`。

Q/K 维度单元：

```text
I_QK(m,h,r) = sum_{WQ row, WK row, bias} |w_i g_i|
```

V/Out 维度单元：

```text
I_VO(m,h,r) = sum_{WV row, WV bias, WO column} |w_i g_i|
```

每个 Attention 模块、每个 head 独立排序。不同模块不共享排名，不同 head
可保留不同的原始局部位置，但每个 head 的最终宽度相同。本轮没有把量化
误差、SQNR、latency proxy 或二阶 Fisher 混入结构排名。

## 物理参数与结构审计

| candidate | d_qk | d_v | E | physical params | parameter reduction |
|---|---:|---:|---:|---:|---:|
| baseline d32 | 32 | 32 | 256 | 10,500,260 | 0.0000% |
| QK-only d24 | 24 | 32 | 256 | 10,303,652 | 1.8724% |
| QK-only d16 | 16 | 32 | 256 | 10,107,044 | 3.7448% |
| B1 uniform d24 | 24 | 24 | 256 | 10,107,044 | 3.7448% |
| B1 uniform d16 | 16 | 16 | 256 | 9,713,828 | 7.4896% |
| B2 global d24 | 24 | 24 | 192 | 9,435,940 | 10.1361% |

6 个候选均通过：物理参数 shape、Q/K/V split、head reshape、attention scale、
residual、FFN、finite output、ONNX checker、TensorRT deserialize 和 GPU
smoke。没有 mask 错位、split 错位、reshape 错位、NaN 或 Inf。

全模型 ONNX Runtime 未运行，因为图中包含 ONNX Runtime 未注册的
`PointPillarScatterTRT` 自定义算子；`onnx_parity.csv` 显式记录该限制。
相同 manifest 上 strict FP32 TensorRT 与 PyTorch mAP 差仅 `-0.000088`，
作为完整导出/插件/后处理链的端到端替代证据。

## 固定 500 帧 PyTorch FP32 精度

| candidate | AP30 | AP50 | AP70 | mAP | delta mAP |
|---|---:|---:|---:|---:|---:|
| baseline d32 | 0.780097 | 0.683205 | 0.483338 | 0.648880 | +0.000000 |
| QK-only d24 | 0.779814 | 0.681750 | 0.478528 | 0.646697 | -0.002183 |
| QK-only d16 | 0.776879 | 0.675640 | 0.467987 | 0.640169 | -0.008711 |
| B1 uniform d24 | 0.779025 | 0.680211 | 0.475760 | 0.644999 | -0.003881 |
| B1 uniform d16 | 0.776028 | 0.673530 | 0.470268 | 0.639942 | -0.008938 |
| B2 global d24 | 0.752835 | 0.561308 | 0.141705 | 0.485283 | -0.163597 |

全部候选为 500/500、0 skip。QK-only d24 和 d16 的初始精度下降分别为
`0.002183` 和 `0.008711`。QK-only 在相同宽度下略优于同步 QKV/Out B1，
说明只压缩注意力相似度子空间能少损失一点精度，但 B1 具有更大的参数和
潜在时延收益。

这些是固定 500 帧初步结果，不能称为 1789 帧 full-validation 结论。

## Strongly Typed 全模型引擎

所有正式引擎均使用 `--stronglyTyped --noTF32`，类型来自 typed ONNX、
显式 Cast 和部署 profile，没有使用 weak precision fallback。

strict FP32 control 的 canonical profile 为 65 FP32 + 1 受保护 FP16；同一
manifest 上 mAP 为 `0.648792`，与 PyTorch FP32 `0.648880` 对齐。strict
FP16 的 66 个 canonical weighted groups 均为 FP16，但 baseline mAP 降至
`0.402059`。定位实验表明崩塌来自 Attention projection 触发的 QK/mask/RPE/
Softmax/AV FP16 dtype propagation，而不是 fixedK、scatter、frontend、FFN、
head 或 AP 后处理。

因此本轮 accuracy-safe 对照将 24 个 Attention projection group 固定 FP32，
其余 42 个 weighted group 设为 FP16。该结果是 mixed profile，不得称为
strict FP16。全部 6 个引擎的 requested/realized precision 一致，unresolved
为 0，mismatch 为 0。

| candidate | AP30 | AP50 | AP70 | mAP | delta mAP | p50 | p90 | p99 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline d32 | 0.779746 | 0.683858 | 0.483353 | 0.648986 | +0.000000 | 6.388 | 7.591 | 9.058 |
| QK-only d24 | 0.779693 | 0.682241 | 0.477297 | 0.646411 | -0.002575 | 6.409 | 7.558 | 16.263 |
| QK-only d16 | 0.776816 | 0.674873 | 0.466569 | 0.639419 | -0.009566 | 6.151 | 7.114 | 8.485 |
| B1 uniform d24 | 0.778779 | 0.680786 | 0.477258 | 0.645608 | -0.003378 | 5.948 | 7.022 | 9.532 |
| B1 uniform d16 | 0.774184 | 0.672532 | 0.470640 | 0.639119 | -0.009867 | 5.506 | 6.231 | 7.477 |
| B2 global d24 | 0.752141 | 0.562050 | 0.141914 | 0.485368 | -0.163617 | 6.056 | 6.937 | 8.402 |

以上 latency 全部标记为 `screening_shared_gpu`。Pyramid GA controller 和
persistent workers 仍在 8 张卡上运行，因此这些数字不得当作正式独占 GPU
延迟或稳定加速比。当前正式 latency 状态为：

```text
no_idle_gpu_for_formal_latency
```

## Attention FP16/INT8 子图

microbenchmark 使用从真实 CoBEVT forward 捕获的 activation 和 mask：

```text
activation = [1, 2, 16, 32, 4, 4, 256]
mask       = [1, 16, 32, 4, 4, 1, 2]
```

扫描 10 个形状、pure Attention 和真实 mask/RPE 两类图、FP16 与 explicit-QDQ
INT8 两种精度，共 40 个 strongly typed engine；40/40 完成构建和真实 GPU
执行。INT8 权重为 symmetric per-output-channel，activation 为 per-tensor；
d32 图包含 11 个 QuantizeLinear 和 11 个 DequantizeLinear。是否实现 INT8
由 EngineInspector 审计，不由 engine build success 推断。

真实 mask/RPE 图的 uniform 结果：

| d_h | FP16 fused | FP16 p50 | INT8 fused | INT8 p50 |
|---:|---:|---:|---:|---:|
| 8 | no | 0.523 | no | 0.548 |
| 12 | no | 0.525 | no | 0.557 |
| 16 | yes | 0.473 | yes | 0.477 |
| 20 | no | 0.540 | no | 0.572 |
| 24 | yes | 0.499 | no | 0.575 |
| 28 | no | 0.549 | no | 0.580 |
| 32 | yes | 0.499 | yes | 0.485 |

由真实 mask/RPE + uniform shape + MHA fusion 三项共同筛选：

```text
FP16 kernel-friendly widths = [16, 24, 32]
INT8 kernel-friendly widths = [16, 32]
```

mask/RPE 并非一概阻止融合。FP16 的 16/24/32 和 INT8 的 16/32 能形成
MHA fusion；QK-only 非对称宽度均未融合。融合后 Softmax 位于内部 kernel，
EngineInspector 不单独暴露其 dtype，因此记录为 `fused_internal_unreported`，
不得宣称 Softmax 已独立实现 INT8。

这些 p50 同样只是共享 GPU screening latency。pure Attention 某些 fused
shape 的数值 parity 较弱；正式合法集合以真实 mask/RPE 图为依据。

## 推荐的后续合法集合

```text
legal_d_qk_pruning_widths = [16, 24, 32]
legal_uniform_dh_fp16_widths = [16, 24, 32]  # Attention 子图 shape 级
legal_uniform_dh_int8_widths = [16, 32]      # Attention 子图 realization 级
legal_global_embed_widths = []
```

前两个 microkernel 集合不能直接进入正式全模型 precision search：strict
FP16 Attention 的全模型精度不稳定，完整 CoBEVT mixed INT8 AP 也未执行。
QK-only/B1 结构动作已有完整单引擎固定 500 帧证据；B2 暂不开放。

CoBEVT 仍需要专用物理 pruning recipe。通用 Torch-Pruning 当前不能完整表达：

- Q/K 输出行的同坐标耦合；
- V 输出行与 Out 输入列耦合；
- QK 与 VO 可使用不同局部位置；
- 不等宽 Q/K/V 的显式 split；
- B2 embedding 的 residual/LayerNorm/FFN/head 全闭包。

## 产物

- `shape_audit.csv/json`：结构与输出审计；
- `parameter_audit.csv`：真实参数量与保留率；
- `ap500_results.csv/json`：PyTorch FP32 固定 500 帧；
- `full_engine_latency.csv`：所有 full-engine 评估与证据分类；
- `onnx_parity.csv`：ONNX checker 与 ORT custom-op 限制；
- `trt_parity.csv`：FP32 全模型 AP parity 和 40 个子图数值 parity；
- `int8_subgraph_build.csv`：40 个子图构建；
- `int8_precision_realization.csv`：Q/K/V/QK/AV/Out/MHA realization；
- `int8_subgraph_latency.csv`：20 warmup + 100 measured 的筛选延迟；
- `engine_layer_info/`：EngineInspector 原始证据；
- `fixed_k_contract.json`：完整 1789 帧单引擎 K 合约。

大型 ONNX、plan、gradient tensor 和缓存仅保留在 `/data` 输出目录，不提交 Git。

## 复现命令

```bash
OUT=/data/lxf/heal_data/outputs/cobevt_attention_dim_pruning_20260718_085644

conda run -n univ2x-opt python -m search.orchestration.lidar_cobevt_attention_pruning \
  --phase prepare --output-dir "$OUT" --device cuda:2 --gradient-samples 50
conda run -n univ2x-opt python -m search.orchestration.lidar_cobevt_attention_pruning \
  --phase fixed-k-scan --output-dir "$OUT"
conda run -n univ2x-opt python -m search.orchestration.lidar_cobevt_attention_pruning \
  --phase structure --output-dir "$OUT" --device cuda:2
conda run -n univ2x-opt python -m search.orchestration.lidar_cobevt_attention_pruning \
  --phase pytorch500 --output-dir "$OUT" --device cuda:2

conda run -n modelopt python -m search.orchestration.lidar_cobevt_attention_pruning \
  --phase export-build --output-dir "$OUT" --device cuda:2 --physical-gpu 2 \
  --diagnostic-profile attention_fp32_rest_fp16
conda run -n modelopt python -m search.orchestration.lidar_cobevt_attention_pruning \
  --phase trt500 --output-dir "$OUT" --physical-gpu 2 \
  --diagnostic-profile attention_fp32_rest_fp16
conda run -n modelopt python -m search.orchestration.lidar_cobevt_attention_microbenchmark \
  --output-dir "$OUT" --device cuda:2 --physical-gpu 2
```

这些 full-engine 命令始终为每候选一个 K=29696 engine；不存在 bucket 或
chunk 选项。

## 未验证项

- 按用户协议未运行 1789 帧 full-validation AP；
- Pyramid worker 活跃期间未运行独占 GPU formal latency；
- 未运行完整 CoBEVT mixed INT8 engine 的 AP/latency；
- 未对 B2 做恢复训练；
- 因此所有 AP 结论仍是固定 500 帧初步结论，所有 latency 仍是 screening。

------------------------------------------------------------
Round completed: 2026-07-19 02:27 CST
------------------------------------------------------------
