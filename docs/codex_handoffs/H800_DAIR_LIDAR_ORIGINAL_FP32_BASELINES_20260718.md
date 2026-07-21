# H800 DAIR LiDAR 原始 PyTorch / 严格 FP32 TensorRT 基线

## 本轮结论

已对 `Auto_Search/original_models/dairv2s/LiDAROnly` 下全部 7 个模型目录的唯一 `net_epoch_bestval_at*.pth` 完成原始 PyTorch 全验证集评估，并为每个模型独立导出、构建和评估了 strongly-typed、no-TF32 的严格 FP32 TensorRT engine。

两条路径均使用同一份 1789-frame manifest、200-frame warmup 后 reset、8-worker dataloader、相同 GPU 后处理、相同 frame 顺序；所有结果均为 1789/1789 evaluated、0 skipped。7 个 FP32 engine 相对各自 PyTorch 模型的最大绝对 mAP 差为 0.000222，因此严格 FP32 部署精度与原模型吻合。

## 输入模型

| 模型 | 最佳 checkpoint |
|---|---|
| lidar_attfuse | `net_epoch_bestval_at33.pth` |
| lidar_coalign | `net_epoch_bestval_at25.pth` |
| lidar_cobevt | `net_epoch_bestval_at19.pth` |
| lidar_disco | `net_epoch_bestval_at35.pth` |
| lidar_fcooper | `net_epoch_bestval_at37.pth` |
| lidar_pyramid | `net_epoch_bestval_at17.pth` |
| lidar_v2xvit | `net_epoch_bestval_at27.pth` |

每个 checkpoint 均以 `strict=True` 重新加载；精确 config/checkpoint 路径、SHA256、参数量和 state tensor 数见：

`outputs/h800_dair_lidar_original_fp32_baselines_20260718_2000/model_input_audit.json`

## 评估协议

- manifest：`outputs/H800_explicit_qdq_acceptance_20260714_023005/protocol_manifests_v2/eval_1789_warmup200_reset.json`
- manifest hash：`e5cbece0ceaf2ac1b2c47305a3fa3bdc5f616baa1ce86b0754ba565a013ac463`
- warmup：200 帧；warmup 后重新遍历验证集，不计入 AP/latency
- evaluation：完整 1789 帧，顺序与 DAIR validation split 完全一致
- dataloader workers：8，`persistent_workers=True`，`prefetch_factor=2`
- AP IoU 与 rotated NMS：CUDA 后端；最终 VOC AP reduction 保留 CPU
- latency：只统计模型/engine forward，不含数据加载、H2D 和后处理；同时记录 p50/p90/p99
- 固定 GPU：物理 GPU 7；各模型串行评估

原始 PyTorch 基线采用该 PyTorch 环境的默认执行语义：`matmul_allow_tf32=false`、`cudnn_allow_tf32=true`。TensorRT 基线则明确使用 `--stronglyTyped --noTF32`，因此表中加速比是用户要求的“原 PyTorch 模型 vs 严格 FP32 TensorRT engine”对比，不应解释成只改变容器格式的纯算子级对比。

## 完整结果

`PT p50/p90/p99` 和 `TRT p50/p90/p99` 单位均为 ms；加速比为 `PT p50 / TRT p50`。

| model | fixed K | PT AP30 | PT AP50 | PT AP70 | PT mAP | PT p50/p90/p99 | TRT AP30 | TRT AP50 | TRT AP70 | TRT mAP | TRT p50/p90/p99 | ΔmAP | p50 加速比 | engine weighted FP32 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| lidar_attfuse | 29164 | 0.741458 | 0.683204 | 0.540851 | 0.655171 | 7.464/9.722/17.069 | 0.741544 | 0.683242 | 0.540984 | 0.655257 | 6.947/7.148/7.404 | +0.000086 | 1.074x | 31 |
| lidar_coalign | 29164 | 0.809367 | 0.762224 | 0.599079 | 0.723557 | 8.132/12.117/24.584 | 0.809435 | 0.762304 | 0.598917 | 0.723552 | 6.875/7.091/7.391 | -0.000005 | 1.183x | 51 |
| lidar_cobevt | 29164 | 0.794357 | 0.702625 | 0.493911 | 0.663631 | 10.783/16.502/30.205 | 0.794395 | 0.702609 | 0.493870 | 0.663625 | 7.243/7.440/7.691 | -0.000006 | 1.489x | 66 |
| lidar_disco | 29164 | 0.734032 | 0.662470 | 0.509993 | 0.635499 | 7.198/11.038/26.440 | 0.734109 | 0.662376 | 0.510394 | 0.635626 | 6.802/6.984/7.156 | +0.000128 | 1.058x | 33 |
| lidar_fcooper | 29164 | 0.686976 | 0.584202 | 0.406372 | 0.559183 | 6.311/9.059/19.724 | 0.687213 | 0.584492 | 0.406135 | 0.559280 | 6.270/6.425/6.544 | +0.000097 | 1.007x | 29 |
| lidar_pyramid | 29164 | 0.826201 | 0.781652 | 0.602331 | 0.736728 | 10.013/15.090/27.580 | 0.826322 | 0.781678 | 0.602849 | 0.736950 | 6.604/6.792/20.634 | +0.000222 | 1.516x | 70 |
| lidar_v2xvit | 27612 | 0.781069 | 0.700086 | 0.519794 | 0.666983 | 21.565/29.908/56.968 | 0.781018 | 0.699987 | 0.519568 | 0.666858 | 16.390/16.978/20.853 | -0.000125 | 1.316x | 97 |

按本次统一协议，PyTorch mAP 排序为：Pyramid（0.736728）> CoAlign（0.723557）> V2XViT（0.666983）> CoBEVT（0.663631）> AttFuse（0.655171）> Disco（0.635499）> FCooper（0.559183）。

## fixed-K 审计

先对每个模型实际遍历完整 1789-frame validation split，再冻结 fixed-K，没有继承其他模型的历史文件名或 train200 上限：

- AttFuse/CoAlign/CoBEVT/Disco/FCooper/Pyramid：最大真实 voxel count 29164；
- V2XViT：最大真实 voxel count 27612；
- 两类最大值均出现在 frame `014509`；
- validation agent 分布：171 个单车样本、1618 个双车样本；
- 通用导出契约因此显式包含 `agent_mask`，同时验证 1/2 agent，不把所有样本伪装为双车。

完整统计见：

`outputs/h800_dair_lidar_original_fp32_baselines_20260718_2000/fixed_k_full_validation_audit.json`

## 严格 FP32 engine 证据

- TensorRT：10.9.0.34，root 为 `/home/lixingfeng/UniAD_examine/TensorRT-10.9_x86_cu118`
- 构建环境：`modelopt`
- Python：`/home/lixingfeng/miniconda3/envs/modelopt/bin/python3.10`
- nvcc：`/home/lixingfeng/miniconda3/envs/modelopt/bin/nvcc`
- GCC：`/home/lixingfeng/miniconda3/envs/modelopt/bin/x86_64-conda-linux-gnu-gcc`
- scatter plugin：`quantization/plugins/pointpillar_scatter_trt/build/libpointpillar_scatter_trt.so`
- plugin SHA256：`61d9adf44855ab2a595220718270d361c993f9ff281e986cdf8a62d5ca317ecd`
- builder flags：`--stronglyTyped --noTF32 --skipInference --staticPlugins=...`；不含 `--fp16` 或 `--int8`
- Inspector：所有 weighted compute 均为 FP32；7 个模型的 FP16/INT8/unknown 计数均为 0
- canonical structure validation 和 requested/realized precision validation：7/7 passed
- engine 均已反序列化并完成真实推理与后处理

短门禁中生成并验收 engine：

`outputs/h800_dair_lidar_trt_fp32_all_models_smoke_20260718_2138/`

正式全量评估只读复用这些 engine，未重复导出或构建：

`outputs/h800_dair_lidar_trt_fp32_full_validation_20260718_2222/`

每个模型目录中的 `reuse_provenance.json` 记录源 engine 路径和 SHA256，并明确 `source_engine_rebuilt=false`。

元数据说明：本轮已完成的 smoke artifact 生成时，复用的 snapshot helper 仍把 `model_family` 默认写为 `heal_lidar_v2xvit`。该字段不参与 engine build，只用于构建后的结构报告；snapshot 中逐模块名称/shape、canonical validation、engine 和全部 AP/latency 结果均不受影响。本轮 production 代码已把该 helper 改为可显式传入 model family，后续复现会写入正确模型名；没有为修复这一纯元数据字段而无理由重建相同 engine。

## 新增/修改代码

- `search/model_family/pytorch_evaluation_worker.py`
  - 严格 checkpoint 加载；统一 200-warmup/reset/1789-eval；CUDA AP/NMS；8-worker；forward/H2D/postprocess 分项延迟。
- `scripts/evaluate_dair_lidar_pytorch_baselines.py`
  - 自动发现 7 个模型的唯一 best checkpoint，并在固定 GPU 上串行执行 PyTorch 基线。
- `scripts/audit_dair_lidar_fixed_k.py`
  - 完整 validation split 的真实 voxel/agent 统计与 fixed-K 冻结。
- `search/model_family/export/heal_lidar_baselines.py`
  - 为 Max、Attention、Disco、CoBEVT 和多尺度 CoAlign 增加独立 fixed-K/fixed-agent 导出适配器；显式屏蔽 padded agent。
- `search/model_family/fp32_deployment_worker.py`
  - 真实帧 wrapper parity、ONNX checker、canonical mapping、strongly-typed no-TF32 build、Inspector 与全量评估入口。
- `search/model_family/deployment.py`
  - physical snapshot 接受显式 model-family 标识；V2XViT 默认行为不变，通用基线不再写入错误的 V2XViT 标签。
- `scripts/evaluate_dair_lidar_trt_fp32_baselines.py`
  - 逐模型串行导出/构建/短门禁；同一模型未完成时不启动下一个。
- `scripts/evaluate_existing_dair_lidar_trt_fp32.py`
  - 对已验收且 hash 一致的 engine 做全量评估，禁止相同配置无理由重建。
- `search/model_family/evaluation_worker.py`、`search/model_family/evaluation.py`
  - 在既有 V2XViT 六输入评估器中加入明确的通用 baseline input contract，不改变 Pyramid 五输入路径。
- `tests/test_dair_lidar_baseline_export.py`
  - fixed-K padding、单/双 agent mask、masked max/attention 和 policy fail-closed 测试。

既有 LiDAR Pyramid 和 V2XViT 生产导出类没有被覆盖；通用适配器位于独立文件中并按 model family 显式路由。

## 测试

以下两组相关测试共 40 项通过，`git diff --check` 通过：

```text
pytest -q tests/test_dair_lidar_baseline_export.py \
  tests/test_search_evaluation_provider.py \
  tests/test_search_model_family_v2xvit.py
21 passed

pytest -q tests/test_search_baseline_engines.py \
  tests/test_search_baseline_precision_validation.py \
  tests/test_search_exact_engine_reuse.py \
  tests/test_search_full_val_manifest.py
19 passed
```

## 复现命令

```bash
cd /home/lixingfeng/UniAD_examine/heal_compress
source /home/lixingfeng/miniconda3/etc/profile.d/conda.sh
conda activate univ2x-opt

CUDA_VISIBLE_DEVICES=7 python scripts/evaluate_dair_lidar_pytorch_baselines.py \
  --physical-gpu 7 --warmup-frames 200 --num-frames 1789 --workers 8 \
  --latency-isolation exclusive_gpu_verified_at_start \
  --output-dir outputs/<new_pytorch_baseline_dir>

python scripts/audit_dair_lidar_fixed_k.py \
  --manifest outputs/H800_explicit_qdq_acceptance_20260714_023005/protocol_manifests_v2/eval_1789_warmup200_reset.json \
  --output outputs/<new_dir>/fixed_k_full_validation_audit.json --workers 8

CUDA_VISIBLE_DEVICES=7 python scripts/evaluate_dair_lidar_trt_fp32_baselines.py \
  --fixed-k-audit outputs/<new_dir>/fixed_k_full_validation_audit.json \
  --eval-manifest outputs/H800_explicit_qdq_acceptance_20260714_023005/protocol_manifests_v2/eval_1789_warmup200_reset.json \
  --output-dir outputs/<new_trt_build_dir> --physical-gpu 7 \
  --warmup-frames 1 --num-frames 2 --workers 8

CUDA_VISIBLE_DEVICES=7 python scripts/evaluate_existing_dair_lidar_trt_fp32.py \
  --accepted-engine-root outputs/<new_trt_build_dir> \
  --models-root /home/lixingfeng/UniAD_examine/Auto_Search/original_models/dairv2s/LiDAROnly \
  --eval-manifest outputs/H800_explicit_qdq_acceptance_20260714_023005/protocol_manifests_v2/eval_1789_warmup200_reset.json \
  --output-dir outputs/<new_full_eval_dir> --physical-gpu 7 \
  --warmup-frames 200 --num-frames 1789 --workers 8
```

## 验收状态

```text
all_best_checkpoints_discovered: true
all_checkpoints_strict_loaded: true
full_validation_fixed_k_audited: true
pytorch_full_validation_complete: true
pytorch_all_1789_evaluated: true
pytorch_zero_skips: true
strict_fp32_onnx_export_complete: true
strongly_typed_no_tf32_build_complete: true
modelopt_toolchain_isolation_verified: true
all_weighted_layers_realized_fp32: true
strict_fp32_engine_full_validation_complete: true
strict_fp32_all_1789_evaluated: true
strict_fp32_zero_skips: true
all_abs_map_delta_le_0_001: true
identical_engines_not_rebuilt_for_full_eval: true
```

---

轮次时间戳：2026-07-19 02:35:25 CST（Asia/Shanghai）

---
