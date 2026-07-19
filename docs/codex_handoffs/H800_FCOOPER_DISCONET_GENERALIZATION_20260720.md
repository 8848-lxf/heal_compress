# H800 F-Cooper / DiscoNet 自动压缩泛化工作记录

## Round 1：fixed-K、搜索空间与真实部署 smoke

分支：`feature/heal-unified-search-h800`

本轮目标是在不改动 lidar_pyramid 专属链路的前提下，将 legal domain-width 剪枝、逐层显式 Q/DQ 混合精度、GPU batched proxy、Greedy/GA 和真实 TensorRT Stage-2 扩展到 HEAL 的 F-Cooper 与 DiscoNet。

### 已实现的生产代码

- 新增 F-Cooper / DiscoNet model-family audit、量化能力、FP16 fusion-island/merge contract：
  - `search/model_family/heal_lidar_baselines.py`
  - `search/model_family/heal_lidar_deployment.py`
  - `search/model_family/heal_lidar_pruning.py`
- 新增 family context、正式 orchestration 和真实 Stage-2 evaluator：
  - `search/integration/heal_lidar_baseline_context.py`
  - `search/orchestration/heal_lidar_baseline_search.py`
  - `search/stage2/heal_lidar_baseline_real_evaluator.py`
- 新增四份正式六预算配置：
  - `search/configs/heal_lidar_fcooper_h800_domain_width_joint_{greedy,ga}.yaml`
  - `search/configs/heal_lidar_disco_h800_domain_width_joint_{greedy,ga}.yaml`
- 新增共同 train200 固定输入构建脚本：
  - `scripts/build_heal_lidar_baseline_train200_calibration.py`
- 修复 formal CLI 的 family routing、Stage-1 包装候选读取、physical-model audit，以及 family closure index-map 传递。
- Greedy 保留主单路径的 Taylor/BOPS 动作选择，但把所有已批量评估的一跳邻居纳入严格预算池；仍缺预算时使用 deterministic narrow-beam 合法动作恢复，硬门槛始终为 `|R_BOPS-target| <= 0.005`。

### fixed-K 结论

- validation 最大真实 voxel K：29164。
- train200 的第 199 个样本达到 29562，旧 fixedK29164 会截断，因此不能作为共同 calibration contract。
- 冻结共同 `fixedK=29696`（覆盖 train200/validation 且按 256 对齐）。
- 权威 manifest：
  - `outputs/heal_lidar_baseline_train200_fixedk29696_20260719_1215_v2/calibration_manifest.json`
  - 200/200，observed max=29562，SHA256=`a08b1b1a4890c9a729531e27cd0f94098e82ae70f102f9de7b2c3739abc869f6`

### 搜索空间静态结果

- F-Cooper：3840 atomic units，21 legal width domains，28 precision genes，24 可选 INT8，4 个受保护层。
- DiscoNet：4000 atomic units，23 legal width domains，32 precision genes，24 可选 INT8，8 个受保护层；pixel-weight soft-fusion weighted island 固定 FP16。
- 两侧正式 Stage-1 均使用 CUDA batched proxy；F-Cooper GPU pool `[0,3,4]`，DiscoNet `[5,6,7]`。

### 六预算 Greedy Stage-1

- F-Cooper 六个预算全部达到硬门槛：
  - run: `outputs/h800_heal_lidar_fcooper_greedy_stage1_frontierfix_fixedk29696_20260719/h800_heal_lidar_fcooper_domain_width_joint_greedy_20260719_085626`
- DiscoNet 六个预算全部达到硬门槛：
  - run: `outputs/h800_heal_lidar_disco_greedy_stage1_beamfix_fixedk29696_20260719/h800_heal_lidar_disco_domain_width_joint_greedy_20260719_090652`
  - 0.25 预算由一层 narrow-beam 恢复，实际 `R_BOPS=0.2545315`，偏差 `0.0045315`，额外 369 个 GPU-batched neighbor。

### 真实非零子网部署 smoke

两个 0.10 候选均走过：phenotype → formal request → plan → materialize → strict replay → physical audit → ONNX → fresh train200 EntropyCalibration2 → per-channel weight Q/DQ → strongly typed TensorRT → precision/merge audit → 10-frame CUDA-postprocess evaluation。

- F-Cooper：
  - artifact: `outputs/h800_heal_lidar_fcooper_real_stage2_smoke_v3_fixedk29696_20260719/h800_heal_lidar_fcooper_domain_width_joint_greedy_20260719_092010/stage2_only/10546cdc59d16d7aa5505dd7121450a831fa5f33beb365fc306bd253746c8153`
  - physical parameter pruning=23.4568%；requested/realized INT8=2/2；10/10，0 skip；mAP=0.737617；forward p50=2.282 ms。
- DiscoNet：
  - artifact: `outputs/h800_heal_lidar_disco_real_stage2_smoke_v3_fixedk29696_20260719/h800_heal_lidar_disco_domain_width_joint_greedy_20260719_092010/stage2_only/b0206801e129508baef6dc85257522329c9416767d40d56040f50071aea5d1d2`
  - physical parameter pruning=60.1178%；requested/realized INT8=10/10；10/10，0 skip；mAP=0.739416；forward p50=2.772 ms。

两侧 `physical_acceptance`、`qdq_acceptance`、`engine_acceptance`、`precision_acceptance`、`merge_acceptance`、`evaluation_acceptance` 均为 true，DataLoader workers=8，AP IoU/NMS 使用 CUDA 后端。

### 已定位并修复的关键问题

1. train200 K 大于 validation K：从 29164 提升并冻结到 29696。
2. Greedy 只保存选中主路径，丢弃已计算的预算内邻居：加入 evaluated frontier 与 strict-band narrow-beam recovery。
3. Stage-2-only 子类重复实现旧 candidate loader，把包装候选变成空 genotype：统一使用 `_load_candidate()`。
4. physical materialization 成功后又按“原始宽度”做 provider audit：加入显式 `require_original_widths=False` physical audit，原始 checkpoint audit 仍 fail-closed。

### 测试

- 当前相关回归最高记录：51 passed（另一次针对最新 physical-audit/部署组合为 34 passed），`git diff --check` 通过。
- 新增/更新测试覆盖：family detection/audit、pruning closure、strict replay、Disco concat double-half map、Q/DQ mapping、entropy calibration、Stage-1 wrapper loader、Greedy multi-action budget recovery、physical audit。

### 尚未完成

- Greedy 六预算完整 1789-frame 结果。
- GA 六预算、每预算 Top-5 的 500-frame Stage-2、round winner 的 1789-frame full validation。
- 最终 F-Cooper/DiscoNet Greedy-vs-GA 汇总与远端提交确认。

---
时间戳：2026-07-20 00:27:45 CST｜轮次：Round 1
---

## Round 2：Greedy 六预算真实 full validation

两模型均完成 6 个唯一 phenotype 的真实 physical pruning → explicit Q/DQ → strongly-typed TensorRT → 1789-frame full validation；每个候选均为 1789/1789、0 skip，且 physical/QDQ/engine/precision/merge/evaluation acceptance 全部通过。共同协议为 fixedK29696、warmup=200 后 reset、latency rounds=3、DataLoader workers=8 和 CUDA IoU/NMS 后处理。

F-Cooper run：
`outputs/h800_heal_lidar_fcooper_greedy_full_fixedk29696_20260720/h800_heal_lidar_fcooper_domain_width_joint_greedy_20260719_093214`

| BOPS预算 | 实际BOPS | 参数剪枝 | INT8/FP16/FP32 | mAP | AP30/50/70 | forward mean/p50/p90/p99 (ms) | 相对FP32加速 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.05 | 0.054306 | 27.56% | 13/15/0 | 0.559043 | 0.686512/0.583102/0.407516 | 2.343/2.330/2.423/2.507 | 2.679x |
| 0.10 | 0.101379 | 25.92% | 2/26/0 | 0.558855 | 0.686572/0.584133/0.405861 | 2.345/2.308/2.448/2.855 | 2.704x |
| 0.15 | 0.154921 | 22.64% | 0/25/3 | 0.558888 | 0.686244/0.584334/0.406086 | 3.104/3.091/3.177/3.304 | 2.019x |
| 0.20 | 0.204398 | 20.48% | 0/20/8 | 0.559181 | 0.687120/0.584350/0.406075 | 3.518/3.490/3.613/3.866 | 1.788x |
| 0.25 | 0.249953 | 19.21% | 0/16/12 | 0.559132 | 0.687075/0.584370/0.405951 | 3.673/3.644/3.759/4.106 | 1.713x |
| 0.30 | 0.303690 | 19.42% | 0/19/9 | 0.559312 | 0.687020/0.584519/0.406395 | 4.186/4.179/4.249/4.386 | 1.493x |

F-Cooper strict-FP32 reference：mAP=0.559322，forward mean/p50=6.250/6.241 ms。

DiscoNet run：
`outputs/h800_heal_lidar_disco_greedy_full_fixedk29696_20260720/h800_heal_lidar_disco_domain_width_joint_greedy_20260719_093214`

| BOPS预算 | 实际BOPS | 参数剪枝 | INT8/FP16/FP32 | mAP | AP30/50/70 | forward mean/p50/p90/p99 (ms) | 相对FP32加速 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.05 | 0.054899 | 70.64% | 12/20/0 | 0.635768 | 0.733912/0.662967/0.510425 | 2.510/2.489/2.614/2.899 | 2.670x |
| 0.10 | 0.104676 | 58.41% | 10/19/3 | 0.635819 | 0.734232/0.662833/0.510392 | 2.635/2.625/2.710/2.800 | 2.531x |
| 0.15 | 0.154722 | 50.95% | 1/24/7 | 0.635614 | 0.734059/0.662542/0.510241 | 2.897/2.881/2.973/3.244 | 2.307x |
| 0.20 | 0.202822 | 50.02% | 1/21/10 | 0.635723 | 0.734146/0.662581/0.510442 | 3.257/3.244/3.333/3.553 | 2.049x |
| 0.25 | 0.246012 | 40.71% | 0/20/12 | 0.635630 | 0.734149/0.662540/0.510199 | 3.842/3.835/3.905/4.018 | 1.733x |
| 0.30 | 0.302809 | 49.12% | 1/20/11 | 0.635682 | 0.734200/0.662706/0.510141 | 4.107/4.094/4.190/4.416 | 1.623x |

DiscoNet strict-FP32 reference：mAP=0.635673，forward mean/p50=6.657/6.645 ms。

GPU 调度核验：首次 DiscoNet Greedy 旧配置在主 GPU 5 运行，并在 GPU 6/7 各保留约 1.14 GiB 的空闲 CUDA context；进程结束后 5/6/7 已全部释放。后续正式配置已改为 F-Cooper 与 DiscoNet 顺序共享 `[0,3,4]`，不再占用 5/6/7。

GA 状态：F-Cooper 正式六预算 GA 已在 `[0,3,4]` 启动；DiscoNet 将在其完成后顺序启动，禁止两个模型抢占同一 GPU pool。

---
时间戳：2026-07-20 01:12:44 CST｜轮次：Round 2
---

## Round 3：F-Cooper GA 六预算与并发部署修复

F-Cooper 正式 GA 已完成 6 个 BOPS budget、每轮 repaired Top-5 的 500-frame Stage-2，以及 6 个唯一 round winner 的 1789-frame full validation。30/30 个 Stage-2 候选均成功；最终复评为 1789/1789、0 skip，`candidate_engine_rebuild_count=0`，precision/merge acceptance 全部通过。

run：
`outputs/h800_heal_lidar_fcooper_ga_full_v2_fixedk29696_20260720/h800_heal_lidar_fcooper_domain_width_joint_ga_20260719_140402`

| BOPS预算 | 实际BOPS | 参数剪枝 | INT8/FP16/FP32 | mAP | AP30/50/70 | forward mean/p50/p90/p99 (ms) | 相对FP32加速 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.05 | 0.054521 | 26.81% | 13/15/0 | 0.558951 | 0.686385/0.583211/0.407257 | 2.312/2.302/2.393/2.464 | 2.666x |
| 0.10 | 0.104221 | 24.99% | 5/22/1 | 0.558315 | 0.686240/0.583448/0.405257 | 2.387/2.370/2.483/2.591 | 2.589x |
| 0.15 | 0.154876 | 22.90% | 0/24/4 | 0.558893 | 0.686731/0.584199/0.405750 | 3.085/3.065/3.179/3.367 | 2.002x |
| 0.20 | 0.204197 | 19.19% | 0/21/7 | 0.559412 | 0.687218/0.584856/0.406162 | 3.350/3.338/3.437/3.534 | 1.839x |
| 0.25 | 0.251769 | 18.96% | 0/16/12 | 0.559165 | 0.687251/0.584451/0.405795 | 3.606/3.593/3.705/3.808 | 1.708x |
| 0.30 | 0.304923 | 18.80% | 0/19/9 | 0.559231 | 0.686947/0.584369/0.406377 | 4.155/4.149/4.228/4.297 | 1.479x |

本轮修复了首次正式 GA 暴露的两个生产问题，并删除了旧失败 run，未复用其产物：

1. 多 GPU worker 线程并发调用 PyTorch 2.0 `torch.onnx.export` 会争用进程全局 exporter 状态并触发空消息 `AssertionError`。`search/model_family/heal_lidar_deployment.py` 现在只对 ONNX export 临界区加进程内锁；calibration、Q/DQ、TensorRT build 和 evaluation 仍可跨 GPU 并行。
2. family evaluator 原先只写 `candidate_stage2_result.json`，通用 round 聚合器只读取 `stage2_score.json`；现同步写两份兼容结果。physical checkpoint 也正式落盘，失败结果记录完整 traceback。
3. `search/stage2/round_results.py` 现支持 family 的 nested artifact layout，将 physical checkpoint、base ONNX、Q/DQ ONNX、engine 和 evaluation 正确复制为 round winner artifact。
4. 新增回归测试覆盖 ONNX export serialization、dual Stage-2 result、nested winner artifact copy；相关定向测试 30 passed，`git diff --check` 通过。
5. GPU 配置统一为 `[0,3,4]`，F-Cooper 与 DiscoNet 串行运行，GPU 5/6/7 保持空闲。

DiscoNet 正式 GA 已在 F-Cooper 完成后启动，尚未把运行中结果标记为完成。

---
时间戳：2026-07-20 05:49:30 CST｜轮次：Round 3
---

## Round 4：DiscoNet GA 完整验收与最终泛化结论

DiscoNet 正式 GA 已完成 6 个 BOPS budget、每轮 repaired Top-5 的 500-frame Stage-2，以及 6 个唯一 round winner 的 1789-frame full validation。30/30 个 Stage-2 候选均成功；最终复评为 1789/1789、0 skip，`candidate_engine_rebuild_count=0`，physical/QDQ/engine/precision/merge/evaluation acceptance 全部通过。

run：
`outputs/h800_heal_lidar_disco_ga_full_v4_fixedk29696_20260720/h800_heal_lidar_disco_domain_width_joint_ga_20260719_151030`

| BOPS预算 | 实际BOPS | 参数剪枝 | INT8/FP16/FP32 | mAP | AP30/50/70 | forward mean/p50/p90/p99 (ms) | 相对FP32加速 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.05 | 0.054949 | 69.15% | 12/20/0 | 0.634754 | 0.732773/0.661795/0.509695 | 2.460/2.443/2.553/2.723 | 2.698x |
| 0.10 | 0.104880 | 59.80% | 9/20/3 | 0.635847 | 0.734197/0.663027/0.510316 | 2.599/2.582/2.701/3.010 | 2.553x |
| 0.15 | 0.154082 | 53.58% | 1/24/7 | 0.635691 | 0.734104/0.662671/0.510299 | 2.860/2.844/2.946/3.279 | 2.318x |
| 0.20 | 0.200672 | 50.06% | 0/23/9 | 0.635622 | 0.734045/0.662537/0.510283 | 3.623/3.610/3.693/3.970 | 1.826x |
| 0.25 | 0.250668 | 41.29% | 0/20/12 | 0.635669 | 0.733969/0.662505/0.510532 | 3.791/3.786/3.867/3.989 | 1.741x |
| 0.30 | 0.304401 | 49.04% | 0/22/10 | 0.635686 | 0.734061/0.662669/0.510327 | 4.343/4.340/4.407/4.465 | 1.519x |

所有实际 BOPS 均满足 `|R_BOPS-target| <= 0.005`。DiscoNet strict-FP32 reference mAP=0.635673；GA 六点相对该基线的绝对 mAP 差为 0.000005～0.000919。与 Greedy 对应预算相比，GA 的 mAP 差为 -0.001014～+0.000077；除 0.20/0.30 的 precision/pruning 组合带来约 +0.366/+0.247 ms p50 外，其余 p50 差在 0.05 ms 左右。

### DiscoNet 并发 parity 根因和修复

首次运行暴露了两个不同量级的问题；失败 run 均已停止并删除，没有复用 calibration、Q/DQ 或 engine：

1. HEAL `PFNLayer.forward` 会切换进程全局 `torch.backends.cudnn.enabled`。多 GPU Stage-2 使用同进程线程时，仅串行 `torch.onnx.export` 不足，reference/wrapper parity 前向仍会竞争，曾产生 mean≈1e-3、max≈8.9e-3 的系统性偏差。现在 `search/model_family/heal_lidar_deployment.py` 将 parity 两次前向和 ONNX export 放入同一串行临界区，calibration/QDQ/TRT/evaluation 继续跨 GPU 并行。
2. 消除竞争后，fixed-K29696 padded GEMM/scatter 与原始变长输入仍存在稀疏舍入差异：mean 通常约 1e-6，局部 max 可到约 0.0039。旧门槛 `max<=0.002 && mean<=1e-5` 会误拒绝。新门槛同时要求 `max<=0.005 && mean<=5e-5`：允许局部舍入，但先前 mean≈1e-3 的系统漂移仍会失败。
3. 新增单测证明稀疏 0.0035 偏差通过、全张量 0.001 系统漂移失败；锁、parity、family result 和 nested winner artifact 定向组合为 23 passed。

### 最终泛化结论

- F-Cooper 和 DiscoNet 均完成 Greedy+GA 两种搜索、六个严格 BOPS budget、真实非零物理剪枝、explicit Q/DQ、strongly-typed TensorRT、precision/merge checker 和 1789-frame full validation。
- Greedy：12/12 最终候选成功；GA：60/60 Top-5 Stage-2 候选成功，12/12 round winner 完整复评成功。
- 全部正式结果为 0 skip，DataLoader workers=8，AP IoU/NMS 使用 CUDA；GA 相同 winner engine 不重建。
- GPU 调度仅使用 `[0,3,4]`，GPU 5/6/7 在长任务期间保持空闲。
- 结论：当前 legal-domain-width + precision-gene + merge-contract 框架已在两种额外 CNN/软融合 HEAL 模型上获得真实部署级泛化证据，不再只是 smoke 或静态兼容声明。

---
时间戳：2026-07-20 07:13:50 CST｜轮次：Round 4
---
