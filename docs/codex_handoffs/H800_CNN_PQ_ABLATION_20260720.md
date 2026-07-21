# H800 CNN 搜索子网 P/Q 解耦消融

分支：`feature/heal-unified-search-h800`

## Round 1：产物审计、通用构建/评估链路与 P-only 合同修复

本轮先核验当前服务器真实文件，未把旧 handoff 当作完成证据。

- `lidar_pyramid` 已有五轮公平消融可直接复用：
  `outputs/h800_lidar_pyramid_split_gpu_ablation_repeat5_20260718_181347/`。
  190/190 次评估均为 1789 evaluated、0 skipped，逐帧 CSV 共
  339910 行；33 个唯一 engine 在本机重新按 size/SHA256 核验通过。
- F-Cooper 与 DiscoNet 的 GA/Greedy 六预算 P+Q engine 和单次
  1789-frame full validation 存在，但两模型的 P-only/Q-only 均不存在。
  每模型缺少 24 个逻辑变体；不能把联合 P+Q 结果当作消融已完成。

新增或修改的正式代码：

- `search/ablation/heal_lidar_prune_quant.py`
  - 从真实 GA round winner 和 Greedy full-validation artifact 收集 12 个
    accepted source phenotype；
  - P+Q 保持原结构/量化合同并只读复用原 engine；
  - P-only 保持原 mask，weighted、semantic merge 和 fusion auxiliary
    全部使用 strict FP32；
  - Q-only 恢复 all-keep 原结构，保持原 precision/merge profile；
  - 按包含 auxiliary/merge 语义的 deployment signature 去重。
- `search/stage2/heal_lidar_baseline_real_evaluator.py`
  - 新增正式 `build_candidate_artifacts()` build-only API；
  - 构建阶段只执行 materialize → ONNX → fresh train200 calibration →
    explicit Q/DQ → strongly-typed TensorRT，不执行评估。
- `search/model_family/heal_lidar_deployment.py`
  - auxiliary precision 成为 mapping、Q/DQ graph metadata、deployment
    policy 和 EngineInspector realization checker 的显式合同；
  - 正常 P+Q/Q-only 仍要求 FP16 auxiliary，P-only 单独要求 FP32。
- `search/model_family/evaluation_worker.py`
  - 每帧 latency row 同时写入 `phase`、`warmup` 与
    `total_ms=forward_ms+postprocess_ms`，防止 warmup 混入正式 CSV。
- `scripts/run_heal_lidar_prune_quant_ablation.py`
  - `prepare/build-one/build-all/finalize`；
  - 单模型单 GPU 串行构建、同 context 复用、逐 owner 原子进度记录和
    SHA256 fail-closed resume。
- `search/integration/heal_lidar_family_fair_evaluation.py`
  与 `scripts/run_heal_lidar_family_split_gpu_fair_evaluation.py`
  - family-aware evaluation-only 五轮复评；
  - GA/Greedy 分卡、单卡串行；每项独立 modelopt worker；
  - 1789 帧、warmup 200 后 reset、latency rounds 3、workers 8、CUDA
    postprocess；
  - 每项后 empty-cache/IPC cleanup 和显存回落审计；
  - engine SHA256/size/mtime 前后锁定；
  - fail-closed `--run-dir` resume；
  - 逐帧 forward/postprocess/total CSV 与五轮 mean/std。
- `scripts/summarize_three_model_pq_ablation.py`
  - 适配 Pyramid 与 family 两种 schema；
  - 强制三模型 114 个 aggregate rows、每模型 190 个 raw repeat rows、
    5×1789、0 skip、同 manifest/hash/frame order 后才生成统一报告。

首次真实 P-only 构建暴露并保留了一组失败证据：

- `outputs/h800_heal_lidar_fcooper_pq_ablation_build_20260720_043453/`
- `outputs/h800_heal_lidar_disco_pq_ablation_build_20260720_043453/`

失败不是 weighted precision 错误；weighted checker 已为 0 INT8/0 FP16、
无 mismatch。直接根因是旧 family checker 无条件要求 fusion island FP16，
与 P-only 的 strict-FP32 因果消融定义冲突。修复后新建 v2 目录，未删除、
覆盖或复用上述失败 artifact：

- `outputs/h800_heal_lidar_fcooper_pq_ablation_build_v2_20260720_044301/`
- `outputs/h800_heal_lidar_disco_pq_ablation_build_v2_20260720_044301/`

两个 0.30 P-only 真实 GPU 回归均通过：weighted precision 为 strict FP32，
fusion/merge required precision 为 FP32，EngineInspector acceptance=true，
`evaluation_invoked=false`。随后已在 GPU2/GPU5 启动单卡串行 build-all；
GPU3/GPU6 留给后续各自 Greedy 评估，GPU7 保持空闲。

环境 manifest 抽查：TensorRT 10.9.0.34；modelopt Python；modelopt CUDA
11.8 nvcc；modelopt GCC/G++ 11.2；H800 CC 9.0；plugin SHA256 已记录；
`system_toolchain_used=false`；builder 为 strongly-typed/no-TF32。

测试：相关 build/ablation/deployment/fair-evaluation/Pyramid regression
组合 34 passed；三模型汇总器 3 passed；`git diff --check` 通过。

当前状态：v2 engine 构建仍在进行，尚未把 F-Cooper/DiscoNet 的五轮完整
评估或三模型最终汇总标记为完成。

---
时间戳：2026-07-20 19:54:58 CST｜轮次：Round 1
---

## Round 2：F-Cooper/DiscoNet 消融 engine 矩阵完成并启动公平复评

本轮完成两个模型全部缺失的 P-only/Q-only 真实部署构建。所有新部署均使用
正式 production build-only 入口，未由消融脚本手工拼接 engine；P+Q 保持对
原搜索 winner engine 的只读引用。

- F-Cooper：36 个逻辑项（GA/Greedy × 6 预算 × 3 variants），33 个唯一
  engine；其中 21 个 P-only/Q-only engine 为本轮 fresh build。
- DiscoNet：36 个逻辑项，34 个唯一 engine；其中 22 个 P-only/Q-only
  engine 为本轮 fresh build。
- 每个新建结果均为 `status=ok`、`evaluation_invoked=false`；正式 finalize
  后逐 engine 重新核验存在性、非零大小与 SHA256，全部与
  `engine_inventory.json` 一致。
- P-only 的 weighted layers、semantic merge 与 fusion auxiliary 均按 strict
  FP32 合同检查；Q-only 为 all-keep 结构并继承源 winner precision/merge
  profile。TensorRT 构建仍为 strongly typed、no-TF32，并记录 modelopt
  CUDA/GCC/TensorRT/plugin provenance。

真实目录：

- `outputs/h800_heal_lidar_fcooper_pq_ablation_build_v2_20260720_044301/`
- `outputs/h800_heal_lidar_disco_pq_ablation_build_v2_20260720_044301/`

随后已启动五轮统一 full validation：固定 1789 帧、warmup 200 后 reset、
0 skip 要求、latency rounds 3、DataLoader workers 8、CUDA postprocess；每个
engine 记录逐帧 forward/postprocess/total latency，并在评估后执行独立 CUDA
cache cleanup 与 engine hash/size/mtime 前后核验。

- F-Cooper：GPU2=GA、GPU3=Greedy，目录
  `outputs/h800_fcooper_pq_fair_repeat5_20260720_052546/`。
- DiscoNet：GPU4=GA、GPU5=Greedy，目录
  `outputs/h800_disconet_pq_fair_repeat5_20260720_053205/`。
- 四张卡各自只运行一条串行评估队列；评估期间绝不构建 engine。GPU6/7
  未由本任务占用。

当前状态：两模型五轮 full validation 正在进行，尚未标记最终汇总完成。

---
时间戳：2026-07-20 20:33:01 CST｜轮次：Round 2
---

## Round 3：外部 GPU 抢占 fail-closed 与跨方法安全恢复

DiscoNet 复评运行约 75 分钟后，另一用户进程开始占用已经分配给本任务的
GPU4。该事件发生在 GA repeat 2 的 P+Q 项结束附近；评估 worker 退出后显存
从本任务启动前的 4 MiB 变为 14019 MiB，50 次回落轮询均不满足阈值，因此
`gpu_cleanup_audit.json` 明确记录 `passed=false`。正式 GA 流 fail-closed 停在
39/95，没有继续加载下一 engine；该项及该 GPU4 GA run 不进入最终统计。未
终止或干预其他用户进程。GPU5 Greedy 流未受影响，继续串行运行。

为既拒绝污染数据、又不重复 GPU5 的有效五轮结果，增强
`scripts/run_heal_lidar_family_split_gpu_fair_evaluation.py`：

- 新增 `--seed-method-root METHOD=RUN_DIR`；逐项验证 family、协议、engine
  inventory/SHA256、原物理 GPU、repeat/item identity、1789/0、逐帧 CSV
  SHA256、engine 不变和 cleanup passed 后，才把完整方法流硬链接到新 run；
  复制后再次执行同一套 resume 校验。
- 新增 `--ga-extra-gpu`/`--greedy-extra-gpu` whole-repeat sharding。不同 GPU
  只分担完整 repeat，每个 repeat 均包含该卡自己的 strict-FP32 baseline 和
  18 个串行 variants；同一 repeat 内不跨卡计算 speedup。
- GPU pool 强制全局互斥，preflight 对每张卡检查空闲；同卡仍严格串行。

计划恢复：F-Cooper 释放 GPU2/3、DiscoNet Greedy 完成并释放 GPU5 后，新建
独立 DiscoNet final run；GPU2 跑 GA repeat 0/2/4，GPU3 跑 GA repeat 1/3，
GPU5 只 seed/resume 已验证 Greedy 结果。旧受干扰目录作为失败证据保留，不
删除、不覆盖。

新增恢复路径单元测试（完整 stream hardlink + 复制前后 fail-closed resume
校验、GPU pool 重叠拒绝）通过：8 passed；`git diff --check` 通过。

---
时间戳：2026-07-20 21:55:37 CST｜轮次：Round 3
---

## Round 4：三模型 P+Q / P-only / Q-only 五轮完整验证收口

F-Cooper 与 DiscoNet 的 GA/Greedy、6 个 BOPS 预算、3 种消融均已完成
五轮独立 full validation。每个 repeat 在同一物理 GPU 上先评 strict-FP32
baseline，再串行评 18 个候选；每项结束后执行独立 CUDA cleanup，评估期间
不构建 engine。协议统一为 1789 帧、warmup 200 后 reset、latency rounds 3、
DataLoader workers 8、CUDA postprocess。两个模型均为 190/190 项 `status=ok`，
每项 1789 evaluated、0 skipped、cleanup passed，且 engine SHA256/size/mtime
前后不变。

最终可信目录：

- F-Cooper：
  `outputs/h800_fcooper_pq_fair_repeat5_20260720_052546/`
  - 逐帧 339,910 行；SHA256
    `c668f04c1d0e9c14641479f4398ba66bfb3252089e1693b6f45e666c8141ef74`。
- DiscoNet：
  `outputs/h800_disconet_pq_fair_repeat5_20260720_233347/`
  - Greedy 95 项从已完整通过且只读锁定的首次 run 验证后硬链接；GA 95 项
    在 GPU7 fresh 评估；逐帧 339,910 行；SHA256
    `e66293255e157392c05887ce2b58e837cf203e5e2ef0b81e1e789e5b68201545`。
- lidar_pyramid（既有公平五轮结果）：
  `outputs/h800_lidar_pyramid_split_gpu_ablation_repeat5_20260718_181347/`。
- 三模型统一报告：
  `outputs/h800_three_model_pq_ablation_summary_20260721_184542/`
  - `three_model_pq_ablation_summary.csv`：114 个 aggregate rows；
  - 同目录 JSON/Markdown 含完整 AP、mean/p50/p90/p99、逐帧来源 hash、
    参数/BOPS/混合权重、覆盖率、压缩比和 speedup；
  - 三模型 validation manifest SHA256 均为
    `e5cbece0bf43bc20995bb19af39ace4f31306444258d3365d235c602315b4463`，
    frame 顺序 hash 均为
    `6f09fef1c702f5310043c8ec8ffb208b26382b4c07f362417530572007a217b1`。

外部 GPU 抢占导致的 DiscoNet 非完整目录没有被静默纳入统计，继续保留：

- `outputs/h800_disconet_pq_fair_repeat5_20260720_053205/`（GA 39/95）；
- `outputs/h800_disconet_pq_fair_repeat5_20260720_225305/`（恢复到 GA 68/95）。

关键实测结论（均为五轮均值）：

- lidar_pyramid P+Q：GA/Greedy 在 0.30--0.10 预算保持 mAP 约 0.7366--
  0.7368；0.05 分别降到 0.707613/0.699603。FP32 p50 约
  7.17/7.15 ms，0.05 P+Q p50 约 3.27/3.33 ms。
- F-Cooper P+Q：全部预算 mAP 为 0.5585--0.5594，接近 FP32
  0.5592；0.05 p50 约 2.58/2.51 ms，对应约 2.50x/2.55x。其 0.05
  Q-only mAP 约 0.5363，而 P-only 仍约 0.5588--0.5589，低预算损失
  主要来自量化。
- DiscoNet P+Q：全部预算 mAP 为 0.6347--0.6360，接近 FP32
  0.6356；0.05 p50 约 2.64/2.63 ms，对应约 2.58x/2.64x。其 0.05
  Q-only mAP 约 0.6211，而 P-only 约 0.6359，同样表明低预算精度损失
  主要来自量化而非结构化剪枝。
- 三模型中，P-only 通常提供中等加速且几乎不损失精度；更低 BOPS 下
  Q-only 提供主要额外加速，同时也是 AP cliff 的主要来源。P+Q 的交互效果
  依赖具体模型与所选 profile，不能由 P-only/Q-only 简单线性相加推断。

本轮正式代码交付包括：production build-only 边界、P-only strict-FP32
weighted/merge/fusion 合同、Q-only all-keep 派生、五轮 family 公平评估、
逐帧三段 latency、engine 锁定、CUDA cleanup、fail-closed resume、跨 GPU
whole-repeat sharding及三模型强校验汇总器。最终相关测试为 40 passed，
`git diff --check` 通过。大型 outputs、ONNX、plan、checkpoint、逐帧 CSV
均不提交 Git，只提交代码、测试和本 handoff。

---
时间戳：2026-07-21 18:47:07 CST｜轮次：Round 4
---
