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
