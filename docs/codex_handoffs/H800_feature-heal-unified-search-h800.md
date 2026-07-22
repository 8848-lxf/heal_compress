# H800 `feature/heal-unified-search-h800` 持续工作记录

本文件是分支 `feature/heal-unified-search-h800` 的唯一持续 handoff。后续在此分支完成的每轮工作都应追加到文件末尾，不覆盖历史记录；每轮结尾统一使用“时间戳 + 轮次”的分割线，以便直接读取本文件恢复任务。

## Round 1：runtime 全图剪枝/量化关系、adaptive merge 与搜索成本审计

### 本轮目标与最终口径

- 将 F-Cooper 和 DiscoNet 从 family 静态/人工审计搜索域切换到 `heal_runtime_graph_v1`：剪枝域、量化组和 merge 关系都来自一次真实前向运行时计算图。
- 不添加 F-Cooper/Disco 专属剪枝或量化保护规则，不再用 Pyramid 自身的 `_select_all_legal_domain_units` 裁剪这两个模型的搜索域。
- 保留通用的不可破坏接口契约：PFN scatter 固定接口、三个 deblock 输出接口以及检测 Head 输出接口。
- merge 不再一律强制 FP16。各分支位宽相同则保持该位宽；不同时按 `INT8 < FP16 < FP32` 提升到共同精度，且只在具体 merge 输入边上插入 cast，避免污染同一生产者的其他 fan-out 消费者。
- 使用同一套 Greedy/GA 六预算搜索，并记录每阶段 GPU-hours、proxy/cache、唯一候选、Greedy 邻居数、GA 每代 canonical phenotype、Stage-2 引擎/帧数、峰值显存和 engine build 分阶段耗时。

### 已完成的主要功能

1. Runtime-based 量化关系生成

   - 新增运行时 precision group/relation 数据结构和追踪器；同一物理模块的所有调用点共享一个精度基因。
   - 从真实 `torch.cat`、逐元素乘法等运行时 merge 自动生成关系，不用模型名称或手写层清单识别 F-Cooper/Disco。
   - merge 关系不强制各分支搜索基因相同，而是标记 `derived_per_candidate`，由候选实际分支精度推导 merge 精度。
   - 通用 tracer 补充 `F.softmax`/`F.sigmoid` 追踪，并修复重复 root alias 的 canonical index-map，避免动态图同一物理模块多次调用产生虚假独立单元。

2. 全图剪枝搜索域

   - `heal_runtime_graph_v1` 直接使用 runtime dependency scopes 构建 legal pruning domains，不再经过 Pyramid family 选择器或 F-Cooper/Disco 人工域规则。
   - F-Cooper：4308 个总原子单元，3840 个进入搜索，21 个剪枝域；28/28 个 weighted modules 均被运行时调用。
   - DiscoNet：4477 个总原子单元，4009 个进入搜索，25 个剪枝域；32/32 个 weighted modules 均被运行时调用。相对 F-Cooper 新增的 169 个 fusion 原子单元全部进入搜索。
   - 两模型均有 468 个原子单元因通用固定接口而不进入剪枝搜索，精确构成为：PFN 64 + 三个 deblock 输出 `3 × 128 = 384` + `cls/reg/dir` Head 输出 `2 + 14 + 4 = 20`。
   - 这 468 仅是剪枝保护；当前量化保护组为 0。

3. Adaptive merge 精度与显式 Q/DQ

   - F-Cooper 自动得到 28 个量化组、1 个 runtime merge relation（3 路 deblock concat）。
   - DiscoNet 自动得到 32 个量化组、2 个 runtime merge relations（3 路 deblock concat、shrink feature 与 pixel-weight 输出的逐元素乘法）。
   - ONNX canonical graph 根据候选精度解析每个 Add/Concat/Mul/Where/MatMul 的最近 weighted producers，并生成 merge contract 与 relation-match 审计。
   - 相同精度 merge 保持原精度；混合精度 merge 逐级提升。INT8 merge 保留 Q/DQ 边，FP16/FP32 merge 插入强类型 edge-local cast；后继 weighted op 继续拥有自己的输入 requantization。
   - Q/DQ 审计同时覆盖 merge 输入是否仍为量化张量、推导出的 merge 精度、显式 cast 和 promoted weighted output contract。

4. 通用部署验收

   - Runtime policy 的部署验收使用 weighted-op precision proof、engine structure proof 和 Q/DQ auxiliary contract，不再依赖 family 命名的 TensorRT layer 清单。
   - physical materialization、ONNX export、precision mapping、calibration、Q/DQ insertion、TensorRT build 分别计时并写出 `engine_build_phase_timings.json`。
   - round 聚合支持无成功候选时先写阶段结果，防止长任务因中间 round 汇总而丢失成本证据。

5. 搜索资源与复杂度记录

   - 新增 0.5 秒间隔的 GPU utilization、显存和功耗采样；精确 phase wall time 乘已分配 GPU 数计算 allocated GPU-hours，另报 utilization-weighted GPU-hours，二者不混用。
   - 生成 `resources/resource_events.jsonl`、`resources/gpu_samples.jsonl`、`resources/resource_summary.json` 和顶层 `search_cost_summary.json`。
   - 记录 Stage-1 preparation/search、Stage-2 reference/candidate/full-validation 的 elapsed time 和 GPU-hours；同时汇总 proxy 请求数、cache hit/miss、唯一 phenotype、GPU batch 数、Stage-2 engine 数与帧数、峰值显存。
   - Greedy 每步记录实际一跳邻居数；GA 每代记录 raw/canonical phenotype 数、canonical collision、cache hit、可行候选及 population/offspring 数。

6. 正式六预算队列

   - 四个搜索使用 GPU 5/6/7 串行排队，顺序为 F-Cooper Greedy → Disco Greedy → F-Cooper GA → Disco GA，避免四项任务争抢同一 GPU pool。
   - 六个目标 BOPS 保留率均为 0.05、0.10、0.15、0.20、0.25、0.30，硬约束继续为 `|R_BOPS - target| <= 0.005`。

### 对应代码文件

Runtime graph 与 precision coupling：

- `tracer/precision_coupling_tracer.py`：新增 runtime precision groups/relations、merge 输入来源传播和稳定 relation/group ID。
- `tracer/runtime_graph_builder.py`：从 runtime channel groups 构建全图 dependency scopes，canonicalize 重复 root alias/index-map，并合入 precision graph。
- `tracer/generic_tracer.py`：补充 functional softmax/sigmoid 运行时追踪。
- `tracer/__init__.py`：导出新的 precision coupling API。

Adaptive merge 与 Q/DQ：

- `quantization/precision/merge_contract.py`：新增 adaptive merge 推导、ONNX merge 解析和 runtime relation matching。
- `quantization/precision/activation_boundary.py`：增加 `stop_before_merge`，让 INT8 输出边界停在 adaptive merge 前。
- `quantization/precision/qdq_inserter.py`：插入 adaptive merge cast、promoted output cast，并扩展 merge/QDQ 审计。

搜索空间、部署与 orchestration：

- `search/integration/heal_lidar_baseline_context.py`：接入 runtime pruning/precision graph，输出 policy、保护原因、runtime group/relation 和完整上下文审计。
- `search/model_family/export/heal_lidar_baselines.py`：将 runtime precision policy/relation 传入导出和 canonical precision mapping。
- `search/model_family/heal_lidar_deployment.py`：实现 runtime adaptive merge Q/DQ 构建与通用验收。
- `search/stage2/heal_lidar_baseline_real_evaluator.py`：传递 runtime policy/relation，使用通用 engine acceptance，并记录 engine build 分阶段耗时。
- `search/stage2/round_results.py`：允许中间 round 在暂无成功 Stage-2 结果时安全落盘。
- `search/orchestration/heal_lidar_baseline_search.py`：创建资源监视器、阶段计时和 `search_cost_summary.json` 汇总。
- `search/orchestration/lidar_pyramid_search.py`：为共同 Greedy/GA 框架增加可选 phase recorder、GA generation stats 和 Stage-2 计时包装。
- `search/resource_monitor.py`：新增 GPU 采样、phase events、GPU-hours、峰值显存与功耗汇总。
- `search/ga/engine.py`：记录每代 canonical phenotype 与 cache/feasibility 统计。
- `search/greedy/engine.py`：在 greedy path 中持久化每步邻居数量。
- `search/stage1/proxy_evaluator.py`：按 cache key 统计全程唯一 phenotype。

正式配置与启动脚本：

- `search/configs/heal_lidar_fcooper_h800_domain_width_joint_greedy.yaml`
- `search/configs/heal_lidar_fcooper_h800_domain_width_joint_ga.yaml`
- `search/configs/heal_lidar_disco_h800_domain_width_joint_greedy.yaml`
- `search/configs/heal_lidar_disco_h800_domain_width_joint_ga.yaml`
- `scripts/run_heal_lidar_runtime_graph_searches.sh`

测试：

- `tests/test_runtime_precision_coupling.py`
- `tests/test_search_model_family_heal_lidar_deployment.py`
- `tests/test_search_heal_lidar_baseline_orchestration.py`
- `tests/test_search_stage2_round_results.py`
- `tests/test_formal_packages_cpu.py`

### 验证证据

- 定向回归：132 passed。
  - runtime coupling、部署、orchestration、round results、formal package：111 passed。
  - Greedy、GA、GPU-batched proxy：21 passed。
- `git diff --check`：通过。
- F-Cooper 与 DiscoNet 均已完成真实模型 runtime context trace，weighted modules 无漏调，量化保护组均为 0。
- 早期 production smoke 已覆盖真实 physical pruning → ONNX → explicit Q/DQ → strongly typed TensorRT → precision/merge audit → 实帧评估。

### 当前正式搜索进度

队列进程：`scripts/run_heal_lidar_runtime_graph_searches.sh`，状态文件为 `outputs/h800_heal_lidar_runtime_graph_search_queue.status.jsonl`。

1. F-Cooper Greedy：已完成

   - run：`outputs/h800_heal_lidar_fcooper_runtime_graph_joint_greedy_20260722_135143`
   - 六个预算全部达到硬门槛，6/6 候选均完成 1789 帧、0 skip，physical/QDQ/engine/precision/merge/evaluation acceptance 全部通过。
   - Stage-1：46.5524 秒，0.0387937 allocated GPU-hours；Greedy 搜索本体 35.6082 秒。
   - proxy：9397 次请求，cache hit/miss = 0/9397，9397 个唯一候选，595 个 GPU batches。
   - Greedy：198 步；邻居数由前 155 步的 49 个逐步下降到最后一步的 33 个，完整逐步记录见 `search_cost_summary.json`。
   - Stage-2：6 个候选 engine，候选 10734 帧；加 reference 1789 帧后共 12523 帧。Stage-2 allocated GPU-hours 为 0.3531523。
   - 总 allocated GPU-hours 0.3947052；utilization-weighted GPU-hours 0.1107797；峰值显存 7296 MiB。
   - engine build 累计 854.778 秒，其中 calibration 609.185 秒、TensorRT build 232.142 秒、ONNX export 6.861 秒、physical materialization 3.908 秒、Q/DQ insertion 2.436 秒。

| 目标 BOPS | 实际 `R_BOPS` | 参数剪枝率 | INT8 层 | mAP | forward p50 (ms) |
|---:|---:|---:|---:|---:|---:|
| 0.05 | 0.054861 | 27.329% | 13 | 0.556384 | 2.0402 |
| 0.10 | 0.104662 | 25.872% | 5 | 0.558855 | 2.2668 |
| 0.15 | 0.154414 | 21.485% | 1 | 0.558704 | 2.4625 |
| 0.20 | 0.200742 | 20.265% | 0 | 0.559215 | 3.3995 |
| 0.25 | 0.249982 | 19.422% | 0 | 0.559203 | 3.6053 |
| 0.30 | 0.301351 | 20.054% | 0 | 0.559390 | 4.1052 |

2. Disco Greedy：运行中

   - run：`outputs/h800_heal_lidar_disco_runtime_graph_joint_greedy_20260722_141355`
   - context 与 Stage-1 文件已生成，当前由队列继续执行；不能在结果文件完整写出前标记为完成。

3. F-Cooper GA：等待 Disco Greedy 完成后自动启动。
4. Disco GA：等待 F-Cooper GA 完成后自动启动。

### 尚未完成/下轮恢复入口

- 等待 Disco Greedy、F-Cooper GA、Disco GA 三项正式六预算搜索结束，不要重复启动同配置任务。
- 每项结束后检查 queue status return code、六预算 hard band、所有 Stage-2 acceptance、1789/1789 和 0 skip。
- 汇总两模型 Greedy vs GA 的精度、实际 BOPS、参数剪枝率、INT8/FP16/FP32、延迟及完整搜索成本。
- GA 完成后重点读取每个 round 的 `ga_generation_resource_stats.json` 以及顶层 `search_cost_summary.json`，报告每代 canonical phenotype 数和 cache reuse。
- 当前 `outputs/` 运行产物不进入 Git；以 run 路径和 JSON 证据恢复。
- 工作区中 `scripts/aggregate_cnn_ga_greedy_results.py`、`tests/test_aggregate_cnn_ga_greedy_results.py`、`tools/recover_historical_search_costs.py` 属于并行存在但不纳入本轮提交的用户文件，后续提交仍需避免误 stage。

---
时间戳：2026-07-23 05:29:42 CST｜轮次：Round 1
---
