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

## Round 2：四项正式搜索终态、F-Cooper 完整结果与 Disco 部署失败审计

### 终态总览

四个队列进程均已退出，queue status 中 return code 均为 0，但“进程结束”不等于“四项实验全部成功”。权威产物复核后的真实状态如下：

| 模型 | 方法 | Stage-1 六预算 | Stage-2 | 1789 帧 full validation | 最终状态 |
|---|---|---:|---:|---:|---|
| F-Cooper | Greedy | 6/6 达标 | 6/6 成功 | 6/6，0 skip | 完整成功 |
| F-Cooper | GA | 6/6 达标 | 30/30 Top-5 成功 | 6/6，0 skip | 完整成功 |
| DiscoNet | Greedy | 6/6 达标 | 0/6 成功 | 0/6 | 部署失败，无 mAP/延迟结论 |
| DiscoNet | GA | 6/6 达标 | 0/30 成功 | 未启动 | 部署失败，无 winner |

权威 run：

- F-Cooper Greedy：`outputs/h800_heal_lidar_fcooper_runtime_graph_joint_greedy_20260722_135143`
- Disco Greedy：`outputs/h800_heal_lidar_disco_runtime_graph_joint_greedy_20260722_141355`
- F-Cooper GA：`outputs/h800_heal_lidar_fcooper_runtime_graph_joint_ga_20260722_143937`
- Disco GA：`outputs/h800_heal_lidar_disco_runtime_graph_joint_ga_20260722_151851`

### 搜索空间证据

两种方法在同一模型上使用相同 runtime context：

| 模型 | 总原子单元 | 进入剪枝搜索 | 固定接口保护 | 剪枝域 | 量化组 | 量化保护组 | runtime merge relations |
|---|---:|---:|---:|---:|---:|---:|---:|
| F-Cooper | 4308 | 3840 | 468 | 21 | 28 | 0 | 1 |
| DiscoNet | 4477 | 4009 | 468 | 25 | 32 | 0 | 2 |

Disco 的 32/32 weighted modules 均在真实前向中被调用，额外 4 个 fusion weighted ops 和 169 个 fusion 原子单元已经进入量化/剪枝搜索。因此本轮 Disco 失败不能解释为 tracer 没有覆盖 fusion 计算图。

### F-Cooper Greedy 六预算完整结果

共同 full-validation manifest hash：`e5cbece0ceaf2ac1b2c47305a3fa3bdc5f616baa1ce86b0754ba565a013ac463`。Greedy strict-FP32 reference 为 mAP 0.559200、forward p50 6.1412 ms、1789/1789、0 skip。

| 目标 | 实际 BOPS | 参数剪枝 | INT8/FP16/FP32 | AP30/50/70 | mAP | p50 ms | 相对 FP32 加速 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.05 | 0.054861 | 27.329% | 13/12/3 | 0.684810/0.580940/0.403403 | 0.556384 | 2.0402 | 3.010x |
| 0.10 | 0.104662 | 25.872% | 5/18/5 | 0.686494/0.584317/0.405755 | 0.558855 | 2.2668 | 2.709x |
| 0.15 | 0.154414 | 21.485% | 1/19/8 | 0.686609/0.584327/0.405176 | 0.558704 | 2.4625 | 2.494x |
| 0.20 | 0.200742 | 20.265% | 0/17/11 | 0.687165/0.584401/0.406078 | 0.559215 | 3.3995 | 1.806x |
| 0.25 | 0.249982 | 19.422% | 0/12/16 | 0.687183/0.584377/0.406048 | 0.559203 | 3.6053 | 1.703x |
| 0.30 | 0.301351 | 20.054% | 0/15/13 | 0.687273/0.584492/0.406405 | 0.559390 | 4.1052 | 1.496x |

6/6 候选的 physical、Q/DQ、engine、precision、merge、evaluation acceptance 全部为 true；每个候选均为 1789/1789、0 skip。

### F-Cooper GA 六预算完整结果

GA strict-FP32 full-validation reference 为 mAP 0.558933、forward p50 6.1332 ms、1789/1789、0 skip。30/30 repaired Top-5 候选均完成 500 帧评估且 0 skip，六个唯一 round winner 复用已有 engine 完成 1789 帧复评，`candidate_engine_rebuild_count=0`。

| 目标 | 实际 BOPS | 参数剪枝 | INT8/FP16/FP32 | AP30/50/70 | mAP | p50 ms | 相对 FP32 加速 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.05 | 0.054868 | 26.646% | 13/11/4 | 0.683989/0.581377/0.405084 | 0.556817 | 2.0497 | 2.992x |
| 0.10 | 0.104052 | 23.781% | 5/19/4 | 0.686034/0.583674/0.405209 | 0.558306 | 2.2591 | 2.715x |
| 0.15 | 0.154925 | 22.227% | 0/20/8 | 0.686499/0.584510/0.405699 | 0.558902 | 3.0582 | 2.005x |
| 0.20 | 0.204901 | 20.476% | 0/16/12 | 0.687029/0.584481/0.406020 | 0.559177 | 3.4468 | 1.779x |
| 0.25 | 0.254038 | 18.556% | 0/12/16 | 0.687231/0.584762/0.406283 | 0.559425 | 3.5984 | 1.704x |
| 0.30 | 0.304043 | 19.547% | 0/15/13 | 0.687248/0.584536/0.406136 | 0.559307 | 4.1023 | 1.495x |

30/30 Top-5 的 physical、Q/DQ、engine、precision、merge、evaluation acceptance 全部为 true；六个 full-validation winner 的 precision、merge、evaluation acceptance 也全部通过。

### F-Cooper Greedy 与 GA 对比

| 目标 | GA−Greedy mAP | GA−Greedy p50 ms | GA−Greedy 参数剪枝率 |
|---:|---:|---:|---:|
| 0.05 | +0.000432 | +0.0095 | −0.683 pp |
| 0.10 | −0.000550 | −0.0076 | −2.092 pp |
| 0.15 | +0.000199 | +0.5957 | +0.742 pp |
| 0.20 | −0.000038 | +0.0473 | +0.211 pp |
| 0.25 | +0.000223 | −0.0069 | −0.866 pp |
| 0.30 | −0.000083 | −0.0029 | −0.507 pp |

结论：GA 在 F-Cooper 上没有形成稳定优势。六点 mAP 差均不超过 0.00055，除 0.15 预算外延迟几乎相同；0.15 处 GA 未选择 INT8，导致 p50 比 Greedy 慢约 0.596 ms。若考虑搜索成本，Greedy 更有性价比。单点最高 mAP 是 GA 0.25（0.559425，实际 BOPS 0.254038，p50 3.5984 ms）；最低延迟是 Greedy 0.05（2.0402 ms）。

### Disco Stage-1 六预算结果

Disco Greedy 已找到六个满足硬门槛的候选，但部署均失败：

| 目标 | 实际 BOPS | Stage-1 参数剪枝 | INT8/FP16/FP32 | 部署状态 |
|---:|---:|---:|---:|---|
| 0.05 | 0.054952 | 69.471% | 12/17/3 | engine build failed |
| 0.10 | 0.101834 | 58.987% | 8/15/9 | engine build failed |
| 0.15 | 0.152767 | 53.387% | 1/17/14 | engine build failed |
| 0.20 | 0.201546 | 51.250% | 1/14/17 | engine build failed |
| 0.25 | 0.247906 | 43.086% | 0/12/20 | engine build failed |
| 0.30 | 0.304124 | 47.931% | 1/11/20 | engine build failed |

Disco GA 各预算 rank-0 repaired Stage-1 候选如下；每轮 Top-5 全部失败，因此这些只是 proxy/物理计划指标，不能作为真实部署结果：

| 目标 | 实际 BOPS | Stage-1 参数剪枝 | INT8/FP16/FP32 | Top-5 成功数 |
|---:|---:|---:|---:|---:|
| 0.05 | 0.054999 | 70.889% | 12/16/4 | 0/5 |
| 0.10 | 0.104668 | 59.295% | 9/14/9 | 0/5 |
| 0.15 | 0.153409 | 51.523% | 2/17/13 | 0/5 |
| 0.20 | 0.204739 | 50.351% | 1/14/17 | 0/5 |
| 0.25 | 0.253515 | 41.253% | 0/12/20 | 0/5 |
| 0.30 | 0.304992 | 46.914% | 1/11/20 | 0/5 |

### Disco 失败根因

- Disco Greedy 的 6/6 engine build logs 和 Disco GA 实际进入 TensorRT 的 29/29 engine build logs 均为同一错误。
- adaptive ONNX merge resolver 将 shape/control 子图中的 `/Where_2` 误判为激活 merge。其输入为 `/Equal_1_output_0`、`/ConstantOfShape_1_output_0`、`/Reshape_4_output_0`，实际用于生成 `/Expand_1` 的 shape。
- `_insert_explicit_adaptive_merge_casts` 跳过了 Where 的 BOOL condition，但把另外两个 INT64 shape 分支 cast 为 FP32。TensorRT 因而报错：`ElementWiseOperation MIN/MAX must have same input types`，具体冲突为 Float 与 Int64，随后 `/Expand_1` 解析失败。
- 根因是 canonical ONNX pass 对 Add/Concat/Mul/Where/MatMul 做递归最近 weighted-producer 搜索时，没有限制为 runtime relation 匹配的激活 merge，也没有用 ONNX dtype/shape-subgraph 信息排除非浮点 shape merges。
- 另有 1 个 Disco GA 候选在 engine build 前被 wrapper parity 拒绝：cls/reg/dir mean abs 分别约 2.27e-4/9.09e-5/2.57e-4，高于 5e-5 门槛，并且局部 max 也有超过 0.005 的输出。
- 因此本轮没有任何 Disco 候选可报告真实 mAP、延迟或 Greedy-vs-GA 优劣；不得使用 Stage-1 proxy 数值替代部署结论。

### 搜索成本与资源占用

| 模型/方法 | Stage-1 s / GPU-h | proxy 请求 / 实算 | hit/miss | 唯一候选 | engine 尝试/成功 | 计入帧数 | Stage-2 GPU-h | 峰值 MiB | 记录总 GPU-h | wall min |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| F-Cooper Greedy | 46.55 / 0.03879 | 9397/9397 | 0/9397 | 9397 | 6/6 | 12523 | 0.35315 | 7296 | 0.39471 | 21.98 |
| Disco Greedy | 185.42 / 0.15452 | 26810/26810 | 0/26810 | 26810 | 6/0 | 0 | 0.37251 | 7292 | 0.52983 | 25.47 |
| F-Cooper GA | 140.10 / 0.11675 | 32345/30776 | 1569/30776 | 30776 | 30/30 | 28023 | 1.16734 | 7506 | 1.28683 | 39.03 |
| Disco GA | 359.79 / 0.29982 | 50149/48543 | 1606/48543 | 48543 | 30/0 | 500 | 1.83205 | 7976 | 2.13459 | 54.62 |

说明：F-Cooper GA 的 28023 帧 = 30×500 Top-5 + 6×1789 full winners + 500/1789 两次 reference。Disco GA 只有 500 帧 reference，候选均未进入真实评估。Disco Greedy 在候选 engine build 前失败，因此候选/reference 帧数均为 0。

Engine build 分阶段累计耗时：

| 模型/方法 | 总 pipeline s | calibration s | ONNX s | materialize s | Q/DQ s | TensorRT s |
|---|---:|---:|---:|---:|---:|---:|
| F-Cooper Greedy | 854.78 | 609.18 | 6.86 | 3.91 | 2.44 | 232.14 |
| Disco Greedy | 1340.90 | 1264.15 | 5.56 | 9.75 | 1.67 | 59.51 |
| F-Cooper GA | 3381.87 | 2039.22 | 70.25 | 42.98 | 17.63 | 1209.77 |
| Disco GA | 6592.38 | 6101.45 | 54.13 | 107.86 | 10.30 | 317.10 |

复杂度明细：

- F-Cooper Greedy：198 步，邻居总数 9396，每步 33–49。
- Disco Greedy：516 步，邻居总数 26809，每步 42–56。
- F-Cooper GA：六轮实际记录 39 代；每代 canonical phenotype 512–1024，累计 23040，canonical collision=0，generation-level cache hits=1455。
- Disco GA：六轮实际记录 40 代；每代 canonical phenotype 512–1024，累计 23552，canonical collision=0，generation-level cache hits=1421。

### 成本统计仍存在的口径缺口

F-Cooper GA 的 `search_cost_summary.json` 能识别 36 次候选评估和 25734 个候选帧，但 phase events 只有 30 次 `stage2.candidate_evaluation`，没有单独记录六个最终 1789 帧 full-validation winner 的 phase。因而表中的 GA `Stage-2 GPU-h` 与“记录总 GPU-h”只对已包裹 phase 精确，未完整计入最终 full validation 的 allocated GPU-hours；utilization-weighted GPU-hours 和 wall time仍覆盖整个进程。后续需要为 GA final full validation/reference 增加独立 resource phase，才能满足“每阶段精确 GPU-hours”的完整口径。

### 下一步必须完成

1. adaptive merge 只允许处理 runtime relation 匹配且数据类型为浮点激活的 ONNX merge；显式排除 shape/control graph，并增加 INT64 `Where → Expand` 回归测试。
2. 修复后先对 Disco 的一个现有候选做 physical → ONNX → Q/DQ → TensorRT smoke，确认 `/Where_2` 不再插入 FP cast。
3. 重新运行 Disco Greedy 六候选和 Disco GA 30 个 Top-5/六 winner；本轮失败结果不能直接续作最终结果。
4. orchestration 应在六预算任一 round 无成功 Stage-2 candidate、或缺少最终 full-validation 文件时返回非零；当前 `allow_no_success=True` 使 Disco 两项任务错误地以 return code 0 结束。
5. 补齐 GA final full-validation/reference 的 resource phase，再生成最终 Disco Greedy-vs-GA 和两模型总表。

---
时间戳：2026-07-23 15:29:54 CST｜轮次：Round 2
---

## Round 3：runtime merge dtype 闭环、逐代 Stage-2 GA 协议与真实 Disco TensorRT smoke

### runtime-based 量化 merge 修复

- `quantization/precision/merge_contract.py`
  - ONNX merge 候选先做 shape inference/dtype audit，只允许浮点 data inputs 与浮点 outputs。
  - runtime relation kind 必须和 ONNX op 类型相容，且 relation members 必须唯一匹配；解析不完整时 fail closed。
  - 不再把所有 Add/Concat/Mul/Where/MatMul 当作强制 FP16 merge；按 `INT8 < FP16 < FP32` 推导候选相关 merge 精度。分支精度相同时保持该精度，混合精度时只在 merge 输入边提升到最高需求精度。
- `quantization/precision/qdq_inserter.py`
  - 插 cast 前再次做 dtype 防御审计，非浮点目标直接拒绝。
  - 后置 merge audit 只审 runtime relation 已解析出的 adaptive targets，避免旧式全图 merge scan 把已经排除的 shape/control 算子再次纳入。
- `tests/test_runtime_precision_coupling.py`
  - 增加 INT64 `Where -> Expand`、runtime relation kind/op 匹配、后置审计不重引入非目标 merge 等回归。

Disco 实图解析结论：

- `/Concat`：三条 `backbone_m1.deblocks.{0,1,2}.0` 浮点激活分支的真实 concat；本次 smoke 分支为 FP32/FP16/INT8，因此 merge 推导为 FP32。
- `/Mul_9`：`fusion_net.pixel_weight_layer.conv1_4` 权重分支与 `shrinker_m1.layers.0.double_conv.2` 特征分支的真实逐元素乘法；本次 smoke 分支为 FP32/FP16，因此 merge 推导为 FP32。
- `/Where_2`、`/Where_4`：data inputs/output 均为 INT64 的 shape/control 子图，已显式排除，不属于可选量化精度。
- `/Concat_5`、`/Where_3` 虽为浮点，但无兼容 runtime relation，按通用规则排除，不新增 Disco 专属规则。

### GA Stage-2 协议纠正

- 历史复核确认此前 H800 Pyramid/F-Cooper/Disco 正式 GA 都是“完成全部 Stage-1 代后，每预算只统一选一次 Top-5”，并非用户要求的逐代真实评估。
- `search/orchestration/lidar_pyramid_search.py`、`search/stage2/lidar_pyramid_real_evaluator.py`、新增 `search/stage2/generation_results.py` 已实现通用逐代协议：
  - 每个实际 GA generation 独立 repair/rescore/dedup，并选最多 5 个候选。
  - 0 个：跳过 engine build，写明 BOPS band/repair/dedup 等原因。
  - 1 个：只部署构建，跳过 500 帧筛选，直接成为该代 winner。
  - 2--5 个：均构建真实 engine 并严格执行 500 帧、0 skip，按 Stage-2 F2 选该代唯一 winner。
  - 汇总各代唯一 winner，复用 engine，在 1789 帧完整验证集上评估，再按同一 F2 选每个 BOPS 预算唯一最终 winner。
  - 跨 generation 部署缓存、评估缓存与 full-validation 多 GPU 调度已接入。
- 三个 GA 正式配置均加入 `stage2_selection_scope: per_generation_topk`：
  - `search/configs/lidar_pyramid_h800_domain_width_joint_ga.yaml`
  - `search/configs/heal_lidar_fcooper_h800_domain_width_joint_ga.yaml`
  - `search/configs/heal_lidar_disco_h800_domain_width_joint_ga.yaml`
- Disco 正式配置已恢复为 GPU 5/6/7、minimum/maximum workers=3；GPU 7 单卡配置只用于 smoke，没有留入正式配置。

### 失败传播与资源计量

- `search/stage2/round_results.py` 与 orchestration 已移除 `allow_no_success=True` 语义；无成功 Stage-2 或缺少最终 full-validation 时写失败产物后抛错，正式进程不能再以假成功 return code 0 结束。
- 逐代 500 帧、deploy-only、逐代 winner full validation/reference 均新增独立 resource phase；继续汇总精确 allocated GPU-hours、utilization-weighted GPU-hours、frames、cache hit/miss、canonical phenotypes、engine build 分阶段耗时与峰值显存。

### 验证证据

- 相关测试：31 passed；runtime coupling 单文件：7 passed。
- 先前修复的第一次 smoke 因后置 audit 仍全扫 merge 而 fail closed，确认无错误 engine 产生。
- 修正后真实 Disco smoke：
  - run：`outputs/h800_heal_lidar_disco_runtime_graph_joint_ga_20260723_012738`
  - 候选：`4983198fcdfd6ef812b518739c753cd57c5fb32338d4fd68dac04a77412b4182`
  - ONNX/QDQ/TensorRT/precision realization/evaluation acceptance 全部通过。
  - 10 帧 candidate 与 10 帧 strict-FP32 reference 均完成；该小样本只作链路验证，不能作为正式精度结论。
  - engine pipeline 313.739 秒：activation calibration 266.317 秒、TensorRT build 44.476 秒、materialization 1.323 秒、ONNX export 1.121 秒、QDQ 0.372 秒、precision mapping 0.106 秒。
  - peak VRAM 与阶段 GPU-hours 见该 run 的 `search_cost_summary.json`。

### 当前目标函数口径

- 参数剪枝率：`1 - parameter_count_after / original_parameter_count`。它不是独立基因或 Stage-2 项；legal domain width 决定物理参数量。BOPS 硬带是准入条件；GA 在最优 Taylor 的 5% 近优集合内以较低参数保留率作 exploitation 次级排序，Greedy 在边际 Taylor/BOPS reduction 与 Taylor 后作次级排序。
- Stage-2：`F2 = 0.8 * max(0, mAP_ref - mAP_candidate) / 0.02 + 0.2 * latency_p50_candidate / latency_p50_ref`。两项均无量纲，不会因毫秒的原始数值大而自动支配；当前参数有意更偏向精度保持。

### 尚未完成

1. 使用新逐代 Stage-2 协议重新运行 F-Cooper 与 Disco 六个 BOPS 预算的 GA；旧的每预算 30 个总 engine 结果不能替代新协议结果。
2. Disco Greedy 也需用 runtime merge 修复重新运行，才能得到正式六预算真实 mAP/延迟。
3. 当前检查时 GPU 5、6 各有外部进程占用，GPU 7 空闲；正式配置要求三卡全部满足空闲门槛，因此未误启动。
4. 完成新实验后汇总每代候选/engine/500 帧 winner、1789 帧最终 winner、精度/时延/BOPS/参数剪枝率和全量资源成本。

---
时间戳：2026-07-23 16:35:17 CST｜轮次：Round 3
---

## Round 4：正式三项六预算重跑队列启动

用户确认 GPU 5/6/7 已完全空闲后完成启动前检查：三卡均为 4 MiB、0% utilization，无 compute process；没有残留 `search.cli` 或旧队列进程。三份正式配置均使用 GPU 5/6/7，六个 BOPS targets 为 0.05/0.10/0.15/0.20/0.25/0.30，Stage-2 为 500 帧，full validation 为 1789 帧；两个 GA 均为 `stage2_selection_scope: per_generation_topk`。

新增正式串行队列脚本：

- `scripts/run_heal_lidar_runtime_graph_formal_reruns.sh`
- 只运行必须重做的三项，不重复已有效的 F-Cooper Greedy：
  1. Disco Greedy runtime merge 修复六预算重跑；
  2. F-Cooper GA 逐代 Top-5 Stage-2 六预算重跑；
  3. Disco GA runtime merge + 逐代 Top-5 Stage-2 六预算重跑。
- 三项串行占用同一 GPU 5/6/7 池；单项失败会记录非零 return code，但队列继续运行其余项，最终 queue return code 汇总失败状态。

启动信息：

- queue PID：`1000283`
- 当前 search PID：`1000288`（Disco Greedy）
- run tag：`20260723_1643_runtime_merge_per_generation`
- status：`outputs/h800_heal_lidar_runtime_graph_formal_rerun_20260723_1643_runtime_merge_per_generation.status.jsonl`
- log：`outputs/h800_heal_lidar_runtime_graph_formal_rerun_20260723_1643_runtime_merge_per_generation.log`
- pid file：`outputs/h800_heal_lidar_runtime_graph_formal_rerun_20260723_1643_runtime_merge_per_generation.pid`
- 第一个正式 run：`outputs/h800_heal_lidar_disco_runtime_graph_joint_greedy_20260723_035226`
- 启动时搜索代码 commit：`4dc60e3047de22631036cb4a2dd2181274895098`

启动后 GPU 5/6/7 分别占用约 4079/1155/1155 MiB，三个 worker 已加载。恢复时先读 status JSONL，再检查 queue/search PID 与当前 run 的 `run_manifest.json`、`search_cost_summary.json`、round/final-validation 产物；不得在队列仍活跃时重复启动。

---
时间戳：2026-07-23 18:53:07 CST｜轮次：Round 4
---

## Round 5：Disco Greedy 正式完成，F-Cooper GA 逐代 Stage-2 接近完成

队列仍正常运行，PID `1000283`。权威 status：

- Disco Greedy 于 `2026-07-23T04:34:14-07:00` return code 0 完成。
- 随后自动启动 F-Cooper GA，当前 search PID `1657864`。
- Disco GA 尚未启动，继续等待同一三卡池。

### Disco Greedy 正式六预算结果

run：`outputs/h800_heal_lidar_disco_runtime_graph_joint_greedy_20260723_035226`

六个预算均进入 BOPS hard band；每个候选均完成 1789/1789 帧、0 skip，physical/QDQ/engine/precision/merge/evaluation acceptance 全部为 true。strict-FP32 reference 为 mAP 0.635584、forward p50 6.7545 ms。

| 目标 BOPS | 实际 BOPS | 参数剪枝率 | INT8 层 | mAP | p50 ms | F2 |
|---:|---:|---:|---:|---:|---:|---:|
| 0.05 | 0.054888 | 70.033% | 12 | 0.635154 | 2.3512 | 0.086818 |
| 0.10 | 0.100805 | 58.887% | 8 | 0.635840 | 2.6341 | 0.077996 |
| 0.15 | 0.153817 | 51.363% | 1 | 0.635495 | 2.9434 | 0.090709 |
| 0.20 | 0.201973 | 50.305% | 1 | 0.635583 | 3.3533 | 0.099332 |
| 0.25 | 0.254354 | 40.908% | 0 | 0.635643 | 4.0078 | 0.118669 |
| 0.30 | 0.303923 | 48.376% | 1 | 0.635746 | 4.0530 | 0.120010 |

搜索成本：Stage-1 207.65 秒/0.17304 allocated GPU-hours；26039 次 proxy 实算、26039 唯一候选、0 cache hit；Greedy 498 步、每步 42--56 个邻居、邻居总数 26038。Stage-2 6 个 engine、候选 10734 帧，加 reference 共 12523 帧；Stage-2 0.63469 allocated GPU-hours。总 allocated GPU-hours 0.81061，utilization-weighted GPU-hours 0.17133，wall 2494.75 秒，峰值显存 7450 MiB。六个 engine build pipeline 累计 1779.30 秒，其中 calibration 1447.74 秒、TensorRT build 309.27 秒。

### F-Cooper GA 当前进度快照

run：`outputs/h800_heal_lidar_fcooper_runtime_graph_joint_ga_20260723_043415`

- 六个 BOPS round 的 Stage-1 均已生成。
- 0.30/0.25/0.20/0.15/0.10 五个预算的逐代 Stage-2 已完整结束：实际代数分别为 6/7/7/6/6，共32代、160个代内候选槽位，全部 `status=ok`。
- 0.05 预算共有7个实际代；generation 0--3 已完成，generation 4 正在运行。
- 全局共39个实际代；已生成36个 generation winner，当前是第37代。
- 当前184/195个候选槽位已解析：118个唯一 Stage-2 score 全部 `ok`，66个跨代 cache hits；没有 build/evaluation failure。
- 39个 generation winner 全部产生后，才会启动各代 winner 的1789帧完整验证与每预算最终 winner 选择；当前 full-validation 文件数为0，不能提前报告 F-Cooper GA 最终结果。
- GPU 5/6/7 正常工作；快照约为 6.8/3.6/3.6 GiB，无错误日志。

### 剩余任务

1. 完成 F-Cooper GA 的 generation 4--6、各代 winner 1789帧复评和六预算最终 winner。
2. 队列自动启动 Disco GA，完成相同逐代 Stage-2 与最终完整验证协议。
3. 队列全部结束后汇总 Disco Greedy vs GA、F-Cooper Greedy vs 新 GA，以及精确资源成本。

---
时间戳：2026-07-23 22:26:21 CST｜轮次：Round 5
---
