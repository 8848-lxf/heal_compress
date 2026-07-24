# H800 `feature/h800-transformer-unified-search` 持续工作记录

本文件只记录隔离分支 `feature/h800-transformer-unified-search` 的开发进展。后续每轮在文件末尾追加，不覆盖既有记录；每轮均以 CST 时间戳分割线结束。

## Round 1：安全审计、隔离 worktree 与 provenance 基线

### 隔离结果

- 主仓库验证路径：`/home/lixingfeng/UniAD_examine/heal_compress`。
- 基线：`origin/feature/heal-unified-search-h800`，完整 commit 为 `896a049830874ef0d40faa87927873a1e88bedb6`。
- 新分支：`feature/h800-transformer-unified-search`。
- 新 worktree：`/home/lixingfeng/UniAD_examine/heal_compress_h800_transformer_unified_search`。
- 没有使用建议的嵌套 `.worktrees/` 路径，因为主仓库没有忽略该目录；使用同级目录可避免正在运行正式搜索的主工作树出现新的未跟踪目录。
- 新分支已推送到 `origin`，首次验证 `HEAD...origin/feature/h800-transformer-unified-search = 0 0`。
- 正式 unified-search 主工作树仍为 `896a0498...`，原有三个未跟踪用户文件保持原样，没有被读取后写回、stage 或删除。

### 独立运行目录

本任务唯一 run root：

`/data/lxf/heal_data/outputs/h800_transformer_unified_search_framework_20260723_111336`

已创建独立的 `provenance/`、`inventory/`、`domain_manifests/`、`rankings/`、`proxy/`、`activation_proxy/`、`structures/`、`onnx/`、`engines/`、`calibration/`、`evaluation/`、`latency/`、`greedy/`、`ga/`、`stage2/`、`scheduler/h800_transformer_unified_search/`、`reports/` 和 `failures/`。没有复用 DiscoNet、F-Cooper 或历史 Transformer 输出目录与 lock。

### 进程与 GPU 保护

- 只读扫描了 `python`、`trtexec`、`search` 相关进程，并读取可访问的 `/proc/<pid>/cwd`、`cmdline`、`CUDA_VISIBLE_DEVICES` 与打开的输出/日志路径。
- `provenance/active_search_processes_before.json` 已写入独立 run root；捕获 47 个匹配进程，其中 35 个属于 DiscoNet/F-Cooper。所有条目均明确记录 `task_touched_process=false` 和 `task_touched_declared_or_open_paths=false`。
- 正式 HEAL Disco GA 仍从主工作树运行并使用 GPU 5/6/7；另一组 F-Cooper 作业使用 GPU 1/2/3/4。它们的源码、日志、engine、ONNX、calibration、checkpoint、manifest 与 lock 均未触碰。
- GPU 0 上只有 `user01` 的 PID `1347748`，首次审计占用约 11.1 GiB。用户已明确允许本任务使用 GPU 0；后续仍只设置 `CUDA_VISIBLE_DEVICES=0`，不向现有进程发送信号，并在每个 GPU 阶段前检查负载和显存。

### Git 与运行环境 provenance

独立 run root 已生成：

- `provenance/git_start_manifest.json`
  - 记录 repo/worktree、branch、base、正式搜索分支 commit、Transformer 对齐实验分支 commit、remote、ahead/behind、worktree 列表和 dirty status。
- `provenance/runtime_environment.json`
  - Python：`/home/lixingfeng/miniconda3/envs/modelopt/bin/python3.10`。
  - NVCC：`/home/lixingfeng/miniconda3/envs/modelopt/bin/nvcc`。
  - GCC/G++：均解析到 `modelopt` 前缀内的 Conda 编译器。
  - TensorRT Python：10.9.0.34；TensorRT root：`/home/lixingfeng/UniAD_examine/TensorRT-10.9_x86_cu118`。
  - PyTorch：2.4.0+cu118，CUDA build 11.8。
  - ModelOpt：0.29.0。
  - GPU 0 UUID：`GPU-2133c0e5-4e29-a3aa-d7c6-c45046e073f7`。
  - fail-closed 检查确认没有落到 `/usr/bin/nvcc` 或 modelopt 之外的 Python/GCC/G++。

### 本轮新增代码

- `tools/audit_active_search_processes.py`
  - 以只读方式扫描 `/proc` 和 `nvidia-smi`，记录 PID、命令、cwd、模型、声明/打开的输出路径、GPU 映射、权限受限字段与“本任务是否触碰”标志。
  - 同一工具将在任务结束时生成 `active_search_processes_after.json`，支持前后保护核对。
- `tools/capture_h800_transformer_provenance.py`
  - 生成 Git 起始血缘清单。
  - 通过项目现有 `modelopt_python_command` 和 `modelopt_subprocess_env` 进入隔离工具链；工具路径越出 `modelopt` 时直接失败。

### 当前边界

- 尚未修改剪枝、量化或搜索框架代码。
- 尚未启动 full1789、正式六预算搜索或任何新的长任务。
- 下一轮从现有框架调用关系与四个目标模型的真实配置/checkpoint/模块图审计开始，并优先复用已有统一搜索和历史 Transformer 分支中经验证的组件。

---
时间戳：2026-07-24 02:19:32 CST｜轮次：Round 1
---

## Round 2：真实结构审计、统一域/代理/搜索与四模型 Stage-1 smoke

### 四模型真实图审计

- 使用真实 YAML、严格 checkpoint load、真实验证样本前向与 hook，而非按模型名猜测结构；结果写入独立 run root 的 `inventory/`。
- V2X-ViT：12 个独立 Attention instance / `attention_dh` 域，包含 3 个 `v2xvit_agent_relation`、各 3 个 `v2xvit_spatial_window_w4/w8/w16`；3 个 FFN / `ffn_hidden` 域。
- CoBEVT：6 个独立 Attention instance，3 个 `cobevt_grid`、3 个 `cobevt_window`；6 个 FFN。family 仅作 adapter/cost/report 元数据，没有合并跨 block 宽度基因。
- AttFusion：真实为 1 个无 Q/K/V/O 投影参数的 `ScaledDotProductAttention`；没有伪造 `attention_dh`/FFN 域。
- CoAlign：真实为 3 个 projection-free `ScaledDotProductAttention`；没有伪造 `attention_dh`/FFN 域。
- 四模型均未检测到 Parameter object 或 storage 共享；每模型的 module inventory、shape trace、attention/FFN instance、共享参数及 unsupported pattern JSON 均已生成。

### 统一域与物理剪枝代码

- `search/pruning_space/local_domains.py`
  - 将原 CNN local domain 扩展为统一 `cnn_channel`、`grouped_conv_channel`、`attention_dh`、`ffn_hidden` schema；搜索器只读合法 scalar width。
- `search/pruning_space/transformer_domains.py`
  - 从真实 fused/separate QKV、HGT、FFN/gated FFN 结构生成每 instance 独立域。
  - Attention 合法宽度使用 deployment-friendly ladder 并始终包含原宽度；FFN power-of-two ladder 不截断到 256。
  - 固定每 head QK/VO nested ranking；允许不同 head 与 QK/VO 子耦合使用不同位置集合。
  - 共享 Parameter/storage 默认 fail closed，不因 residual 或同 family 合并 `d_h`。
- `search/pruning_space/transformer_physical_pruner.py`
  - 对 fused/separate QKV、bias、W_O 输入列、HGT relation tensor、标准/门控 FFN 做真实 Parameter shape 重写；更新 `head_dim`/`inner_dim`/scale，拒绝 requested/realized 不一致、mask-only 与隐藏 padding。
- `search/pruning_space/unified_physical_pruner.py`
  - 新增 CNN formal plan-first materializer 与 Transformer pruner 的单一 Stage-2 decoder；先执行 CNN dependency closure，再在同一物理模型执行 d_h/d_ff 重写并统一审计宽度、参数量、shape/structure hash。
  - 真实 V2X-ViT 联合探针已验证 CNN 64→60、Attention d_h 16→8、FFN 256→128；CNN validation 通过、前向 finite，参数 13,453,197→13,249,029。
- 真实独立物理 smoke：V2X-ViT d_h 32→16 + d_ff 256→128，CoBEVT d_h 32→16 + d_ff 256→128，均实现真实参数下降和有限前向。

### 权重—激活联合 Taylor、精度与成本模型

- `search/proxy/joint_weight_activation_taylor.py`
  - 采集同一任务损失的输出 `y/g/h≈g²`，候选重放完整结构+权重+激活扰动，按 `sum|gΔy| + 0.5 sum(g²Δy²)` 计算原始公共损失单位代理；不做逐层任意归一化。
  - 支持 module output 以及 functional Softmax/einsum/matmul/bmm 边界；Softmax A8 与 AV 激活扰动不会按“无权重”错误记零；quantizer ID 去重并 fail closed 检查冲突。
  - 修复 active graph 过滤：只要实例投影真实执行，必须保留该实例 Softmax 和 FFN activation 边界。
- `search/quantization_space/transformer_precision.py`
  - 只开放 W32A32/W16A16/W8A8；QK operands/accumulation/output 固定 FP32；LayerNorm 固定 FP32；residual Add 初版禁止未经引擎审计的 INT8。
  - Softmax A8 精确表示为 floating Softmax 后 INT8 Q/DQ output，不冒充 native INT8 exponential/reduction。
- `search/quantization_space/smoothquant.py`
  - 实现 alpha `{0.6,0.7,0.75,0.8}` 离线选择、结构/校准 manifest 兼容校验以及 activation/weight scale hash；不将 alpha 放入 GA 基因。
- `search/proxy/transformer_bops.py`、`transformer_latency.py`
  - 分解 projection/QK/AV/O/FFN BOPS；QK 固定 32×32，projection-free op 不虚构 weight bits；缺失 latency 映射返回 `missing_unit_mapping`，不静默记零。

### Greedy、GA 与真实 Stage-1 smoke

- Greedy 统一枚举 CNN/grouped/Attention/FFN width 与相邻精度动作，主分数为 `Delta joint Taylor / Delta BOPS`；近 Taylor tie 按 latency、参数、mixed weight size、多样性、domain ID；保留最近合法 budget capture。
- GA 使用统一 width/precision genotype、domain-aware mutation/crossover/repair、QK FP32 保护、稳定 candidate hash；constraint-first 排序改为 joint Taylor 主项。
- Stage-2 Top-K 预选记录 Taylor/latency/parameter rank、BOPS deviation、selection reason、width/precision config。修复 bounded repair pool：先按 BOPS hard gate，再按 Taylor，避免不合预算低损失候选挤掉全部可行候选。
- `scripts/smoke_transformer_unified_search.py` 在 GPU0、真实严格 checkpoint、真实至少 2-agent validation frame 上完成有界 pop=8/gen=2 smoke：
  - AttFusion：projection-free Softmax/AV + CNN，12 个 GA 可行记录，Top-2。
  - CoAlign：projection-free Softmax/AV + CNN，12 个 GA 可行记录，Top-2。
  - V2X-ViT：CNN + attention_dh + ffn_hidden + precision，4 个 GA 可行记录，Top-2，8 个真实输出 Taylor 边界。
  - CoBEVT：CNN + grid attention_dh + grid ffn_hidden + precision，4 个 GA 可行记录，Top-2，8 个真实输出 Taylor 边界。
- 这些是框架有界 smoke，预算目标由最小正 BOPS 动作构造（约 0.989–0.993），不是正式 0.30/0.20 大空间预算，也不是正式六预算结论。

### 隔离与当前边界

- 修复 `search/stage2/trt_modelopt.py`：modelopt worker 的 `PYTHONPATH` 不再硬编码正式工作树，改为当前新 worktree 的实际 repo root；相关隔离回归已增加。
- 已通过多轮 Transformer/domain/proxy/precision/BOPS/Greedy/GA/tracer/adapter 测试、`py_compile` 和 `git diff --check`；完整回归统计将在最终阶段统一生成。
- GPU0 任务始终使用 `CUDA_VISIBLE_DEVICES=0`；未向 user01 或 DiscoNet/F-Cooper 进程发送信号，未触碰其源码、输出或 lock。
- 尚未完成新的物理候选 ONNX/TensorRT Stage-2 engine；下一轮从 unified physical candidate 的 fixed-K wrapper parity、ONNX checker/shape inference、strongly typed engine 和 requested/realized precision 审计继续。
- 未执行 full1789；未启动正式完整六预算搜索。

---
时间戳：2026-07-24 04:08:52 CST｜轮次：Round 2
---

## Round 3：物理候选 ONNX/TensorRT 闭环、W8A8 实现与完整回归

### Stage-2 导出与精度审计代码

- `search/stage2/transformer_precision_export.py`
  - 将精确 module precision profile 展开到每个 canonical ONNX call，缺失/多余 module 或非法 precision 均 fail closed。
  - 通过真实图拓扑从所选 Q/K/V projection 追踪最近 QK、Softmax 与 AV，而不是依赖节点名称猜测；检查 QK operands/output 及 Softmax input/output 的 ONNX dtype。
  - 将 canonical ONNX identity 与 TensorRT inspector layer/tactic 对齐，要求 QK 与 Softmax compute 的 realized precision 均为 FP32。
- `scripts/smoke_transformer_stage2_engine.py`
  - 组合一个 CNN 64→60、一个 Attention d_h、一个 FFN d_ff 的真实物理候选；执行 strict checkpoint load、物理 state reload、有限前向、fixed-K wrapper parity、ONNX checker、shape inference、显式 Q/DQ/Cast、strongly typed TensorRT 10.9 build 和 inspector 审计。
  - W8A8 选择通过真实校准得到的 activation scale；calibration manifest 同时绑定 config/checkpoint/dataset/frames/physical structure hash，结构不兼容时拒绝复用。
- `quantization/precision/qdq_inserter.py`
  - 将 merge-boundary 最近加权算子审计从每分支递归遍历改为 ONNX DAG 的正向/反向动态规划；保留相同最近边界语义，并将 CoBEVT 实图的病理级审计从超过 20 分钟降至约 0.75 秒。
  - 新增 26 层 reconvergent diamond 回归，防止分支数量导致指数遍历。
- `quantization/tensorrt/layer_info.py`
  - TensorRT 将 Q/DQ Linear、Cast、bias/activation 融合为 `LayerType=fusion` 时，仅当 tactic 明确为 GEMM/MatMul/Conv 才承认其为 realized weighted compute，避免把任意 fusion 错当量化实现。
- `search/stage2/trt_modelopt.py`、`trt_build_worker.py`、`search/model_family/evaluation.py`、`evaluation_worker.py`
  - 在各自独立输出目录建立 canonical `heal_compress` package symlink，并把它放在 worker `PYTHONPATH` 首位。
  - worker 记录实际 loaded source module 路径；任何模块落到正式 unified-search 工作树时直接失败，解决同级 worktree 目录名不叫 `heal_compress` 时的源码串用风险。
- `opencood/tools/compression/latency_lut/tensorRT_benchmark.py`
  - 非 dry-run 在导出前先解析 plugin/trtexec，使 runtime 缺失与 ONNX/QDQ 导出失败可区分；不再因不可消费的任务先写中间产物。

### TensorRT 工具链与真实引擎验收

- 所有 build 使用 `modelopt` 环境、该环境内 NVCC/GCC/G++、TensorRT 10.9.0.34 和 GPU0；没有使用系统 NVCC。
- 为 H800 独立构建 sm90 PointPillar scatter plugin：
  - 路径：`provenance/pointpillar_scatter_trt_sm90_modelopt_20260723T132000/libpointpillar_scatter_trt.so`。
  - SHA256：`7be6450d65174dac28483aa631f2d8a4db416b500964f5c683fe1cabcc829c3c`。
- 接受的 W16A16 physical mixed engines：
  - V2X-ViT：fixed-K 27904，engine SHA256 `6535c8b9773500fd25d39315776866ae8673b7bc88fe9bf4b424f006ffef72b3`，85,338,236 bytes。
  - CoBEVT：fixed-K 29184，engine SHA256 `e64bda5d845c687456bf479855660f6c7e5da04d87ab7e717d1e2bd6dd0eb09b`，76,843,508 bytes。
  - 二者均准确实现 CNN 64→60、Attention d_h（V2X 16→8 / CoBEVT 32→16）和 FFN 256→128；无 mask-only、无隐藏 padding；3 个请求 FP16 weighted calls 均 realized FP16；QK 与 Softmax compute 均由 ONNX 和 inspector 双重证明为 FP32。
- 接受的 W8A8 engines：
  - V2X-ViT FFN1：engine SHA256 `29dd192d0b545415b3180063c003e54166894bd0f87816a0f2d1ff048fa721bb`；calibration hash `f870f937d19545adb6f69d976354884f3ded85b83f885fe447aa5bc6e4290863`。
  - CoBEVT FFN2：engine SHA256 `c7635ac1e79264103457fcbd5c604ed1a731f448780c9756267346ed2427dd83`；calibration hash `0ab49025094bd03dfba5dc9504bae9bae22188b7bda5fc324cd5fcccc98d382d`。
  - 二者 requested/realized INT8 weighted calls 均为 1/1，unresolved=0、mismatch=0；QK/Softmax compute 仍为 FP32。
- 两个早期 V2X W8 输出目录保留了可复现的 package-source 泄漏失败证据；修复后新目录通过。CoBEVT 早期递归 QDQ 审计的 task-owned 进程终止与恢复信息保留在 `failures/`；没有向任何外部搜索进程发送信号。

### 真实评估 smoke

- V2X-ViT W16A16：smoke10 为 10/10、0 skipped、mAP 0.786634、forward p50 18.7966 ms；fixed50 为 50/50、mAP 0.572844、p50 17.2977 ms。
- CoBEVT W16A16：smoke10 为 10/10、0 skipped、mAP 0.729694、p50 7.3627 ms；fixed50 为 50/50、mAP 0.572872、p50 7.8027 ms。
- V2X-ViT W8A8 FFN1：smoke10 为 10/10、0 skipped、mAP 0.779515、p50 16.7148 ms。
- CoBEVT W8A8 FFN2：smoke10 为 10/10、0 skipped、mAP 0.742745、p50 7.8793 ms。
- 这些时延来自 GPU0 共享条件下的评估 smoke，只用于引擎可运行性观察；没有冒充 200 warmup/500 timed/5 repeats 的无争用正式 latency 结论。

### SmoothQuant、Softmax A8 与剩余边界

- SmoothQuant 代码支持 `{0.6,0.7,0.75,0.8}` 离线网格、alpha 冻结和 scale/calibration/structure hash 审计；只读历史设备证据分别为 V2X-ViT alpha 0.75、CoBEVT alpha 0.8。
- 当前接受的 W8A8 引擎分别选择 FFN1/FFN2，因此本轮没有把 SmoothQuant 错标为已应用；下一步仍需为物理宽度后的 QKV projection 刷新 scales 并构建 QKV W8A8 engine anchor。
- Softmax A8 合同已实现为 floating Softmax 后 output Q/DQ，不标成 native INT8 exponential/reduction；当前四个 engine 候选的 Softmax 均 realized FP32，因此没有声称已完成 Softmax A8 engine 选择。

### 回归、隔离与停止条件

- 使用 output-local package alias 预导入后执行完整测试：`966 passed, 0 failed, 82 warnings in 49.50s`。
- 与本轮 Stage-2/隔离修改直接相关的复核：`38 passed, 4 warnings`。
- `python -m compileall` 与 `git diff --check` 均通过。
- 主工作树 commit 仍为 `896a049830874ef0d40faa87927873a1e88bedb6`，起始的三个用户未跟踪文件保持一致。
- 运行后只读审计写入 `provenance/active_search_processes_after.json`：3 个长期 supervisor/watcher PID 持续存在；worker PID 变化为外部调度生命周期。本任务发送外部信号 0、写入外部路径 0、引用本任务路径的外部搜索进程 0。
- 未执行 full1789；未运行正式六预算；未启动任何新的长搜索。

---
时间戳：2026-07-24 06:04:30 CST｜轮次：Round 3
---

## Round 4：最终清单、远端同步与后续接续点

### 最终报告产物

独立 run root 已由 `scripts/finalize_transformer_unified_search_reports.py` 从真实 JSON 产物汇总生成：

- `reports/framework_architecture.md`
- `reports/model_support_matrix.csv`
- `reports/domain_inventory.csv`
- `reports/attention_domain_manifest.csv`
- `reports/ffn_domain_manifest.csv`
- `reports/precision_unit_manifest.csv`
- `reports/activation_proxy_manifest.csv`
- `reports/smoothquant_manifest.csv`
- `reports/greedy_smoke_results.csv`
- `reports/ga_smoke_results.csv`
- `reports/stage2_smoke_results.csv`
- `reports/engine_precision_audit.csv`
- `reports/unsupported_patterns.csv`
- `reports/requested_vs_realized.json`
- `reports/regression_test_summary.json`
- `reports/final_acceptance.json`
- `root_conclusion.md`

汇总统计为：4 个模型支持行、82 个 CNN/grouped-conv 域、18 个 Attention d_h 域、9 个 FFN d_ff 域；150 个已发现 Transformer/functional precision units，其中 24 个进入有界搜索 smoke；22 个 deployment units 具有联合激活代理清单。4 个接受的 engine candidate 均无 requested/realized conflict。

### 最终代码与验证状态

- Stage-2 闭环实现提交：`796e2abd`（`feat: add transformer physical stage2 engine acceptance`）。
- 审计/报告与 Round 3 文档提交：`dbd50596`（`docs: report H800 transformer search framework acceptance`）。
- `dbd50596` 推送后曾验证本地/远端为 `0 0`；本轮文档提交后需再次执行 push/fetch/`rev-list`，并以 `reports/final_acceptance.json` 中的 final commit 为准。
- 最终重新执行 `python -m compileall -q opencood pruning quantization search scripts tests tools` 通过；`git diff --check` 通过。
- 完整隔离 pytest 仍为 966 passed / 0 failed；Stage-2 直接相关复核为 38 passed / 0 failed。
- 正式 worktree 当前仍是 `896a049830874ef0d40faa87927873a1e88bedb6`；dirty status 与任务开始一致，三个用户未跟踪文件未变。
- 最新 `active_search_processes_after.json` 捕获 31 个 DiscoNet/F-Cooper 相关进程，起始的 supervisor/watcher PID `591652`、`990778`、`4153684` 仍持续存在；外部 task-path reference、task signal、task path write 均为 0。

### 接续开发建议

1. 在物理 d_h 宽度固定后，为 V2X-ViT/CoBEVT 的 Q/K 或 fused QKV 重新采集 SmoothQuant activation statistics，离线确认 alpha 与 scale hash，并各构建一个 QKV W8A8 engine anchor。
2. 等待 GPU0 无争用窗口，再执行 baseline→candidate→baseline、200 warmup/500 timed/5 repeats 的正式 full-engine latency；不要使用本轮共享 GPU smoke p50 代替正式数据。
3. 若后续获得明确授权，再把已验证框架配置扩展到 0.30/0.25/0.20/0.15/0.10/0.05 六预算；本轮没有启动这些正式搜索。
4. AttFusion/CoAlign 继续按 projection-free functional attention 处理；除非真实实现新增可验证 Q/K/V/O projection，否则不得人为创建 attention_dh 域。

---
时间戳：2026-07-24 06:06:16 CST｜轮次：Round 4
---

## Round 5：Repair 审计与 V2X-ViT 0.05 Greedy 验收

### 隔离与审计

- 本轮继续使用 `feature/h800-transformer-unified-search` 与独立 worktree `/home/lixingfeng/UniAD_examine/heal_compress_h800_transformer_unified_search`，起始 commit 为 `be640fbbea06403dd33f80aa2b171c554a58f824`。
- 新 run root：`/data/lxf/heal_data/outputs/h800_v2xvit_repair_audit_greedy005_20260724_141650/`。没有复用历史 run root 的可写目录。
- `provenance/active_processes_before.json` 与 `active_processes_after.json` 已生成；本任务发送外部信号 0、触碰外部搜索路径 0。after 快照没有匹配的 DiscoNet/F-Cooper 行，这只记录快照差异，不归因于本任务终止进程。
- GPU 阶段仅使用当时可用的 GPU5；没有抢占 GPU0 上的用户进程，也没有执行正式无争用 latency 协议。

### Phase A：实际 repair 审计

- 历史 smoke artifact 共 72 个候选，全部缺失 raw genotype；没有猜测，使用 V2X-ViT 与 CoBEVT population=8、generation=1 的 deterministic replay 补齐审计轨迹。
- legacy replay 的 18 个 transition 中，16 个候选发生过精度字段写回，18 个 precision repair fields（其中 12 个 QK 写回）；结构宽度、Attention d_h、FFN d_ff、CNN、dependency、budget projection 均为 0。Greedy legacy 的 2 个修改同样仅为 precision 写回。
- 当前严格编码 replay 的 actual repair fields 为 0。canonicalization、hard-gate rejection、deduplication 被明确归类为非 repair；repair 后 phenotype 会重新计算 Taylor/BOPS/params/mixed-size/latency/hash。
- `repair_audit/legal_by_construction_audit.json` 门槛：`STRUCTURE_LEGAL_BY_CONSTRUCTION=true`、`GREEDY_REPAIR_FREE=true`、`GA_LEGAL_BY_CONSTRUCTION=true`、`PHASE_A_ACCEPTED=true`。
- 编码已改为合法状态 index：Attention/FFN gene 只能访问 legal widths，mutation 为同 locus 合法邻接状态，crossover 按固定 locus，QK/LayerNorm protected precision 不进入 genotype，超预算 offspring hard-gate reject；Greedy 不执行 structural/precision/budget repair。

### Phase B：V2X-ViT `R_BOPS=0.05` Greedy

- 模型 `lidar_v2xvit` 的严格 W32A32 基线 BOPS 为 `122842499383296.0`；统一搜索域为 20 个 CNN、12 个独立 Attention d_h、3 个 FFN d_ff，grouped-conv 域数量为 0。
- 离散最小可达 retention 为 `0.030587942852870195`，因此 `[0.045, 0.055]` 预算带可达。
- Greedy 从 retention=1.0 运行到无合法正 BOPS 降幅动作，共 1002 步；BOPS 单调不增、candidate hash 稳定唯一、每步 structural/precision/budget repair 均为 0。
- 预算带共有 1811 个候选。joint Taylor 优先 winner：hash `44551dcb6358b38447662e376ad1731d61862343d56da4c054103c784029547b`，retention `0.05491842298159359`，绝对偏差 `0.00491842298159359`，joint Taylor `62.76118526354641`。winner 的 Attention/FFN 宽度与 precision map 已写入 `greedy/v2xvit_greedy_winner_config.json`。
- Stage-2 Top-5 全部通过物理宽度、finite-forward、ONNX 与 requested/realized exact 审计；对 Top-1 winner 使用 fixed-K=27904 构建 TensorRT 10.9 strongly-typed engine，engine 成功，INT8 requested/realized 为 67/67，QK/Softmax compute FP32，Softmax output 为 FP16（A8 语义按 floating compute + output Q/DQ 审计）。
- joint candidate smoke10/fixed50 均完成；fixed50 AP@0.3=`0.5457148330906878`、AP@0.5=`0.12610580491800644`、AP@0.7=`0.00629311430358704`，forward p50=`11.078222072683275 ms`。这些是筛选 smoke，不是正式 latency 结论。
- strict baseline 与 precision-only control 均物理构建并完成 fixed50，用于 precision/structure effect 分解；proxy validation 因只有 1 个具备完整三类代理与 fixed50 的候选，标记 `insufficient_sample_count`，没有夸大相关性。

### 验证、提交与后续

- 相关定向测试：32 passed；`compileall` 与 `git diff --check` 通过。完整 pytest 为 973 passed、2 个既有 generic tracer Transformer tests 失败（`test_runtime_tracer_captures_imported_einsum_alias_and_restores_it`、`test_runtime_tracer_captures_matmul_operator`），未修改其 tracer 逻辑。
- 本轮未执行 formal GA、full1789、六预算长跑或正式 200 warmup/500 timed/5 repeats latency。
- 起始时正式分支实际观察到 `origin/feature/heal-unified-search-h800=460764fdecf9ea6298667586c033a51aa4b71c7d`，与任务给定预期 `896a049830874ef0d40faa87927873a1e88bedb6` 不一致；本任务未 checkout/reset/merge/cherry-pick 正式分支，需在最终报告中保留该血缘异常。
- 下一步：完成本分支提交与 push 后写入 final commit/remote 0 0；若后续授权，先为物理 d_h 后的 QKV W8A8 刷新 SmoothQuant scales，再做无争用正式 latency，最后才考虑六预算扩展。

---
时间戳：2026-07-24 14:45:00 CST｜轮次：Round 5
---

## Round 6：提交、推送与最终状态固化

- 隔离分支已提交为 `0aaa36a7486ee8095ff7932e2be6c605914adf8c`，并成功推送到 `origin/feature/h800-transformer-unified-search`；`git rev-list --left-right --count HEAD...origin/feature/h800-transformer-unified-search` 为 `0 0`。
- worktree 当前干净，正式分支仍为 `460764fdecf9ea6298667586c033a51aa4b71c7d`，其既有 3 个未跟踪文件未被修改；任务给定的正式预期 commit `896a049830874ef0d40faa87927873a1e88bedb6` 与实际远端血缘不一致，已写入 `reports/final_acceptance.json`，未对正式分支采取任何修复操作。
- 最终 after 进程审计重新生成：12 个快照进程，10 个为其他 Transformer alignment 生命周期、2 个为本任务/相关 shell 快照；signals_sent=0、external_paths_written_by_task=[]。before 快照中的 42 个 F-Cooper 进程在 after 快照自然消失，但没有证据归因于本任务，未发送信号。
- `reports/final_acceptance.json` 已固化 base/start/final commit、remote 0/0、四模型阶段状态、repair gate、domain 数、Stage-2 engine、fixed50、测试和禁止项；`root_conclusion.md` 与本 handoff 文档同步记录了正式 latency、GA、full1789、六预算均未执行。

---
时间戳：2026-07-24 14:52:00 CST｜轮次：Round 6
---

## Round 7：最终文档提交

- 为固化 Round 6 进程审计与同步记录，新增文档提交 `2e4f1bf99d162a060fba0c53cb4e4cfa951a6c2f`；该 commit 为当前分支最终提交，远端复核仍为 `0 0`。
- `reports/final_acceptance.json` 的 `final_commit` 已同步更新为 `2e4f1bf99d162a060fba0c53cb4e4cfa951a6c2f`；此前 `0aaa36a7` 为代码与实验实现提交。

---
时间戳：2026-07-24 14:55:00 CST｜轮次：Round 7
---

## Round 8：最终 commit 校正

- Round 7 文档记录本身产生了新的文档 commit；因此当前最终 commit 以本轮结束后的 Git 复核为准，避免把中间文档提交误当作最终血缘。
- 代码/实验实现提交为 `0aaa36a7486ee8095ff7932e2be6c605914adf8c`，随后文档同步提交为 `2e4f1bf99d162a060fba0c53cb4e4cfa951a6c2f`；下一次 push 后的最新 commit 将写入最终验收 JSON。

---
时间戳：2026-07-24 14:57:00 CST｜轮次：Round 8
---

## Round 9：最终远端复核

- Round 8 之后的文档提交已完成，当前最终 commit 为 `8649a9f0cf8141313bbd569027eaa9fcbe12d1e2`；`origin/feature/h800-transformer-unified-search` 与本地为 `0 0`，worktree clean。
- 最终验收 JSON 已采用该 commit 作为 `final_commit`；后续若再次修改本 handoff，必须同步更新 JSON 和远端计数。

---
时间戳：2026-07-24 15:00:00 CST｜轮次：Round 9
---

## Round 10：V2X-ViT 0.05 winner 精度崩塌根因归因

### 隔离、输入冻结与静态审计

- 本轮从隔离分支 commit `8ae8ae02473869cf55785dcc0afcd648ea7a93f3` 开始，只使用 worktree `/home/lixingfeng/UniAD_examine/heal_compress_h800_transformer_unified_search`。
- 新 run root 为 `/data/lxf/heal_data/outputs/h800_v2xvit_005_accuracy_collapse_attribution_20260724_053307/`；所有历史 artifact 均只读。
- winner hash `44551dcb6358b38447662e376ad1731d61862343d56da4c054103c784029547b`、physical hash `c045f7c1421948f651f83e4dce81ac7c68f0fecb515d792479b3ffe15695a082`、checkpoint SHA、fixed50 manifest hash 和 engine SHA 均唯一恢复并验证。
- 12 个 Attention d_h、3 个 FFN d_ff、20 个 CNN width 与上一轮完全一致；QK/VO coordinate coupling、head_dim scale、FFN coupling、strict state load、无 padding/mask-only 均通过静态审计。
- 旧 calibration 与正确 physical hash、checkpoint、manifest 绑定，没有跨结构复用；旧 key 缺少显式 `precision_map_hash`，因此 key 不完整但 `stale_cache_risk=false`。JMIX-FRESH 强制重新采集 4 帧训练 calibration 并生成 structure/precision/manifest 绑定的新 scale。

### 同结构分解与明确根因

- strict PyTorch fixed50：AP@0.3/0.5/0.7=`0.69904/0.60404/0.41958`，mAP=`0.57422`。
- winner 物理结构 S32 PyTorch：`0.56339/0.15892/0.00724`，mAP=`0.24318`。灾难性下降已经发生在物理 FP32 模型中，早于量化、ONNX 和 TensorRT。
- 同结构 TensorRT S16：`0.56226/0.15712/0.00720`，mAP=`0.24219`；没有出现额外 FP16 崩塌。PyTorch 原生 half 路径因 V2X scatter 的 index-put source/destination dtype 不一致而明确失败，未静默回退。
- 同结构 JMIX-FRESH TensorRT：`0.54883/0.12647/0.00651`，mAP=`0.22727`；fresh PTQ 只带来次级损失，无法恢复结构损失，且与上一轮旧 joint 基本一致。
- 最终分类为 `structural_collapse`，primary backend=`pytorch_physical`，不是 stale calibration、export semantic mismatch 或 TensorRT numeric mismatch。

### 结构子系统定位

- CNN-only mAP=`0.32458`，Attention-only=`0.38133`，FFN-only=`0.58038`；FFN 不是根因。
- `shrinker_m1.layers.0.double_conv.0` 单独从 256→28 时 mAP=`0.32496`；full winner 仅恢复 shrinker 后 mAP 上升到 `0.41873`，因此它是主要结构根因。
- stage0/stage1/stage2 单独物理剪枝的 mAP 分别为 `0.57521/0.58214/0.58207`，包括 stage1 的 width=16 域也没有单独崩塌。
- Attention 为次级来源：agent relation 和 w4 基本无损；w16 family mAP=`0.48137` 最弱，layer2 mAP=`0.50909` 比 layer0/1 更差。CNN+Attention mAP=`0.24195` 与 full winner 接近，说明两者组合放大损失。

### 后端、量化与限制

- S32 PyTorch/ORT/TensorRT mAP 分别为 `0.24318/0.24139/0.24319`，三后端在已崩塌精度上对齐，排除 S32 export/TRT 主因。
- ORT 使用明确标记的 ScatterND diagnostic bridge 替换 TensorRT-only scatter plugin；S16 被 FP16 Pad 类型绑定拒绝，JMIX-FRESH 被 FP16 QuantizeLinear 输入类型拒绝。两项为显式 ORT 图兼容性阻塞，不作为数值精度结果。
- Phase 4 INT8 role rescue 按规则未启动：其前提 S32/S16 正常不成立。本轮未启用 SmoothQuant、未搜索 alpha，`smoothquant_followup_recommended=false`。
- 没有完成跨三后端的逐层 max-abs/cosine tensor capture；已有 block/family 结构控制和 fixed50 后端指标对齐，缺项在 `intermediate_tensor_errors.csv` 中显式记录。

### 代码与验证

- 新增 provenance/static audit、physical control builder、PyTorch/ORT/TRT diagnostic runners、归因报告生成器与纯判定 helper；TensorRT fresh calibration key 现在显式包含 precision-map hash。
- 定向测试 `41 passed`；全量 pytest `977 passed, 2 failed`，仍是此前相同的 generic tracer einsum/matmul 两项已知失败。本轮无新增测试失败。
- `python -m compileall -q search scripts tests tools` 通过；提交前仍需复核 `git diff --check`。
- 本轮未运行 formal GA、六预算搜索、full1789、训练/微调、SmoothQuant、alpha 网格、LUT 扩展或正式 latency。
- before/after 进程快照均为 11 个匹配进程；外部 signals 和 external path writes 均为空，GPU 阶段只使用 GPU5。

---
时间戳：2026-07-24 21:54:07 CST｜轮次：Round 10
---

## Round 11：归因代码提交与远端同步

- 归因实现、测试和 Round 10 报告已提交为 `7da22f816aa3ea69e0209a3d0a94822f0b72771c`（`diagnose: attribute V2X-ViT greedy accuracy collapse`）。
- 该提交已推送到 `origin/feature/h800-transformer-unified-search`，push 后首次复核 `HEAD...origin/feature/h800-transformer-unified-search` 为 `0 0`。
- 正式分支仍观察为 `460764fdecf9ea6298667586c033a51aa4b71c7d`，本轮未对其 checkout/reset/merge/rebase 或写入。
- 本文档同步提交后，以最终 `git rev-parse HEAD` 和再次执行的远端 `0 0` 为最终血缘。

---
时间戳：2026-07-24 21:58:00 CST｜轮次：Round 11
---

## Round 12：weight-only absolute Taylor Greedy 与 GPU5 隔离状态

- 本轮从 clean commit `9d23c4a842e55c91814935dfadab614e19b94458` 开始，新增运行根目录 `/data/lxf/heal_data/outputs/h800_v2xvit_greedy005_weight_only_abs_taylor_20260725_010334/`；只在隔离 worktree 修改，正式 unified-search worktree 未触碰。
- Taylor 结构动作现在只统计当前状态到下一状态中新删除的耦合参数元素：`Σ_i [abs(g_i·(-w_i)) + 0.5·abs(h_i·w_i²)]`；权重量化动作只统计保留参数在当前→下一相邻精度间的量化增量：`Σ_i [abs(g_i·Δw_i) + 0.5·abs(h_i·Δw_i²)]`。两项均逐元素取绝对值后才聚合到参数、耦合组、层和样本，禁止符号抵消和量化风险返还。
- Fisher 采集新增 `absolute_gradients`，在样本聚合前保存逐元素绝对梯度；统计版本更新为 `common-task-loss-fisher-abs-reduction-v2`。Greedy 新循环只使用 weight-only Taylor，activation/joint/cross 项权重均为 0；真实 W8A8 部署脚本仍保留激活校准和 INT8 激活路径。
- 新增 `search/greedy/weight_only_abs.py`、V2X-ViT Greedy runner、B0/S32/JMIX-FRESH control engine builder、fixed500 evaluator、TensorRT 200 warmup/500×5 latency runner 和报告汇总脚本；Greedy 循环计数器设计为 forward/backward/physical export/ONNX/TRT build 全部为 0。
- 新增 `tests/test_weight_only_abs_taylor.py`，定向相关测试 `45 passed`，`compileall` 与 `git diff --check` 通过；全量 pytest 为 `981 passed, 2 failed`，仍是既有 generic tracer 的 einsum alias 与 matmul operator 两项失败，未新增失败。
- 新运行计划、before/after 进程快照均已保存。GPU5（`GPU-4d414d37-9a66-becc-0ffe-f5544e75fb38`）在实验前后仍被外部 PID `3625899`（`/exdata/jichengzhi/tvm310/bin/python`）占用，当前 GPU 利用率 100%；按照“只用 GPU5、不抢占外部进程”约束，Greedy、engine、fixed500 和正式 latency 阶段均安全延期，未使用其他 GPU，未发送任何外部信号。
- 代理实现提交 `202f095d0e13aa8d7e5e793955a6c584546f7782` 与延期报告提交均已推送；本轮最终提交为 `3d601d758ac936bcdd6b2282f3680e5d46af2798`，本地/远端为 `0 0`。本轮未启动 GA、六预算搜索、full1789、训练、SmoothQuant、alpha 搜索或 LUT 扩展。后续必须先重新确认 GPU5 连续空闲，再按固定顺序运行 Greedy→三控制 engine→500 帧→隔离延迟。

---
时间戳：2026-07-25 01:30:00 CST｜轮次：Round 12
---
