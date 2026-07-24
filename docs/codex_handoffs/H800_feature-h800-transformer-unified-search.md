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
