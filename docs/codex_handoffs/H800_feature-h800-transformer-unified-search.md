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
