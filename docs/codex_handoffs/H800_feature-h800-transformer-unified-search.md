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
