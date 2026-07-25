# H800 V2X-ViT BOPS 与精度部署闭环修复交接

分支：`fix/h800-v2xvit-bops-precision-closure`

worktree：`/home/lixingfeng/UniAD_examine/heal_compress_h800_v2xvit_bops_precision_fix`

基线提交：`2d17d6e08cb30b6beb5412f462b71537542a973a`

运行根目录：`/data/lxf/heal_data/outputs/h800_v2xvit_bops_precision_closure_20260725_082219`

## Round 1：HGT BOPS 与 functional precision deployment closure

### 代码修改

- `search/proxy/transformer_bops.py` 将 HGT 的 `relation_att` 与
  `relation_msg` 两次真实关系矩阵乘加入生产 BOPS、MAC、参数量和
  mixed-weight storage；同时按真实 type-specific 参数副本修正 Q/K/V/O
  projection 参数量。生产公式版本升级为
  `unified-bops-v2-hgt-relation-closure`，没有修改历史预算标签或参数保留率定义。
- `search/pruning_space/transformer_domains.py` 为 HGT domain 增加 relation
  参数路径、relation count 与 operand precision 语义，供生产 BOPS 和独立审计共同验证。
- `search/adapters/transformer_models.py` 修复 LayerNorm 被 weighted-module
  active filter 错误删除的问题；所有 12 个真实 LayerNorm 都进入固定精度清单。
- `search/quantization_space/transformer_precision.py` 将不可独立实现的
  Softmax、AV 与 FFN activation loci 从 chromosome 移除或绑定到真实部署边界：
  QK、Softmax、AV、LayerNorm 固定 FP32；attention/FFN residual 与 window
  merge 固定 FP16；FFN activation 绑定 FFN2 输入 Q/DQ。最终真实搜索空间为
  113 个 policy group：53 个 mutable weighted gene 与 60 个固定 functional group。
- `scripts/run_v2xvit_greedy005_stage2.py` 在 Q/DQ 后显式建立 QK/logits/
  Softmax/AV 和 LayerNorm FP32 boundary；不再依赖 TensorRT tactic 的隐式提升。
- 新增 `search/stage2/v2xvit_functional_precision.py`，以 canonical ONNX identity
  和 Engine Inspector 逐项闭合 QK、Softmax、AV、LayerNorm、residual、merge
  以及 FFN activation input；missing/conflict/fallback 均 fail closed。
- 新增 maximal-mixed 构建与断点续建脚本。断点续建只复用已绑定 provenance
  的 ONNX/QDQ/train200，不重复导出或校准，只执行一次真实 TensorRT build。
- `scripts/audit_v2xvit_transformer_bops_precision.py` 改为使用真实生产
  `TransformerBOPSProxy` 与完全独立的 runtime-shape calculator 对账，并接入
  representative engine 的 weighted/functional requested-realized 证据。

### 审计与真实部署结果

- 三个固定样本的 12 个 Attention、3 个 FFN 及共享调用已完成 runtime trace。
  HGT relation、Q/K/V/O、QK、AV、FFN 均覆盖；production 与独立 MAC/BOPS
  对账通过。
- 37 个 shrinker/attention d_h 边际动作逐项对账，最大相对误差为 0，未发现
  重复收益或共享调用重复计数。
- strict B0 使用新公式时 `R_BOPS=1.0`；原始宽度 maximal-mixed profile 的
  `R_BOPS=0.32931979066827877`。历史候选标签未被重写。
- 固定清单：12 QK + 12 Softmax + 12 AV + 12 LayerNorm 为 FP32；9 residual
  + 3 merge 为 FP16。当前默认保护并非只有 QK。
- 代表性 maximal-mixed candidate `7291d250...999dc` 在 GPU2 构建成功；
  train200 为 200 processed、0 skipped；engine SHA256 为
  `f1bb184265448e7b8bebf63d7cd841c1987f195db9c8a17cce31c2a2d7071718`。
- weighted requested-realized、attention FP32 与 63 条 functional inspector
  mapping 全部精确：0 unmapped、0 conflict、0 fallback。
- 最终 `reports/final_audit_acceptance.json`：`bops_audit_passed=true`、
  `precision_protection_audit_passed=true`、`formal_search_allowed=true`，blockers 为空。
- 本轮没有重跑 Greedy、正式 GA 或 full1789。`formal_search_allowed=true` 只代表
  两项阻塞审计解除，不表示本轮已经启动搜索。

### 测试

- 新增 HGT relation MAC/BOPS、共享调用、functional canonical mapping、
  LayerNorm FP32 rewrite、requested-realized fail-closed 等测试。
- 定向回归 `20 passed`；全量 pytest `1012 passed, 82 warnings, 0 failed`。
  compileall、py_compile 与 `git diff --check` 均通过。

---
时间戳：2026-07-26 00:05:08 CST｜轮次：Round 1
---
