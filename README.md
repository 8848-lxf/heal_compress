# HEAL Unified Pruning and Mixed-Precision Search

该分支提供可公开发布的 HEAL 结构化剪枝与逐层混合位宽量化搜索框架，统一支持以下四个模型族：

- LiDAR Pyramid
- DiscoNet
- F-Cooper
- V2X-ViT

搜索协议由可复用模块组成：合法结构域、精度基因、Stage1 Taylor 代理、GA/Greedy、Beam Recovery、Stage2 TensorRT 评估、Greedy 锚点精度门和最佳引擎归档。仓库不包含模型、数据集、校准缓存、ONNX、TensorRT 引擎或搜索输出。

正式 V3 配置固定使用 64 个父代、64 个子代、每代最多 5 个新 Stage2 引擎，以及 10 个进化世代；generation 0 只用于初始化，不计入 10 代进化。

## 环境与路径

建议使用独立 conda 环境。正式运行前通过环境变量提供本机资源位置：

```bash
export MODEL_ROOT=/path/to/HEAL/models
export TENSORRT_ROOT=/path/to/TensorRT
export CALIBRATION_MANIFEST=/path/to/calibration_manifest.json
export GREEDY_ANCHOR_MANIFEST=/path/to/greedy/anchor_manifest.json
```

代码中的 HEAL 默认位置为仓库根目录相对路径 `../../HEAL`。也可通过命令行参数覆盖所有路径，不需要修改源码。

## 统一入口

查看支持的模型：

```bash
python -m search.unified --list-families
```

只检查配置与调用链，不加载模型、不运行 GA、不构建引擎：

```bash
python -m search.unified \
  --config search/configs/unified/lidar_pyramid_ga.yaml \
  --output-root /tmp/heal_search_dryrun \
  --dry-run
```

四个模板位于 `search/configs/unified/`。正式运行时应显式传入真实模型和协议文件：

```bash
python -m search.unified \
  --config search/configs/unified/lidar_disco_ga.yaml \
  --checkpoint /path/to/model.pth \
  --model-config /path/to/config.yaml \
  --heal-root ../../HEAL \
  --tensorrt-root /path/to/TensorRT \
  --calibration-manifest /path/to/calibration_manifest.json \
  --greedy-anchor-manifest /path/to/greedy/anchor_manifest.json \
  --output-root /path/to/new_search_output
```

Stage1 激活量化扰动 Taylor 项可显式切换：

```bash
python -m search.unified ... --activation-taylor on
python -m search.unified ... --activation-taylor off
```

启用后若没有完成激活统计和代理绑定，流程会失败关闭，不会静默退化为权重量化 Taylor。

## 正式搜索协议

1. 先用同一模型、校准清单、BOPS 目标和评估帧运行 Greedy，产生 `greedy/anchor_manifest.json`。Greedy 在直接轨迹无法到达预算时启用 Beam Recovery。
2. 再运行 GA。Stage2 候选必须满足 `candidate_mAP >= greedy_anchor_mAP - tolerance`，且 `tolerance` 的硬上限为 `0.005`。
3. 只有通过精度门、TensorRT 构建、请求/实现精度一致性和正式评估的候选才能参与最终排序。
4. 最佳可部署引擎会硬链接或复制到输出根目录下独立的 `best_engines/<family>/`，并写入 SHA256 清单。

V2X-ViT 的候选搜索与部署闭环分别由公共 Stage1/Stage2 模块和 `scripts/deploy_searched_heal_v2xvit.py` 提供。部署脚本不带机器私有默认路径，模型、训练清单、搜索产物、TensorRT 和插件都必须通过参数显式提供。

## 最小验证

以下检查不会执行完整搜索：

```bash
python -m pytest -q \
  tests/test_unified_search_public_api.py \
  tests/test_public_release_hygiene.py
python tools/check_public_release.py
```

提交或推送前必须通过公开发布检查。`outputs/`、`best_engines/`、`docs/codex_handoffs/` 以及模型和引擎后缀均已加入忽略与拒绝规则。
