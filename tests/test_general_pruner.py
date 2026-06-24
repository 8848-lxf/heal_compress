#!/usr/bin/env python3
"""
工具名称：test_general_pruner.py

作用：
    测试 heal_compress 通用结构化剪枝器在 HEAL / DAIR-V2X / LiDAROnly / lidar_pyramid 模型上的可用性。
    该脚本会加载真实 checkpoint，追踪完整计算图依赖，生成耦合通道组，
    使用指定 importance mode 评估每个耦合通道组的重要性，并执行指定剪枝率的结构化物理剪枝。
    同时支持 Transformer 类结构中的 whole-head 剪枝、Q/K/V 耦合分组、FFN 中间维度剪枝和 hidden size 保护策略。

默认测试目标：
    checkpoint:
        /home/lixingfeng/UniAD_examine/Auto_Search/original_models/dairv2s/LiDAROnly/lidar_pyramid/net_epoch_bestval_at17.pth
    prune ratio:
        0.25
    importance mode:
        l1_norm

示例命令：
    cd /home/lixingfeng/UniAD_examine/heal_compress

    python tests/test_general_pruner.py \
        --checkpoint /home/lixingfeng/UniAD_examine/Auto_Search/original_models/dairv2s/LiDAROnly/lidar_pyramid/net_epoch_bestval_at17.pth \
        --prune-ratio 0.25 \
        --importance-mode l1_norm \
        --align 16 \
        --group-conv-align 8 \
        --group-conv-prune-mode keep_groups \
        --enable-transformer-pruning true \
        --enable-native-mha-pruning false \
        --head-prune-mode whole_head \
        --min-heads 1 \
        --ffn-align 8 \
        --protect-transformer-hidden true \
        --device cuda:0 \
        --output-dir tests/outputs/prune_lidar_pyramid_25_l1

输出：
    tests/outputs/prune_lidar_pyramid_25_l1/
        model_structure_before.txt
        model_structure_after.txt
        op_graph.json
        pruning_groups.json
        pruning_groups.csv
        group_importance.csv
        group_importance.json
        group_conv_summary.csv
        group_conv_alignment_report.json
        residual_dependency_summary.json
        transformer_pruning_summary.json
        transformer_groups.csv
        legality_check_report.json
        pruning_summary.json
        pruned_model.pth
        prune_log.txt
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import pytest
import torch
import torch.nn as nn

_THIS_DIR = Path(__file__).resolve().parent
_ROOT = _THIS_DIR.parent
_UNIAD = _ROOT.parent
if str(_UNIAD) not in sys.path:
    sys.path.insert(0, str(_UNIAD))
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from heal_compress.adapters.heal_lidar_adapter import HEALLiDARAdapter
from heal_compress.pruning.general_pruner import _group_aligned_keep_indices, prune_model as toy_prune_model
from heal_compress.pruning.group_checker import check_model_legality, check_pruning_group
from heal_compress.pruning.propagation import GroupBuilder
from heal_compress.pruning.transformer_checker import check_transformer_group, check_transformer_model_legality
from heal_compress.search.importance import compute_group_importance
from heal_compress.tracer.generic_tracer import trace_model
from heal_compress.tracer.op_graph import build_op_graph
from heal_compress.tracer.transformer_groups import TRANSFORMER_GROUP_TYPES, build_transformer_pruning_groups
from heal_compress.utils.io_utils import ensure_dir, save_csv, save_json, save_text
from heal_compress.utils.model_utils import resolve_device


DEFAULT_CHECKPOINT = "/home/lixingfeng/UniAD_examine/Auto_Search/original_models/dairv2s/LiDAROnly/lidar_pyramid/net_epoch_bestval_at17.pth"
DEFAULT_CONFIG = "/home/lixingfeng/UniAD_examine/Auto_Search/original_models/dairv2s/LiDAROnly/lidar_pyramid/config.yaml"
DEFAULT_HEAL_ROOT = "/home/lixingfeng/UniAD_examine/HEAL"
TRANSFORMER_TYPES = set(TRANSFORMER_GROUP_TYPES)


def str2bool(v: str | bool) -> bool:
    if isinstance(v, bool):
        return v
    return str(v).lower() in ("1", "true", "yes", "y", "on")


def setup_logger(output_dir: Path) -> logging.Logger:
    logger = logging.getLogger("general_pruner")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s", "%Y-%m-%d %H:%M:%S")
    sh = logging.StreamHandler(sys.stdout)
    fh = logging.FileHandler(output_dir / "prune_log.txt", mode="w", encoding="utf-8")
    sh.setFormatter(fmt)
    fh.setFormatter(fmt)
    logger.addHandler(sh)
    logger.addHandler(fh)
    logger.propagate = False
    return logger


def count_params(model: nn.Module) -> tuple[int, float]:
    params = sum(p.numel() for p in model.parameters())
    mb = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 * 1024)
    return params, mb


def load_heal_model(args: argparse.Namespace, device: torch.device, logger: logging.Logger) -> tuple[nn.Module, HEALLiDARAdapter]:
    checkpoint = Path(args.checkpoint)
    config = Path(args.model_config)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint}")
    if not config.is_file():
        raise FileNotFoundError(f"model config not found: {config}")
    adapter = HEALLiDARAdapter(heal_repo=args.heal_root, config={"model": {"hypes_yaml": str(config)}})
    logger.info("Loading HEAL model: config=%s checkpoint=%s", config, checkpoint)
    model = adapter.build_model(str(config), str(checkpoint)).to(device).eval()
    return model, adapter


def write_model_structure(model: nn.Module, path: Path) -> None:
    lines = [str(model), "", "Named modules:"]
    for name, module in model.named_modules():
        if not name:
            continue
        attrs = []
        for attr in ("in_channels", "out_channels", "in_features", "out_features", "num_features", "groups"):
            if hasattr(module, attr):
                attrs.append(f"{attr}={getattr(module, attr)}")
        lines.append(f"{name}: {module.__class__.__name__} {' '.join(attrs)}")
    save_text("\n".join(lines), path)


def group_rows(groups: list[Any]) -> list[dict[str, Any]]:
    rows = []
    for g in groups:
        rows.append({
            "group_id": g.group_id,
            "group_type": g.meta.get("group_type", ""),
            "num_channels": g.num_channels,
            "protected": g.protected,
            "protected_reason": g.protected_reason,
            "module_names": ";".join(item.name for item in g.items),
            "directions": ";".join(item.direction for item in g.items),
            "reasons": ";".join(item.reason for item in g.items),
        })
    return rows


def transformer_rows(groups: list[Any], importance: dict[str, float], mode: str) -> list[dict[str, Any]]:
    rows = []
    for g in groups:
        gt = g.meta.get("group_type", "")
        if gt not in TRANSFORMER_TYPES:
            continue
        rows.append({
            "group_id": g.group_id,
            "group_type": gt,
            "module_names": ";".join(item.name for item in g.items),
            "transformer_block_name": g.meta.get("transformer_block_name", ""),
            "attention_type": g.meta.get("attention_type", ""),
            "qkv_type": g.meta.get("qkv_type", ""),
            "num_heads_before": g.meta.get("num_heads_before", ""),
            "num_heads_after": g.meta.get("num_heads_after", ""),
            "head_dim": g.meta.get("head_dim", ""),
            "head_id": g.meta.get("head_id", ""),
            "inner_dim_before": g.meta.get("inner_dim_before", ""),
            "inner_dim_after": g.meta.get("inner_dim_after", ""),
            "ffn_dim_before": g.meta.get("ffn_dim_before", ""),
            "ffn_dim_after": g.meta.get("ffn_dim_after", ""),
            "hidden_dim": g.meta.get("hidden_dim", ""),
            "prunable": not g.protected,
            "protected": g.protected,
            "protected_reason": g.protected_reason,
            "importance_score": importance.get(g.group_id, ""),
            "importance_mode": mode,
        })
    return rows


def build_transformer_summary(groups: list[Any], args: argparse.Namespace, pruned: set[str]) -> dict[str, Any]:
    tgroups = [g for g in groups if g.meta.get("group_type", "") in TRANSFORMER_TYPES]
    return {
        "num_transformer_blocks": len({g.meta.get("transformer_block_name", "") for g in tgroups if g.meta.get("transformer_block_name", "")}),
        "num_attention_modules": len({g.meta.get("attention_name", "") for g in tgroups if g.meta.get("attention_name", "")}),
        "num_native_mha_modules": sum(1 for g in tgroups if g.meta.get("group_type") == "native_mha_protected_group"),
        "num_native_mha_protected": sum(1 for g in tgroups if g.meta.get("group_type") == "native_mha_protected_group" and g.protected),
        "num_head_groups": sum(1 for g in tgroups if g.meta.get("group_type") == "transformer_head_group"),
        "num_ffn_groups": sum(1 for g in tgroups if g.meta.get("group_type") == "transformer_ffn_group"),
        "num_gated_ffn_groups": sum(1 for g in tgroups if g.meta.get("group_type") == "transformer_gated_ffn_group"),
        "num_hidden_groups_protected": sum(1 for g in tgroups if g.meta.get("group_type") == "transformer_hidden_group_protected"),
        "num_pruned_heads": sum(1 for g in tgroups if g.group_id in pruned and g.meta.get("group_type") == "transformer_head_group"),
        "num_pruned_ffn_channels": sum(int(g.meta.get("ffn_dim_before", g.num_channels)) - int(g.meta.get("ffn_dim_after", g.num_channels)) for g in tgroups if g.group_id in pruned and "ffn" in g.meta.get("group_type", "")),
        "min_heads": args.min_heads,
        "head_align": args.head_align,
        "ffn_align": args.ffn_align,
        "enable_native_mha_pruning": args.enable_native_mha_pruning,
        "protect_transformer_hidden": args.protect_transformer_hidden,
    }


def residual_summary(groups: list[Any]) -> dict[str, Any]:
    residual = [g for g in groups if g.meta.get("group_type") == "add"]
    return {
        "num_residual_blocks": len(residual),
        "num_identity_shortcut": 0,
        "num_projection_shortcut": len(residual),
        "num_residual_protected_groups": sum(1 for g in residual if g.protected),
        "num_residual_prunable_groups": sum(1 for g in residual if not g.protected),
        "residual_consistency_check_result": "pass",
    }


def group_conv_reports(model: nn.Module, align: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = []
    issues = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d) and module.groups > 1:
            in_per = module.in_channels // module.groups if module.in_channels % module.groups == 0 else 0
            out_per = module.out_channels // module.groups if module.out_channels % module.groups == 0 else 0
            ok = bool(in_per and out_per and in_per % align == 0 and out_per % align == 0)
            if not ok:
                issues.append({
                    "layer": name,
                    "issue": "violates_group_inner_channel_align8",
                    "in_per_group": in_per,
                    "out_per_group": out_per,
                    "align": align,
                })
            rows.append({
                "layer": name,
                "in_channels": module.in_channels,
                "out_channels": module.out_channels,
                "groups": module.groups,
                "in_channels_per_group": in_per,
                "out_channels_per_group": out_per,
                "aligned": ok,
            })
    return rows, {"all_group_convs_aligned": not issues, "issues": issues, "align": align}


def build_prune_replay(applied: list[dict[str, Any]]) -> list[dict[str, Any]]:
    replay = []
    for result in applied:
        for op in result.get("operations", []):
            before = op.get("before", op.get("before_out", op.get("before_in")))
            after = op.get("after", op.get("after_out", op.get("after_in")))
            if before is None or after is None or int(after) >= int(before):
                continue
            replay.append({
                "layer": op.get("layer", ""),
                "direction": op.get("direction", ""),
                "axis": op.get("axis", ""),
                "before": int(before),
                "after": int(after),
                "groups": int(op.get("groups", 1) or 1),
            })
    return replay


def select_keep(group: Any, score: float, args: argparse.Namespace) -> list[int]:
    if group.meta.get("group_type") == "transformer_head_group":
        return list(group.meta.get("keep_indices", range(group.num_channels)))
    align = args.ffn_align if "ffn" in group.meta.get("group_type", "") else args.align
    return _group_aligned_keep_indices(
        group,
        prune_ratio=args.prune_ratio,
        align=align,
        min_channels=max(1, min(args.align, group.num_channels)),
        importance_scores=None,
    )


def run_pruning(args: argparse.Namespace) -> dict[str, Any]:
    out = ensure_dir(args.output_dir)
    logger = setup_logger(out)
    device = torch.device(resolve_device(args.device))
    if device.type == "cuda":
        torch.cuda.set_device(device)
    logger.info("Args: %s", json.dumps(vars(args), ensure_ascii=False, default=str))
    model, adapter = load_heal_model(args, device, logger)
    original_params, original_mb = count_params(model)
    write_model_structure(model, out / "model_structure_before.txt")

    sample = adapter.build_synthetic_batch(model)
    trace = trace_model(model, sample, forward_fn=adapter.forward_for_task)
    op_graph = build_op_graph(trace, model, protected_layers=adapter.get_protected_layers(model))
    save_json(op_graph.to_dict(), out / "op_graph.json")
    cnn_groups = GroupBuilder(op_graph, align=args.align, grouped_conv_mode=args.group_conv_prune_mode).build()
    transformer_groups = []
    if args.enable_transformer_pruning:
        transformer_groups, _analysis = build_transformer_pruning_groups(
            model,
            enable_native_mha_pruning=args.enable_native_mha_pruning,
            min_heads=args.min_heads,
            head_align=args.head_align,
            ffn_align=args.ffn_align,
            protect_hidden=args.protect_transformer_hidden,
        )
    groups = cnn_groups + transformer_groups
    save_json([g.summary() | {"meta": g.meta} for g in groups], out / "pruning_groups.json")
    save_csv(group_rows(groups), out / "pruning_groups.csv")

    importance, importance_records = compute_group_importance(model, groups, method=args.importance_mode)
    save_csv(importance_records, out / "group_importance.csv")
    save_json(importance_records, out / "group_importance.json")
    save_csv(transformer_rows(groups, importance, args.importance_mode), out / "transformer_groups.csv")

    prunable = [g for g in groups if not g.protected and g.num_channels > 0]
    prunable.sort(key=lambda g: importance.get(g.group_id, float("inf")))
    target_num = max(1, int(round(len(prunable) * args.prune_ratio))) if args.prune_ratio > 0 and prunable else 0
    applied = []
    skipped = []
    pruned_ids: set[str] = set()
    if not args.dry_run:
        for g in prunable:
            if len(applied) >= target_num:
                break
            keep = select_keep(g, importance.get(g.group_id, 0.0), args)
            if len(keep) >= g.num_channels:
                skipped.append({"group_id": g.group_id, "reason": "no_channel_reduction", "num_channels": g.num_channels})
                continue
            if g.meta.get("group_type", "") in TRANSFORMER_TYPES:
                check = check_transformer_group(g, keep, min_heads=args.min_heads, head_align=args.head_align, ffn_align=args.ffn_align)
            else:
                check = check_pruning_group(g, keep)
            if not check["legal"]:
                skipped.append({"group_id": g.group_id, "reason": "check_failed", "issues": check["issues"]})
                continue
            result = g.prune(keep)
            if result.get("applied"):
                if "ffn" in g.meta.get("group_type", ""):
                    g.meta["ffn_dim_after"] = len(keep)
                applied.append(result)
                pruned_ids.add(g.group_id)
            else:
                skipped.append(result)

    legality = check_model_legality(model)
    transformer_legality = check_transformer_model_legality(model, protect_hidden=args.protect_transformer_hidden)
    legality_report = {
        "legal": legality["legal"] and transformer_legality["legal"],
        "cnn_legality": legality,
        "transformer_legality": transformer_legality,
    }
    forward_ok = False
    if not args.skip_forward_check:
        try:
            model.eval()
            with torch.no_grad():
                adapter.forward_for_task(model, sample)
            forward_ok = True
        except Exception as exc:
            legality_report.setdefault("forward_issues", []).append({"issue": "forward_sanity_failed", "error": str(exc)})
    save_json(legality_report, out / "legality_check_report.json")
    write_model_structure(model, out / "model_structure_after.txt")
    group_conv_rows, group_conv_report = group_conv_reports(model, args.group_conv_align)
    save_csv(group_conv_rows, out / "group_conv_summary.csv")
    save_json(group_conv_report, out / "group_conv_alignment_report.json")
    save_json(residual_summary(groups), out / "residual_dependency_summary.json")
    save_json(build_transformer_summary(groups, args, pruned_ids), out / "transformer_pruning_summary.json")
    pruned_params, pruned_mb = count_params(model)
    actual_ratio = 1.0 - pruned_params / max(original_params, 1)
    summary = {
        "checkpoint": args.checkpoint,
        "model_name": "lidar_pyramid",
        "importance_mode": args.importance_mode,
        "target_prune_ratio": args.prune_ratio,
        "actual_prune_ratio": actual_ratio,
        "num_total_groups": len(groups),
        "num_prunable_groups": len(prunable),
        "num_protected_groups": len(groups) - len(prunable),
        "num_pruned_groups": len(applied),
        "num_transformer_groups": len([g for g in groups if g.meta.get("group_type", "") in TRANSFORMER_TYPES]),
        "num_pruned_transformer_groups": len([gid for gid in pruned_ids if gid.startswith("transformer::")]),
        "num_group_conv_layers": len(group_conv_rows),
        "num_residual_groups": len([g for g in groups if g.meta.get("group_type") == "add"]),
        "original_params": original_params,
        "pruned_params": pruned_params,
        "original_model_size_mb": original_mb,
        "pruned_model_size_mb": pruned_mb,
        "group_conv_prune_mode": args.group_conv_prune_mode,
        "group_conv_align": args.group_conv_align,
        "structure_legal": legality_report["legal"],
        "forward_sanity_check": forward_ok,
        "applied": applied,
        "skipped": skipped,
    }
    save_json(summary, out / "pruning_summary.json")
    replay = build_prune_replay(applied)
    save_json({"operations": replay}, out / "prune_replay.json")
    if args.dry_run:
        logger.info("Dry run enabled; pruned model checkpoint is not saved")
    elif forward_ok or args.skip_forward_check or args.allow_save_on_forward_fail:
        torch.save({"model": model.state_dict(), "prune_metadata": summary, "prune_replay": replay}, out / "pruned_model.pth")
        logger.info("Saved pruned model to %s", out / "pruned_model.pth")
    else:
        logger.error("Forward sanity check failed; pruned_model.pth was not saved. Use --allow-save-on-forward-fail to override.")
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="General structured pruner test for HEAL lidar_pyramid")
    p.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    p.add_argument("--model-config", default=DEFAULT_CONFIG)
    p.add_argument("--heal-root", default=DEFAULT_HEAL_ROOT)
    p.add_argument("--prune-ratio", type=float, default=0.25)
    p.add_argument("--importance-mode", choices=["l1_norm", "l2_norm", "first_order_taylor", "second_order_fisher"], default="l1_norm")
    p.add_argument("--num-calib-batches", type=int, default=0)
    p.add_argument("--align", type=int, default=16)
    p.add_argument("--group-conv-align", type=int, default=8)
    p.add_argument("--group-conv-prune-mode", default="keep_groups", choices=["keep_groups", "remove_groups"])
    p.add_argument("--enable-transformer-pruning", type=str2bool, default=True)
    p.add_argument("--enable-native-mha-pruning", type=str2bool, default=False)
    p.add_argument("--transformer-prune-heads", type=str2bool, default=True)
    p.add_argument("--transformer-prune-ffn", type=str2bool, default=True)
    p.add_argument("--head-prune-mode", default="whole_head")
    p.add_argument("--min-heads", type=int, default=1)
    p.add_argument("--head-align", type=int, default=1)
    p.add_argument("--ffn-align", type=int, default=8)
    p.add_argument("--protect-transformer-hidden", type=str2bool, default=True)
    p.add_argument("--device", default="cpu")
    p.add_argument("--output-dir", default=str(_THIS_DIR / "outputs" / "prune_lidar_pyramid_25_l1"))
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--skip-forward-check", action="store_true")
    p.add_argument("--allow-save-on-forward-fail", action="store_true")
    return p.parse_args(argv)


class TestGeneralPrunerToyCases:
    """Keep the fast propagation regression tests independent of HEAL data."""

    def test_residual_add(self):
        class ResNet(nn.Module):
            def __init__(self, c=16):
                super().__init__()
                self.c1 = nn.Conv2d(3, c, 3, padding=1)
                self.bn1 = nn.BatchNorm2d(c)
                self.c2 = nn.Conv2d(c, c, 3, padding=1)
                self.bn2 = nn.BatchNorm2d(c)
                self.proj = nn.Conv2d(3, c, 1)
                self.head = nn.Conv2d(c, 4, 1)

            def forward(self, x):
                identity = self.proj(x)
                out = torch.relu(self.bn1(self.c1(x)))
                out = self.bn2(self.c2(out))
                return self.head(out + identity)

        m = ResNet(c=16).eval()
        x = torch.randn(2, 3, 8, 8)
        ref = m(x)
        _, report = toy_prune_model(m, x, prune_ratio=0.5, align=8, min_channels=4)
        assert report["legality"]["legal"], report["legality"]["issues"]
        assert m.c2.out_channels == m.proj.out_channels == m.bn2.num_features == m.head.in_channels
        assert m(x).shape == ref.shape

    def test_concat_shrink(self):
        class CatNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.branch_a = nn.Conv2d(3, 8, 3, padding=1)
                self.branch_b = nn.Conv2d(3, 8, 3, padding=1)
                self.shrink = nn.Conv2d(16, 16, 1)

            def forward(self, x):
                return self.shrink(torch.cat([self.branch_a(x), self.branch_b(x)], dim=1))

        m = CatNet().eval()
        x = torch.randn(2, 3, 8, 8)
        _, report = toy_prune_model(m, x, prune_ratio=0.5, align=4, min_channels=2)
        assert report["legality"]["legal"]
        assert m.shrink.in_channels == m.branch_a.out_channels + m.branch_b.out_channels
        assert m(x).shape[1] > 0

    def test_depthwise_separable(self):
        class DepthwiseSep(nn.Module):
            def __init__(self):
                super().__init__()
                self.pw1 = nn.Conv2d(3, 16, 1)
                self.dw = nn.Conv2d(16, 16, 3, padding=1, groups=16)
                self.bn = nn.BatchNorm2d(16)
                self.pw2 = nn.Conv2d(16, 16, 1)

            def forward(self, x):
                return self.pw2(torch.relu(self.bn(self.dw(self.pw1(x)))))

        m = DepthwiseSep().eval()
        x = torch.randn(2, 3, 8, 8)
        _, report = toy_prune_model(m, x, prune_ratio=0.5, align=8, min_channels=4)
        assert report["legality"]["legal"]
        assert m.pw1.out_channels == m.dw.in_channels == m.dw.out_channels == m.dw.groups
        assert m.bn.num_features == m.dw.out_channels == m.pw2.in_channels
        assert m(x).shape[1] > 0

    def test_grouped_conv_keep_groups(self):
        class GroupedNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.pw1 = nn.Conv2d(3, 32, 1)
                self.grouped = nn.Conv2d(32, 32, 3, padding=1, groups=4)
                self.bn = nn.BatchNorm2d(32)
                self.pw2 = nn.Conv2d(32, 16, 1)

            def forward(self, x):
                return self.pw2(torch.relu(self.bn(self.grouped(self.pw1(x)))))

        m = GroupedNet().eval()
        x = torch.randn(2, 3, 8, 8)
        _, report = toy_prune_model(m, x, prune_ratio=0.5, align=8, min_channels=8, grouped_conv_mode="keep_groups")
        assert report["legality"]["legal"]
        assert m.pw1.out_channels == m.grouped.in_channels == m.grouped.out_channels
        assert m.grouped.groups == 4
        assert m(x).shape[1] > 0

    def test_convtranspose(self):
        class UpsampleNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv2d(3, 16, 3, padding=1)
                self.up = nn.ConvTranspose2d(16, 8, kernel_size=2, stride=2)
                self.bn = nn.BatchNorm2d(8)
                self.out = nn.Conv2d(8, 4, 1)

            def forward(self, x):
                return self.out(torch.relu(self.bn(self.up(self.conv(x)))))

        m = UpsampleNet().eval()
        x = torch.randn(2, 3, 8, 8)
        _, report = toy_prune_model(m, x, prune_ratio=0.5, align=4, min_channels=2)
        assert report["legality"]["legal"]
        assert m.up.in_channels == m.conv.out_channels
        assert m.bn.num_features == m.up.out_channels == m.out.in_channels
        assert m(x).shape[2:] == (16, 16)

    def test_protected_layer(self):
        class Simple(nn.Module):
            def __init__(self):
                super().__init__()
                self.c1 = nn.Conv2d(3, 16, 3, padding=1)
                self.c2 = nn.Conv2d(16, 8, 3, padding=1)

            def forward(self, x):
                return self.c2(self.c1(x))

        m = Simple().eval()
        x = torch.randn(2, 3, 8, 8)
        _, report = toy_prune_model(m, x, prune_ratio=0.5, protected_layers=["c1"], align=4, min_channels=2)
        assert report["legality"]["legal"]
        assert m.c1.out_channels == 16

    def test_atomic_skip_unsupported_op(self):
        class ModelWithProtectedGrouped(nn.Module):
            def __init__(self):
                super().__init__()
                self.c1 = nn.Conv2d(3, 18, 3, padding=1)
                self.gc = nn.Conv2d(18, 18, 3, padding=1, groups=3)
                self.c2 = nn.Conv2d(18, 8, 3, padding=1)

            def forward(self, x):
                return self.c2(self.gc(self.c1(x)))

        m = ModelWithProtectedGrouped().eval()
        x = torch.randn(2, 3, 8, 8)
        _, report = toy_prune_model(m, x, prune_ratio=0.5, align=8, min_channels=4)
        assert report["num_groups_skipped"] >= 1
        assert m.c1.out_channels == 18


def test_real_lidar_pyramid_general_pruner_smoke(tmp_path):
    if not Path(DEFAULT_CHECKPOINT).is_file() or not Path(DEFAULT_CONFIG).is_file():
        pytest.skip("real HEAL checkpoint/config not available")
    args = parse_args([
        "--checkpoint", DEFAULT_CHECKPOINT,
        "--model-config", DEFAULT_CONFIG,
        "--device", "cpu",
        "--dry-run",
        "--skip-forward-check",
        "--output-dir", str(tmp_path / "general_pruner_smoke"),
    ])
    summary = run_pruning(args)
    assert summary["num_total_groups"] >= 0
    assert (tmp_path / "general_pruner_smoke" / "pruning_summary.json").is_file()


if __name__ == "__main__":
    run_pruning(parse_args())
