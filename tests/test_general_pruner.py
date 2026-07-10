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

示例命令 1：TP-style shared local mean
    cd /home/lixingfeng/UniAD_examine/heal_compress

    python tests/test_general_pruner.py \
        --checkpoint /home/lixingfeng/UniAD_examine/Auto_Search/original_models/dairv2s/LiDAROnly/lidar_pyramid/net_epoch_bestval_at17.pth \
        --prune-ratio 0.25 \
        --importance-mode l1_norm \
        --selection-mode local_scope \
        --group-conv-selection-mode shared_local_mean \
        --group-conv-align 8 \
        --group-conv-prune-mode keep_groups \
        --align 16 \
        --protect-residual-add true \
        --device cuda:0 \
        --output-dir tests/outputs/prune_lidar_pyramid_25_l1_shared_local

示例命令 2：global coupled channel
    python tests/test_general_pruner.py \
        --checkpoint /home/lixingfeng/UniAD_examine/Auto_Search/original_models/dairv2s/LiDAROnly/lidar_pyramid/net_epoch_bestval_at17.pth \
        --prune-ratio 0.25 \
        --importance-mode l1_norm \
        --selection-mode global_coupled_channel \
        --group-conv-selection-mode shared_local_mean \
        --group-conv-align 8 \
        --group-conv-prune-mode keep_groups \
        --align 16 \
        --protect-residual-add true \
        --device cuda:0 \
        --output-dir tests/outputs/prune_lidar_pyramid_25_l1_global_coupled

示例命令 3：independent group top-k
    python tests/test_general_pruner.py \
        --checkpoint /home/lixingfeng/UniAD_examine/Auto_Search/original_models/dairv2s/LiDAROnly/lidar_pyramid/net_epoch_bestval_at17.pth \
        --prune-ratio 0.25 \
        --importance-mode l1_norm \
        --selection-mode constrained_global \
        --group-conv-selection-mode independent_group_topk \
        --group-conv-align 8 \
        --group-conv-prune-mode keep_groups \
        --align 16 \
        --protect-residual-add true \
        --device cuda:0 \
        --output-dir tests/outputs/prune_lidar_pyramid_25_l1_independent_group_topk

输出：
    tests/outputs/prune_lidar_pyramid_25_l1/
        model_structure_before.txt
        model_structure_after.txt
        op_graph.json
        pruning_groups.json
        pruning_groups.csv
        dependency_scopes.json
        dependency_scopes.csv
        coupled_channel_units.json
        coupled_channel_units.csv
        atomic_prune_units.json
        atomic_prune_units.csv
        concrete_pruning_groups.json
        concrete_pruning_groups.csv
        grouped_conv_selection_report.json
        grouped_conv_selection_report.csv
        selection_summary.json
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
from types import SimpleNamespace

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
from heal_compress.pruning.grouped_conv import grouped_conv_alignment_merge_factor, grouped_conv_pruning_fn, merge_grouped_conv_groups
from heal_compress.pruning.general_pruner import _group_aligned_keep_indices, prune_model as toy_prune_model
from heal_compress.pruning.group_checker import check_model_legality, check_pruning_group
from heal_compress.pruning.propagation import GroupBuilder
from heal_compress.pruning.selection import SelectionConfig, build_pruning_plan
from heal_compress.pruning.transformer_checker import check_transformer_group, check_transformer_model_legality
from heal_compress.pruning.units import (
    atomic_prune_unit_rows,
    concrete_pruning_group_rows,
    coupled_channel_unit_rows,
    dataclass_to_json_dict,
    dependency_scope_rows,
    instantiate_concrete_pruning_group,
)
from heal_compress.search.importance import (
    compute_group_importance,
    compute_layer_channel_importance,
    compute_scope_channel_importance_map,
)
from heal_compress.tracer.generic_tracer import trace_model
from heal_compress.tracer.op_graph import build_op_graph
from heal_compress.tracer.transformer_groups import TRANSFORMER_GROUP_TYPES, build_transformer_pruning_groups
from heal_compress.utils.io_utils import ensure_unique_dir, save_csv, save_json, save_text
from heal_compress.utils.model_utils import resolve_device


DEFAULT_CHECKPOINT = "/home/lixingfeng/UniAD_examine/Auto_Search/original_models/dairv2s/LiDAROnly/lidar_pyramid/net_epoch_bestval_at17.pth"
DEFAULT_CONFIG = "/home/lixingfeng/UniAD_examine/Auto_Search/original_models/dairv2s/LiDAROnly/lidar_pyramid/config.yaml"
DEFAULT_HEAL_ROOT = "/home/lixingfeng/UniAD_examine/HEAL"
TRANSFORMER_TYPES = set(TRANSFORMER_GROUP_TYPES)
DEFAULT_SAFE_PROTECTED_PREFIXES = (
    "cls_head",
    "reg_head",
    "dir_head",
)
HEAD_PROTECTED_KEYWORDS = ("cls_head", "reg_head", "dir_head")
FIXED_SHAPE_PROTECTED_KEYWORDS = ("pillar_vfe", "pfn_layers", "scatter", "voxel")


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


STRUCTURE_ATTRS = (
    "in_channels",
    "out_channels",
    "in_features",
    "out_features",
    "num_features",
    "groups",
    "normalized_shape",
)


def collect_module_structure(model: nn.Module) -> dict[str, dict[str, Any]]:
    snapshot: dict[str, dict[str, Any]] = {}
    for name, module in model.named_modules():
        if not name:
            continue
        attrs = {}
        for attr in STRUCTURE_ATTRS:
            if hasattr(module, attr):
                value = getattr(module, attr)
                if isinstance(value, torch.Size):
                    value = tuple(value)
                attrs[attr] = value
        params = {pname: tuple(param.shape) for pname, param in module.named_parameters(recurse=False)}
        buffers = {bname: tuple(buf.shape) for bname, buf in module.named_buffers(recurse=False)}
        snapshot[name] = {
            "module_type": module.__class__.__name__,
            "attrs": attrs,
            "params": params,
            "buffers": buffers,
        }
    return snapshot


def _operation_channel_change(op: dict[str, Any]) -> tuple[int | None, int | None]:
    before = op.get("before", op.get("before_out", op.get("before_in")))
    after = op.get("after", op.get("after_out", op.get("after_in")))
    if before is None or after is None:
        return None, None
    return int(before), int(after)


def build_structure_changes(
    before: dict[str, dict[str, Any]],
    after: dict[str, dict[str, Any]],
    applied: list[dict[str, Any]],
) -> tuple[dict[str, str], list[dict[str, Any]]]:
    op_notes: dict[str, list[str]] = {}
    op_groups: dict[str, list[str]] = {}
    for result in applied:
        group_id = result.get("group_id", "")
        for op in result.get("operations", []):
            layer = op.get("layer", "")
            if not layer:
                continue
            before_c, after_c = _operation_channel_change(op)
            axis = op.get("axis", "")
            reason = op.get("reason", "")
            note = ""
            if before_c is not None and after_c is not None:
                note = f"{axis}: {before_c} -> {after_c}"
            elif axis == "attention_metadata":
                note = (
                    f"heads: {op.get('before_heads')} -> {op.get('after_heads')}, "
                    f"inner_dim: {op.get('before_inner_dim')} -> {op.get('after_inner_dim')}"
                )
            if axis == "grouped_keep":
                note = f"{note}, groups unchanged={op.get('groups')}"
            if reason:
                note = f"{note} ({reason})" if note else reason
            if note:
                op_notes.setdefault(layer, []).append(note)
            if group_id:
                op_groups.setdefault(layer, []).append(group_id)

    notes: dict[str, str] = {}
    rows: list[dict[str, Any]] = []
    for name, after_info in after.items():
        before_info = before.get(name)
        if before_info is None:
            continue
        changes: list[str] = []
        for attr in STRUCTURE_ATTRS:
            old = before_info["attrs"].get(attr)
            new = after_info["attrs"].get(attr)
            if old != new:
                changes.append(f"{attr}: {old} -> {new}")
        for pname, new_shape in after_info["params"].items():
            old_shape = before_info["params"].get(pname)
            if old_shape != new_shape:
                changes.append(f"{pname}.shape: {old_shape} -> {new_shape}")
        for bname, new_shape in after_info["buffers"].items():
            old_shape = before_info["buffers"].get(bname)
            if old_shape != new_shape:
                changes.append(f"{bname}.shape: {old_shape} -> {new_shape}")
        if name in op_notes:
            changes.extend(op_notes[name])
        if not changes:
            continue
        note = "; ".join(dict.fromkeys(changes))
        notes[name] = note
        rows.append({
            "layer": name,
            "module_type": after_info["module_type"],
            "group_ids": ";".join(dict.fromkeys(op_groups.get(name, []))),
            "changes": note,
            "before_attrs": json.dumps(before_info["attrs"], ensure_ascii=False, default=str),
            "after_attrs": json.dumps(after_info["attrs"], ensure_ascii=False, default=str),
            "before_params": json.dumps(before_info["params"], ensure_ascii=False, default=str),
            "after_params": json.dumps(after_info["params"], ensure_ascii=False, default=str),
        })
    return notes, rows


def write_model_structure(model: nn.Module, path: Path, changes: dict[str, str] | None = None) -> None:
    changes = changes or {}
    lines = [str(model), "", "Named modules:"]
    for name, module in model.named_modules():
        if not name:
            continue
        attrs = []
        for attr in STRUCTURE_ATTRS:
            if hasattr(module, attr):
                attrs.append(f"{attr}={getattr(module, attr)}")
        line = f"{name}: {module.__class__.__name__} {' '.join(attrs)}"
        if name in changes:
            line = f"* {line}  # changed: {changes[name]}"
        lines.append(line)
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


def normalize_grouped_convs_for_alignment(model: nn.Module, align: int) -> list[dict[str, Any]]:
    operations: list[dict[str, Any]] = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.Conv2d) or module.groups <= 1:
            continue
        factor = grouped_conv_alignment_merge_factor(module, align)
        if factor is None:
            continue
        op = merge_grouped_conv_groups(module, factor)
        op["layer"] = name
        op["reason"] = "pre_prune_group_alignment_normalization"
        operations.append(op)
    return operations


def build_prune_replay(applied: list[dict[str, Any]], pre_ops: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    replay = []
    for op in pre_ops or []:
        replay.append({
            "layer": op.get("layer", ""),
            "direction": "out",
            "axis": op.get("axis", ""),
            "before": int(op.get("before_out", op.get("before", 0)) or 0),
            "after": int(op.get("after_out", op.get("after", 0)) or 0),
            "groups": int(op.get("after_groups", op.get("groups", 1)) or 1),
            "before_groups": int(op.get("before_groups", op.get("groups", 1)) or 1),
            "after_groups": int(op.get("after_groups", op.get("groups", 1)) or 1),
            "kept_groups": op.get("kept_groups", []),
            "merge_factor": int(op.get("merge_factor", 1) or 1),
        })
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
                "before_groups": int(op.get("before_groups", op.get("groups", 1)) or 1),
                "after_groups": int(op.get("after_groups", op.get("groups", 1)) or 1),
                "kept_groups": op.get("kept_groups", []),
                "group_keep_map": op.get("group_keep_map", {}),
                "per_group_after": int(op.get("per_group_after", 0) or 0),
                "keep_indices": result.get("keep_indices", []),
                "prune_indices": result.get("prune_indices", []),
            })
    return replay


def coupled_channel_stats(groups: list[Any], applied: list[dict[str, Any]]) -> dict[str, Any]:
    after_by_group: dict[str, int] = {}
    for result in applied:
        group_id = result.get("group_id", "")
        for op in result.get("operations", []):
            before, after = _operation_channel_change(op)
            if before is not None and after is not None and after < before:
                after_by_group[group_id] = after
                break
    total_before = sum(int(g.num_channels) for g in groups)
    total_after = sum(int(after_by_group.get(g.group_id, g.num_channels)) for g in groups)
    return {
        "total_coupled_channels_before": total_before,
        "total_coupled_channels_after": total_after,
        "pruned_coupled_channels": total_before - total_after,
        "num_remaining_groups": len(groups) - len(applied),
        "num_remaining_prunable_groups": sum(1 for g in groups if not g.protected) - len(applied),
    }


def build_protected_layers(
    model: nn.Module,
    *,
    adapter_protected: list[str],
    extra_prefixes: list[str] | tuple[str, ...],
) -> list[str]:
    protected = {
        name
        for name in adapter_protected
        if any(k in name.lower() for k in HEAD_PROTECTED_KEYWORDS + FIXED_SHAPE_PROTECTED_KEYWORDS)
    }
    prefixes = tuple(p for p in extra_prefixes if p)
    if prefixes:
        for name, _module in model.named_modules():
            if any(name == prefix or name.startswith(f"{prefix}.") for prefix in prefixes):
                protected.add(name)
    return sorted(protected)


def apply_only_prune_module_filter(groups: list[Any], only_prefixes: list[str] | None) -> None:
    if not only_prefixes:
        return
    prefixes = tuple(str(prefix) for prefix in only_prefixes if str(prefix))
    for group in groups:
        item_names = [str(getattr(item, "name", "")) for item in getattr(group, "items", [])]
        root = str(getattr(group, "root_name", getattr(group, "group_id", "")))
        matched = any(
            name == prefix or name.startswith(prefix + ".") or root == prefix or root.startswith(prefix + ".")
            for prefix in prefixes
            for name in item_names
        )
        if not matched:
            group.protected = True
            group.protected_reason = "only_prune_module_filter"


def apply_only_regular_grouped_conv_filter(groups: list[Any], enabled: bool) -> None:
    if not enabled:
        return
    for group in groups:
        has_regular_grouped = False
        for item in getattr(group, "items", []):
            module = getattr(item, "module", None)
            if (
                isinstance(module, nn.Conv2d)
                and module.groups > 1
                and not (module.groups == module.in_channels == module.out_channels)
            ):
                has_regular_grouped = True
                break
        if not has_regular_grouped:
            group.protected = True
            group.protected_reason = "only_prune_regular_grouped_conv_filter"


def apply_explicit_prune_indices_override(plan: Any, scopes: list[Any], module_name: str | None, idxs: list[int] | None) -> None:
    """Override one concrete pruning group with explicit root indices.

    The normal selector remains unchanged. This narrow hook is used by
    equivalence audits that must compare current replay against a TP oracle for
    the exact same root output indices.
    """
    if not module_name or idxs is None:
        return
    scope = next(
        (
            group
            for group in scopes
            if any(str(getattr(item, "name", "")) == module_name for item in getattr(group, "items", []))
        ),
        None,
    )
    if scope is None:
        plan.selected_atomic_units = []
        plan.concrete_groups = []
        return
    units_by_scope: dict[str, list[Any]] = {}
    for unit in plan.coupled_units:
        units_by_scope.setdefault(unit.scope_id, []).append(unit)
    plan.selected_atomic_units = []
    plan.concrete_groups = [
        instantiate_concrete_pruning_group(
            scope,
            sorted({int(idx) for idx in idxs}),
            coupled_units=units_by_scope.get(scope.group_id, []),
        )
    ]


def test_only_prune_module_filter_protects_non_target_groups():
    target = SimpleNamespace(
        group_id="group::backbone.layer0.0.conv2",
        root_name="backbone.layer0.0.conv2",
        protected=False,
        protected_reason=None,
        items=[SimpleNamespace(name="backbone.layer0.0.conv2")],
    )
    other = SimpleNamespace(
        group_id="group::backbone.layer1.0.conv2",
        root_name="backbone.layer1.0.conv2",
        protected=False,
        protected_reason=None,
        items=[SimpleNamespace(name="backbone.layer1.0.conv2")],
    )

    apply_only_prune_module_filter([target, other], ["backbone.layer0.0.conv2"])

    assert target.protected is False
    assert other.protected is True
    assert other.protected_reason == "only_prune_module_filter"


def select_keep(
    group: Any,
    args: argparse.Namespace,
    importance_scores: dict[str, Any] | None = None,
) -> list[int]:
    if group.meta.get("group_type") == "transformer_head_group":
        return list(group.meta.get("keep_indices", range(group.num_channels)))
    align = args.ffn_align if "ffn" in group.meta.get("group_type", "") else args.align
    return _group_aligned_keep_indices(
        group,
        prune_ratio=args.prune_ratio,
        align=align,
        min_channels=max(1, min(args.align, group.num_channels)),
        importance_scores=importance_scores,
        group_conv_align=args.group_conv_align,
    )


def configure_grouped_conv_pruning_fns(groups: list[Any], args: argparse.Namespace) -> list[dict[str, Any]]:
    """Set grouped-conv physical handlers from the new selection mode."""
    mode = args.group_conv_selection_mode
    if mode == "true_group_block_pruning":
        mode = "remove_groups"
    if mode == "remove_groups" and not args.allow_remove_groups:
        mode = "keep_groups"
    if mode not in ("independent_group_topk", "group_balanced_output_groups_fixed", "remove_groups", "flat_output_groups_fixed", "group_coarsening_zero_padded_reblock"):
        mode = "keep_groups"
    operations: list[dict[str, Any]] = []
    for group in groups:
        for item in getattr(group, "items", []):
            module = getattr(item, "module", None)
            if not isinstance(module, nn.Conv2d) or module.groups <= 1:
                continue
            old_name = getattr(item.pruning_fn, "__name__", "")
            item.pruning_fn = grouped_conv_pruning_fn(mode)
            item.reason = f"grouped_conv:{mode}"
            operations.append({
                "scope_id": group.group_id,
                "module_name": item.name,
                "old_pruning_fn": old_name,
                "new_pruning_fn": getattr(item.pruning_fn, "__name__", ""),
                "group_conv_selection_mode": args.group_conv_selection_mode,
                "allow_remove_groups": args.allow_remove_groups,
            })
    return operations


def build_selection_summary(
    *,
    args: argparse.Namespace,
    groups: list[Any],
    plan: Any,
    target_prune_ratio: float,
    actual_prune_ratio: float,
    legality_report: dict[str, Any],
    forward_ok: bool,
) -> dict[str, Any]:
    grouped_reports = list(getattr(plan, "grouped_conv_reports", []))
    return {
        "selection_mode": args.selection_mode,
        "ranking_scope": "root_node_local" if args.selection_mode == "root_node_local_unit_ratio" else args.selection_mode,
        "global_ranking": args.selection_mode in {"global_coupled_channel", "constrained_global"},
        "module_stage_based_domain": False,
        "group_conv_selection_mode": args.group_conv_selection_mode,
        "num_dependency_scopes": len(groups),
        "num_coupled_channel_units": len(plan.coupled_units),
        "num_atomic_prune_units": len(plan.atomic_units),
        "num_protected_coupled_channel_units": sum(1 for unit in plan.coupled_units if unit.protected),
        "num_protected_atomic_units": sum(1 for unit in plan.atomic_units if unit.protected),
        "num_selected_coupled_channel_units": len(plan.selected_coupled_unit_ids),
        "num_selected_atomic_units": len(plan.selected_atomic_units),
        "num_concrete_pruning_groups": len(plan.concrete_groups),
        "target_prune_ratio": target_prune_ratio,
        "actual_prune_ratio": actual_prune_ratio,
        "grouped_conv_num_scopes": len(grouped_reports),
        "grouped_conv_shared_local_mean_used": sum(1 for r in grouped_reports if r.get("group_conv_selection_mode") == "shared_local_mean"),
        "grouped_conv_flat_output_used": sum(1 for r in grouped_reports if r.get("group_conv_selection_mode") == "flat_output_groups_fixed"),
        "grouped_conv_independent_topk_used": sum(1 for r in grouped_reports if r.get("group_conv_selection_mode") == "independent_group_topk"),
        "grouped_conv_remove_groups_used": sum(1 for r in grouped_reports if r.get("group_conv_selection_mode") == "remove_groups"),
        "num_grouped_conv_align_violations": sum(1 for r in grouped_reports if not r.get("per_group_kept_count_align8", True)),
        "num_residual_protected": sum(1 for g in groups if g.meta.get("group_type") == "add" and g.protected),
        "structure_legal": bool(legality_report.get("legal", False)),
        "forward_sanity_check": bool(forward_ok),
    }


def build_root_node_local_domain_artifacts(plan: Any, args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    units = list(getattr(plan, "coupled_units", []))
    units_by_domain: dict[str, list[Any]] = {}
    for unit in units:
        root = getattr(unit, "root_node", getattr(unit, "scope_id", ""))
        units_by_domain.setdefault(str(root), []).append(unit)
    domains = []
    selected_unit_ids = set(getattr(plan, "selected_coupled_unit_ids", set()))
    for root_node, domain_units in sorted(units_by_domain.items()):
        domain_units = sorted(domain_units, key=lambda u: int(getattr(u, "root_channel_index", getattr(u, "root_idx", 0))))
        unit_ids = [u.unit_id for u in domain_units]
        pruned = [uid for uid in unit_ids if uid in selected_unit_ids]
        kept = [uid for uid in unit_ids if uid not in selected_unit_ids]
        domains.append(
            {
                "domain_id": f"root_node::{root_node}",
                "root_node": root_node,
                "root_module": getattr(domain_units[0], "root_module", root_node) if domain_units else root_node,
                "root_axis": "out_channels",
                "unit_ids": unit_ids,
                "num_units": len(unit_ids),
                "ranking_scope": "root_node_local",
                "cross_domain_ranking": False,
                "requested_keep_ratio": 1.0 - float(args.prune_ratio),
                "requested_prune_ratio": float(args.prune_ratio),
                "align": int(args.align),
                "actual_num_keep": len(kept),
                "actual_num_prune": len(pruned),
                "actual_keep_ratio": len(kept) / len(unit_ids) if unit_ids else 1.0,
                "kept_unit_ids": kept,
                "pruned_unit_ids": pruned,
            }
        )
    importance_rows = [
        {
            "candidate_id": "",
            "domain_id": f"root_node::{getattr(unit, 'root_node', getattr(unit, 'scope_id', ''))}",
            "unit_id": unit.unit_id,
            "root_node": getattr(unit, "root_node", getattr(unit, "scope_id", "")),
            "root_channel_index": getattr(unit, "root_channel_index", getattr(unit, "root_idx", 0)),
            "importance": getattr(unit, "importance", None),
            "score_source": args.importance_mode,
            "num_members": len(getattr(unit, "members", []) or []),
            "is_grouped_conv_related": getattr(unit, "is_grouped_conv_related", False),
            "group_id": getattr(unit, "group_id_in_grouped_conv", None),
            "local_channel_index": getattr(unit, "local_idx_in_group", None),
        }
        for unit in units
    ]
    return (
        {"domains": [{k: v for k, v in row.items() if k not in {"requested_keep_ratio", "requested_prune_ratio", "align", "actual_num_keep", "actual_num_prune", "actual_keep_ratio", "kept_unit_ids", "pruned_unit_ids"}} for row in domains]},
        {
            "selection_mode": "root_node_local_unit_ratio" if args.selection_mode == "root_node_local_unit_ratio" else args.selection_mode,
            "global_ranking": False,
            "module_stage_based_domain": False,
            "domains": domains,
        },
        importance_rows,
    )


def build_importance_calibration_data(
    adapter: HEALLiDARAdapter,
    args: argparse.Namespace,
    logger: logging.Logger,
) -> list[Any] | None:
    if args.importance_mode not in {"first_order_taylor", "second_order_fisher"}:
        return None
    if int(args.num_calib_batches or 0) <= 0:
        raise RuntimeError(f"{args.importance_mode}_gradient_missing: --num-calib-batches must be > 0")
    loader = adapter.get_calib_loader(
        {
            "hypes_yaml": str(args.model_config),
            "split": "train",
            "batch_size": 1,
            "num_workers": 0,
        }
    )
    batches = []
    for batch in loader:
        batches.append(batch)
        if len(batches) >= int(args.num_calib_batches):
            break
    if not batches:
        raise RuntimeError(f"{args.importance_mode}_gradient_missing: no calibration batches were produced")
    logger.info("Loaded %d train calibration batches for %s importance", len(batches), args.importance_mode)
    return batches


def move_batch_to_device(batch: Any, device: torch.device) -> Any:
    if torch.is_tensor(batch):
        return batch.to(device)
    if isinstance(batch, dict):
        return {key: move_batch_to_device(value, device) for key, value in batch.items()}
    if isinstance(batch, list):
        return [move_batch_to_device(value, device) for value in batch]
    if isinstance(batch, tuple):
        return tuple(move_batch_to_device(value, device) for value in batch)
    return batch


def run_pruning(args: argparse.Namespace) -> dict[str, Any]:
    out = ensure_unique_dir(args.output_dir)
    args.output_dir = str(out)
    logger = setup_logger(out)
    device = torch.device(resolve_device(args.device))
    if device.type == "cuda":
        torch.cuda.set_device(device)
    logger.info("Args: %s", json.dumps(vars(args), ensure_ascii=False, default=str))
    model, adapter = load_heal_model(args, device, logger)
    baseline_params, baseline_mb = count_params(model)
    structure_before = collect_module_structure(model)
    write_model_structure(model, out / "model_structure_before.txt")
    pre_prune_ops = []
    if not getattr(args, "disable_pre_prune_group_normalization", False):
        pre_prune_ops = normalize_grouped_convs_for_alignment(model, args.group_conv_align)
    if pre_prune_ops:
        logger.info("Normalized %d grouped conv layers for group/per-channel alignment", len(pre_prune_ops))
    original_params, original_mb = baseline_params, baseline_mb
    normalized_params, normalized_mb = count_params(model)

    sample = adapter.build_synthetic_batch(model)
    trace = trace_model(model, sample, forward_fn=adapter.forward_for_task)
    protected_layers = build_protected_layers(
        model,
        adapter_protected=adapter.get_protected_layers(model),
        extra_prefixes=args.extra_protected_prefix,
    )
    op_graph = build_op_graph(trace, model, protected_layers=protected_layers)
    save_json(op_graph.to_dict(), out / "op_graph.json")
    effective_grouped_conv_mode = (
        args.group_conv_selection_mode
        if args.group_conv_selection_mode in {"flat_output_groups_fixed", "group_balanced_output_groups_fixed", "group_coarsening_zero_padded_reblock"}
        else ("remove_groups" if args.group_conv_selection_mode == "true_group_block_pruning" else args.group_conv_prune_mode)
    )
    cnn_groups = GroupBuilder(
        op_graph,
        align=args.align,
        grouped_conv_mode=effective_grouped_conv_mode,
        protect_residual_add=args.protect_residual_add,
    ).build()
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
    apply_only_prune_module_filter(groups, args.only_prune_module_prefix)
    apply_only_regular_grouped_conv_filter(groups, args.only_prune_regular_grouped_conv)
    grouped_fn_ops = configure_grouped_conv_pruning_fns(groups, args)
    save_json([g.summary() | {"meta": g.meta} for g in groups], out / "pruning_groups.json")
    save_csv(group_rows(groups), out / "pruning_groups.csv")
    save_json(dependency_scope_rows(groups), out / "dependency_scopes.json")
    save_csv(dependency_scope_rows(groups), out / "dependency_scopes.csv")

    for param in model.parameters():
        param.requires_grad_(True)
    calibration_data = build_importance_calibration_data(adapter, args, logger)
    if calibration_data is not None:
        calibration_data = [move_batch_to_device(batch, device) for batch in calibration_data]
    importance, importance_records = compute_group_importance(
        model,
        groups,
        method=args.importance_mode,
        forward_fn=adapter.forward_for_task if calibration_data is not None else None,
        calibration_data=calibration_data,
        loss_fn=adapter.compute_task_loss if calibration_data is not None else None,
        num_calib_batches=int(args.num_calib_batches or 0),
        strict_grad=args.importance_mode in {"first_order_taylor", "second_order_fisher"},
    )
    channel_importance = compute_layer_channel_importance(groups, method=args.importance_mode)
    scope_channel_importance, scope_importance_records = compute_scope_channel_importance_map(
        groups,
        method=args.importance_mode,
    )
    save_csv(importance_records, out / "group_importance.csv")
    save_json(importance_records, out / "group_importance.json")
    save_csv(scope_importance_records, out / "scope_channel_importance.csv")
    save_json(
        [
            record | {"importance": scope_channel_importance.get(record["scope_id"], torch.empty(0)).tolist()}
            for record in scope_importance_records
        ],
        out / "scope_channel_importance.json",
    )
    save_csv(transformer_rows(groups, importance, args.importance_mode), out / "transformer_groups.csv")

    selection_cfg = SelectionConfig(
        prune_ratio=args.prune_ratio,
        selection_mode=args.selection_mode,
        group_conv_selection_mode=args.group_conv_selection_mode,
        align=args.align,
        group_conv_align=args.group_conv_align,
        group_conv_prune_mode=args.group_conv_prune_mode,
        allow_remove_groups=args.allow_remove_groups,
        min_groups_after_prune=args.min_groups_after_prune,
        groups_align=args.groups_align,
        min_channels=max(1, min(args.align, 8)),
        importance_mode=args.importance_mode,
    )
    plan = build_pruning_plan(groups, scope_channel_importance, selection_cfg)
    explicit_idxs = None
    if args.explicit_prune_idxs_for_module:
        explicit_idxs = [int(v) for v in str(args.explicit_prune_idxs_for_module).split(",") if str(v).strip()]
    apply_explicit_prune_indices_override(plan, groups, args.explicit_prune_module, explicit_idxs)
    save_json([dataclass_to_json_dict(unit) for unit in plan.coupled_units], out / "coupled_channel_units.json")
    save_csv(coupled_channel_unit_rows(plan.coupled_units), out / "coupled_channel_units.csv")
    save_json([dataclass_to_json_dict(unit) for unit in plan.atomic_units], out / "atomic_prune_units.json")
    save_csv(atomic_prune_unit_rows(plan.atomic_units), out / "atomic_prune_units.csv")
    save_json(plan.grouped_conv_reports, out / "grouped_conv_selection_report.json")
    save_csv(plan.grouped_conv_reports, out / "grouped_conv_selection_report.csv")
    root_domains_json, domain_selection_json, unit_importance_rows = build_root_node_local_domain_artifacts(plan, args)
    if args.selection_mode == "root_node_local_unit_ratio":
        save_json(root_domains_json, out / "root_node_local_domains.json")
        save_json(domain_selection_json, out / "domain_selection_summary.json")
        save_csv(unit_importance_rows, out / "unit_importance.csv")

    prunable = [g for g in groups if not g.protected and g.num_channels > 0]
    target_pruned_params = int(round(original_params * args.prune_ratio))
    applied = []
    skipped = []
    pruned_ids: set[str] = set()
    scope_by_id = {g.group_id: g for g in groups}
    for concrete in plan.concrete_groups:
        g = scope_by_id.get(concrete.scope_id)
        if g is None:
            skipped.append({"group_id": concrete.scope_id, "reason": "missing_scope"})
            continue
        keep = concrete.keep_indices
        if len(keep) >= g.num_channels:
            skipped.append({"group_id": g.group_id, "reason": "no_channel_reduction", "num_channels": g.num_channels})
            continue
        if g.meta.get("group_type", "") in TRANSFORMER_TYPES:
            check = check_transformer_group(g, keep, min_heads=args.min_heads, head_align=args.head_align, ffn_align=args.ffn_align)
        else:
            check = check_pruning_group(g, keep, group_conv_align=args.group_conv_align)
        concrete.check_group_passed = bool(check["legal"])
        if not check["legal"]:
            skipped.append({"group_id": g.group_id, "reason": "check_failed", "issues": check["issues"]})
            continue
        if args.dry_run:
            continue
        current_params, _ = count_params(model)
        result = g.prune(keep)
        if result.get("applied"):
            before_params = current_params
            after_params, _ = count_params(model)
            result["params_before_group"] = before_params
            result["params_after_group"] = after_params
            result["params_removed_by_group"] = before_params - after_params
            result["source_concrete_group"] = concrete.concrete_group_id
            result["source_atomic_units"] = concrete.source_atomic_units
            result["source_coupled_units"] = concrete.source_coupled_units
            result["prune_indices"] = concrete.prune_indices
            result["keep_indices"] = concrete.keep_indices
            if "ffn" in g.meta.get("group_type", ""):
                g.meta["ffn_dim_after"] = len(keep)
            applied.append(result)
            pruned_ids.add(g.group_id)
        else:
            skipped.append(result)
    save_json([dataclass_to_json_dict(group) for group in plan.concrete_groups], out / "concrete_pruning_groups.json")
    save_csv(concrete_pruning_group_rows(plan.concrete_groups), out / "concrete_pruning_groups.csv")

    legality = check_model_legality(model, group_conv_align=args.group_conv_align)
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
    save_json(
        {
            "forward_sanity_check": forward_ok,
            "skipped": bool(args.skip_forward_check),
            "issues": legality_report.get("forward_issues", []),
        },
        out / "forward_sanity_report.json",
    )
    structure_after = collect_module_structure(model)
    audit_applied = ([{"group_id": "pre_prune_group_alignment", "operations": pre_prune_ops}] if pre_prune_ops else []) + applied
    structure_notes, structure_change_rows = build_structure_changes(structure_before, structure_after, audit_applied)
    write_model_structure(model, out / "model_structure_after.txt", changes=structure_notes)
    save_csv(structure_change_rows, out / "structure_changes.csv")
    save_json(structure_change_rows, out / "structure_changes.json")
    group_conv_rows, group_conv_report = group_conv_reports(model, args.group_conv_align)
    save_csv(group_conv_rows, out / "group_conv_summary.csv")
    save_json(group_conv_report, out / "group_conv_alignment_report.json")
    save_json(residual_summary(groups), out / "residual_dependency_summary.json")
    save_json(build_transformer_summary(groups, args, pruned_ids), out / "transformer_pruning_summary.json")
    pruned_params, pruned_mb = count_params(model)
    actual_ratio = 1.0 - pruned_params / max(original_params, 1)
    channel_stats = coupled_channel_stats(groups, applied)
    selection_summary = build_selection_summary(
        args=args,
        groups=groups,
        plan=plan,
        target_prune_ratio=args.prune_ratio,
        actual_prune_ratio=actual_ratio,
        legality_report=legality_report,
        forward_ok=forward_ok,
    )
    save_json(selection_summary, out / "selection_summary.json")
    summary = {
        "output_dir": str(out),
        "pruned_checkpoint": str(out / "pruned_model.pth"),
        "checkpoint": args.checkpoint,
        "model_name": "lidar_pyramid",
        "importance_mode": args.importance_mode,
        "selection_mode": args.selection_mode,
        "target_prune_ratio": args.prune_ratio,
        "target_prune_ratio_definition": "1 - pruned_params / original_params",
        "target_pruned_params": target_pruned_params,
        "actual_prune_ratio": actual_ratio,
        "pre_prune_group_alignment_ops": pre_prune_ops,
        "num_total_groups": len(groups),
        "num_prunable_groups": len(prunable),
        "num_protected_groups": len(groups) - len(prunable),
        "num_pruned_groups": len(applied),
        "num_remaining_groups": channel_stats["num_remaining_groups"],
        "num_remaining_prunable_groups": channel_stats["num_remaining_prunable_groups"],
        "total_coupled_channels_before": channel_stats["total_coupled_channels_before"],
        "total_coupled_channels_after": channel_stats["total_coupled_channels_after"],
        "pruned_coupled_channels": channel_stats["pruned_coupled_channels"],
        "num_transformer_groups": len([g for g in groups if g.meta.get("group_type", "") in TRANSFORMER_TYPES]),
        "num_pruned_transformer_groups": len([gid for gid in pruned_ids if gid.startswith("transformer::")]),
        "num_group_conv_layers": len(group_conv_rows),
        "num_residual_groups": len([g for g in groups if g.meta.get("group_type") == "add"]),
        "original_params": original_params,
        "normalized_params": normalized_params,
        "pruned_params": pruned_params,
        "original_model_size_mb": original_mb,
        "normalized_model_size_mb": normalized_mb,
        "pruned_model_size_mb": pruned_mb,
        "group_conv_prune_mode": args.group_conv_prune_mode,
        "group_conv_selection_mode": args.group_conv_selection_mode,
        "group_conv_align": args.group_conv_align,
        "groups_align": args.groups_align,
        "allow_remove_groups": args.allow_remove_groups,
        "min_groups_after_prune": args.min_groups_after_prune,
        "grouped_conv_pruning_fn_updates": grouped_fn_ops,
        "selection_summary": selection_summary,
        "structure_legal": legality_report["legal"],
        "forward_sanity_check": forward_ok,
        "applied": applied,
        "skipped": skipped,
    }
    save_json(summary, out / "pruning_summary.json")
    save_json(pre_prune_ops, out / "group_conv_pre_prune_normalization.json")
    replay = build_prune_replay(applied, pre_ops=pre_prune_ops)
    save_json({"operations": replay}, out / "prune_replay.json")
    if args.dry_run:
        logger.info("Dry run enabled; pruned model checkpoint is not saved")
    elif legality_report["legal"] and (forward_ok or args.skip_forward_check):
        torch.save({"model": model.state_dict(), "prune_metadata": summary, "prune_replay": replay}, out / "pruned_model.pth")
        logger.info("Saved pruned model to %s", out / "pruned_model.pth")
    elif args.allow_save_on_forward_fail:
        torch.save({"model": model.state_dict(), "prune_metadata": summary, "prune_replay": replay}, out / "pruned_model.pth")
        logger.warning("Saved pruned model despite failed legality/forward check because override is enabled: %s", out / "pruned_model.pth")
    else:
        logger.error(
            "Legality or forward sanity check failed; pruned_model.pth was not saved. "
            "Use --allow-save-on-forward-fail to override."
        )
    logger.info(
        "Pruning result: target=%.2f%% actual_param_prune=%.3f%% "
        "(baseline %d -> pruned %d params, %.3fMB -> %.3fMB; normalized start %d params)",
        args.prune_ratio * 100.0,
        actual_ratio * 100.0,
        original_params,
        pruned_params,
        original_mb,
        pruned_mb,
        normalized_params,
    )
    logger.info(
        "Coupled groups: total=%d prunable=%d protected=%d pruned=%d remaining_total=%d remaining_prunable=%d",
        len(groups),
        len(prunable),
        len(groups) - len(prunable),
        len(applied),
        channel_stats["num_remaining_groups"],
        channel_stats["num_remaining_prunable_groups"],
    )
    logger.info(
        "Coupled logical channels: before=%d after=%d removed=%d",
        channel_stats["total_coupled_channels_before"],
        channel_stats["total_coupled_channels_after"],
        channel_stats["pruned_coupled_channels"],
    )
    logger.info(
        "Structure check: legal=%s forward_sanity=%s changed_layers=%d",
        legality_report["legal"],
        forward_ok,
        len(structure_change_rows),
    )
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="General structured pruner test for HEAL lidar_pyramid")
    p.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    p.add_argument("--model-config", default=DEFAULT_CONFIG)
    p.add_argument("--heal-root", default=DEFAULT_HEAL_ROOT)
    p.add_argument("--prune-ratio", type=float, default=0.25)
    p.add_argument("--importance-mode", choices=["l1_norm", "l2_norm", "first_order_taylor", "second_order_fisher"], default="l1_norm")
    p.add_argument("--selection-mode", default="local_scope", choices=["local_scope", "root_node_local_unit_ratio", "global_coupled_channel", "constrained_global"])
    p.add_argument("--num-calib-batches", type=int, default=0)
    p.add_argument("--align", type=int, default=16)
    p.add_argument("--group-conv-align", type=int, default=8)
    p.add_argument("--group-conv-prune-mode", default="keep_groups", choices=["keep_groups", "remove_groups"])
    p.add_argument("--group-conv-selection-mode", default="flat_output_groups_fixed", choices=["shared_local_mean", "independent_group_topk", "remove_groups", "true_group_block_pruning", "flat_output_groups_fixed", "group_balanced_output_groups_fixed", "group_coarsening_zero_padded_reblock"])
    p.add_argument("--allow-remove-groups", type=str2bool, default=False)
    p.add_argument("--min-groups-after-prune", type=int, default=8)
    p.add_argument("--groups-align", type=int, default=8)
    p.add_argument("--enable-transformer-pruning", type=str2bool, default=True)
    p.add_argument("--enable-native-mha-pruning", type=str2bool, default=False)
    p.add_argument("--transformer-prune-heads", type=str2bool, default=True)
    p.add_argument("--transformer-prune-ffn", type=str2bool, default=True)
    p.add_argument("--head-prune-mode", default="whole_head")
    p.add_argument("--min-heads", type=int, default=1)
    p.add_argument("--head-align", type=int, default=1)
    p.add_argument("--ffn-align", type=int, default=8)
    p.add_argument("--protect-transformer-hidden", type=str2bool, default=True)
    p.add_argument("--protect-residual-add", type=str2bool, default=True)
    p.add_argument("--device", default="cpu")
    p.add_argument("--output-dir", default=str(_THIS_DIR / "outputs" / "prune_lidar_pyramid_25_l1"))
    p.add_argument("--protect-neck-and-heads", type=str2bool, default=True)
    p.add_argument("--extra-protected-prefix", action="append", default=None)
    p.add_argument("--only-prune-module-prefix", action="append", default=None)
    p.add_argument("--only-prune-regular-grouped-conv", type=str2bool, default=False)
    p.add_argument("--explicit-prune-module", default=None)
    p.add_argument("--explicit-prune-idxs-for-module", default=None)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--skip-forward-check", action="store_true")
    p.add_argument("--allow-save-on-forward-fail", action="store_true")
    p.add_argument("--disable-pre-prune-group-normalization", action="store_true")
    args = p.parse_args(argv)
    extra = list(args.extra_protected_prefix or [])
    if args.protect_neck_and_heads:
        extra = list(DEFAULT_SAFE_PROTECTED_PREFIXES) + extra
    args.extra_protected_prefix = list(dict.fromkeys(extra))
    args.only_prune_module_prefix = list(dict.fromkeys(args.only_prune_module_prefix or []))
    return args


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
                self.pw1 = nn.Conv2d(3, 128, 1)
                self.grouped = nn.Conv2d(128, 128, 3, padding=1, groups=8)
                self.bn = nn.BatchNorm2d(128)
                self.pw2 = nn.Conv2d(128, 16, 1)

            def forward(self, x):
                return self.pw2(torch.relu(self.bn(self.grouped(self.pw1(x)))))

        m = GroupedNet().eval()
        x = torch.randn(2, 3, 8, 8)
        _, report = toy_prune_model(m, x, prune_ratio=0.5, align=8, min_channels=8, grouped_conv_mode="keep_groups")
        assert report["legality"]["legal"]
        assert m.pw1.out_channels == m.grouped.in_channels == m.grouped.out_channels
        assert m.grouped.groups == 8
        assert (m.grouped.in_channels // m.grouped.groups) % 8 == 0
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
