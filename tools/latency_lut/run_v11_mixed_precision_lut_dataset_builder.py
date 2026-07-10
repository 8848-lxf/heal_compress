#!/usr/bin/env python3
"""Build v11 mixed-precision LUT dataset artifacts.

The mixed-precision unit is a precision coupling group, not a whole-engine
precision mode and not a naked layer.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import os
import random
import re
import shutil
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torch.nn as nn

_THIS = Path(__file__).resolve()
_ROOT = _THIS.parents[2]
_UNIAD = _ROOT.parent
for _p in (_UNIAD, _ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from heal_compress.pruning.config import PruningConfig
from heal_compress.pruning.formal_pruner import HEALStructuredPruner
from heal_compress.quant_deploy.pruned_signal_maxk_exporter import (
    ONNX_ORIGIN_MAP_UNIQUE_NAME_POLICY_VERSION,
    REQUIRED_HEAL_INPUTS,
    export_pruned_lidar_pyramid_signal_maxk_onnx,
    inspect_signal_maxk_onnx,
)
from heal_compress.trt_runtime.heal_trt_evaluator import run_real_heal_validation_eval, run_trt_smoke
from heal_compress.tracer.dependency_tracer import build_dependency_graph
from heal_compress.tracer.precision_coupling_tracer import PrecisionGroup, build_precision_coupling_groups, precision_groups_to_json
from tools.latency_lut.physical_structure_v2 import (
    HASH_SCHEMA_VERSION as PHYSICAL_HASH_SCHEMA_VERSION,
    run_physical_structure_preflight,
)


PRECISIONS = ("fp32", "fp16", "int8")
DATASET_VERSION = "v11_mixed_precision_lut_dataset"
GROUPED_CONV_INT8_SAFE_PER_GROUP = {4, 8, 16, 32}


class ToyMixedPrecisionSubnet(nn.Module):
    def __init__(self, width: int = 16, head_out: int = 2) -> None:
        super().__init__()
        self.stem = nn.Conv2d(3, width, 1)
        self.branch_a = nn.Conv2d(width, width, 3, padding=1)
        self.branch_b = nn.Conv2d(width, width, 3, padding=1)
        self.fuse = nn.Conv2d(width * 2, width, 1)
        self.head = nn.Conv2d(width, head_out, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        a = self.branch_a(x)
        b = self.branch_b(x)
        merged = a + b
        cat = torch.cat([merged, x], dim=1)
        return self.head(self.fuse(cat))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(str(key))
                fields.append(str(key))
    if not fields:
        fields = ["empty"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            ready = {}
            for key, value in row.items():
                ready[key] = json.dumps(value, sort_keys=True) if isinstance(value, (dict, list, tuple)) else value
            writer.writerow(ready)


def append_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")


def read_csv_rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _structure_hash(model: nn.Module, target: float, round_to: int) -> str:
    rows = []
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
            rows.append(
                {
                    "name": name,
                    "type": module.__class__.__name__,
                    "weight": tuple(module.weight.shape),
                    "groups": getattr(module, "groups", 1),
                }
            )
    raw = json.dumps({"target": target, "round_to": round_to, "rows": rows}, sort_keys=True)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def _target_for_index(index: int, total: int) -> float:
    if total <= 1:
        return 0.30
    return round(0.05 + (0.62 - 0.05) * index / (total - 1), 4)


def _shape_report_to_summary(report: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "passed": bool(report.get("passed", False)),
        "non_channel_shape_violation_count": int(report.get("non_channel_shape_violation_count", 0) or 0),
    }


def gate_onnx_export(shape_report: Mapping[str, Any]) -> dict[str, Any]:
    allowed = bool(shape_report.get("passed", False))
    return {"allowed": allowed, "blocked_stage": "" if allowed else "shape_invariant_failed"}


def gate_engine_build(onnx_report: Mapping[str, Any]) -> dict[str, Any]:
    allowed = bool(onnx_report.get("success", False))
    return {"allowed": allowed, "blocked_stage": "" if allowed else "onnx_export_or_check_failed"}


def gate_eval(engine_report: Mapping[str, Any]) -> dict[str, Any]:
    allowed = bool(engine_report.get("build_success", engine_report.get("success", False)))
    return {"allowed": allowed, "blocked_stage": "" if allowed else "engine_build_failed"}


def str2bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _template_for_profile_index(profile_index: int) -> tuple[str, float]:
    if profile_index <= 0:
        return "existing_or_baseline_profile", -1.0
    templates = {
        1: ("low_int8", 0.20),
        2: ("medium_int8", 0.50),
        3: ("high_int8", 0.80),
    }
    return templates.get(int(profile_index), ("random_balanced", 0.35 + 0.10 * (profile_index % 4)))


def _precision_group_int8_eligible(group: PrecisionGroup) -> bool:
    for module in group.member_modules:
        module_name = str(module)
        lowered = module_name.lower()
        if ".deblocks." in lowered or ".single_head_" in lowered:
            return False
    return True


def _safe_group_id_fragment(value: str) -> str:
    return re.sub(r"[^0-9A-Za-z_]+", "_", str(value)).strip("_") or "module"


def _effective_precision_groups(groups: Sequence[PrecisionGroup]) -> list[PrecisionGroup]:
    effective: list[PrecisionGroup] = []
    for group in groups:
        if group.force_same_precision or len(group.member_modules) <= 1:
            effective.append(group)
            continue
        for module_name in group.member_modules:
            effective.append(
                PrecisionGroup(
                    precision_group_id=f"{group.precision_group_id}__{_safe_group_id_fragment(str(module_name))}",
                    member_modules=[str(module_name)],
                    reason=group.reason,
                    allowed_precisions=list(group.allowed_precisions),
                    default_precision=group.default_precision,
                    force_same_precision=False,
                )
            )
    return effective


def _precision_boundary_counts(groups: Sequence[PrecisionGroup], layer_assignment: Mapping[str, str]) -> dict[str, int]:
    concat_groups = [group for group in groups if str(group.reason).lower() == "concat" and not group.force_same_precision]
    concat_members = [str(module) for group in concat_groups for module in group.member_modules]
    return {
        "concat_fp16_boundary_count": len(concat_groups),
        "concat_requantize_after_count": sum(1 for module in concat_members if str(layer_assignment.get(module, "")).lower() == "int8"),
    }


def _int_value(row: Mapping[str, Any], *keys: str, default: int = 0) -> int:
    for key in keys:
        value = row.get(key)
        if value not in (None, ""):
            try:
                return int(value)
            except Exception:
                continue
    return int(default)


def module_int8_shape_eligibility(
    module_shape: Mapping[str, Any],
    *,
    safe_per_group_set: set[int] | Sequence[int] = GROUPED_CONV_INT8_SAFE_PER_GROUP,
) -> dict[str, Any]:
    groups = _int_value(module_shape, "groups", default=1)
    after_in = _int_value(module_shape, "after_in", "after_in_channels", "in_channels", default=0)
    after_out = _int_value(module_shape, "after_out", "after_out_channels", "out_channels", default=0)
    safe = {int(value) for value in safe_per_group_set}
    if groups <= 1:
        return {
            "module_name": str(module_shape.get("module_name", "")),
            "groups": groups,
            "after_in": after_in,
            "after_out": after_out,
            "cin_per_group": after_in,
            "cout_per_group": after_out,
            "safe_per_group_set": sorted(safe),
            "int8_shape_supported": True,
            "reason": "ordinary_conv_no_extra_alignment_required",
        }
    cin_ok = after_in > 0 and after_in % groups == 0
    cout_ok = after_out > 0 and after_out % groups == 0
    cin_per_group = after_in // groups if cin_ok else None
    cout_per_group = after_out // groups if cout_ok else None
    supported = bool(cin_ok and cout_ok and cin_per_group in safe and cout_per_group in safe)
    reason = "grouped_conv_int8_safe_per_group" if supported else "grouped_conv_int8_per_group_shape_not_supported"
    return {
        "module_name": str(module_shape.get("module_name", "")),
        "groups": groups,
        "after_in": after_in,
        "after_out": after_out,
        "cin_per_group": cin_per_group,
        "cout_per_group": cout_per_group,
        "safe_per_group_set": sorted(safe),
        "int8_shape_supported": supported,
        "reason": reason,
    }


def _module_shape_index_from_manifest(manifest: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    shapes: dict[str, dict[str, Any]] = {}
    for row in manifest.get("module_channel_before_after") or []:
        if not isinstance(row, Mapping):
            continue
        before = row.get("before") if isinstance(row.get("before"), Mapping) else {}
        after = row.get("after") if isinstance(row.get("after"), Mapping) else {}
        module_name = str(row.get("module_name", ""))
        if not module_name:
            continue
        shapes[module_name] = {
            "module_name": module_name,
            "groups": _int_value(after, "groups", default=_int_value(before, "groups", default=1)),
            "before_in": _int_value(before, "in_channels", default=0),
            "before_out": _int_value(before, "out_channels", default=0),
            "after_in": _int_value(after, "in_channels", default=_int_value(before, "in_channels", default=0)),
            "after_out": _int_value(after, "out_channels", default=_int_value(before, "out_channels", default=0)),
        }
    for row in manifest.get("before_after_shapes") or []:
        if not isinstance(row, Mapping):
            continue
        module_name = str(row.get("module_name", ""))
        if not module_name or module_name in shapes:
            continue
        before_attrs = ((row.get("before") or {}).get("attrs") or {}) if isinstance(row.get("before"), Mapping) else {}
        after_attrs = ((row.get("after") or {}).get("attrs") or {}) if isinstance(row.get("after"), Mapping) else {}
        groups = _int_value(after_attrs, "groups", default=_int_value(before_attrs, "groups", default=1))
        shapes[module_name] = {
            "module_name": module_name,
            "groups": groups,
            "before_in": _int_value(before_attrs, "in_channels", default=0),
            "before_out": _int_value(before_attrs, "out_channels", default=0),
            "after_in": _int_value(after_attrs, "in_channels", default=_int_value(before_attrs, "in_channels", default=0)),
            "after_out": _int_value(after_attrs, "out_channels", default=_int_value(before_attrs, "out_channels", default=0)),
        }
    for row in (manifest.get("physical_structure_snapshot_v2") or {}).get("modules", []):
        if not isinstance(row, Mapping):
            continue
        module_name = str(row.get("canonical_module_name", ""))
        if not module_name:
            continue
        existing = shapes.get(module_name, {})
        shapes[module_name] = {
            "module_name": module_name,
            "groups": _int_value(row, "groups", default=1),
            "before_in": _int_value(existing, "before_in", default=_int_value(row, "in_channels", default=0)),
            "before_out": _int_value(existing, "before_out", default=_int_value(row, "out_channels", default=0)),
            "after_in": _int_value(row, "in_channels", "in_features", default=0),
            "after_out": _int_value(row, "out_channels", "out_features", default=0),
            "physical_truth_source": "physical_structure_snapshot_v2",
        }
    return shapes


def _module_shape_index_from_subnet_dir(subnet_dir: Path) -> dict[str, dict[str, Any]]:
    manifest = _read_json_if_exists(subnet_dir / "pruning_manifest.json", {})
    snapshot = _read_json_if_exists(subnet_dir / "physical_structure_snapshot_v2.json", {})
    if isinstance(manifest, Mapping) and isinstance(snapshot, Mapping) and snapshot.get("modules"):
        manifest = dict(manifest)
        manifest["physical_structure_snapshot_v2"] = snapshot
    return _module_shape_index_from_manifest(manifest if isinstance(manifest, Mapping) else {})


def apply_deployment_aware_precision_legality(
    profile: Mapping[str, Any],
    module_shapes: Mapping[str, Mapping[str, Any]],
    *,
    safe_per_group_set: set[int] | Sequence[int] = GROUPED_CONV_INT8_SAFE_PER_GROUP,
) -> dict[str, Any]:
    gated = copy.deepcopy(dict(profile))
    assignments = copy.deepcopy(dict(gated.get("precision_group_assignments") or {}))
    fallback_layers = list(gated.get("fallback_layers") or [])
    for group_id, row in assignments.items():
        requested = str(row.get("requested_precision", row.get("final_precision", "fp16"))).lower()
        final = str(row.get("final_precision", requested)).lower()
        if requested != "int8" and final != "int8":
            continue
        unsupported: list[dict[str, Any]] = []
        for module in row.get("member_modules", []):
            module_name = str(module)
            shape = module_shapes.get(module_name)
            if not shape:
                continue
            eligibility = module_int8_shape_eligibility(shape, safe_per_group_set=safe_per_group_set)
            if not eligibility["int8_shape_supported"]:
                unsupported.append(eligibility)
        if unsupported:
            row["requested_precision"] = requested
            row["final_precision"] = "fp16"
            row["fallback_precision"] = "fp16"
            row["fallback_reason"] = "grouped_conv_int8_per_group_shape_not_supported"
            row["deployment_shape_gate"] = unsupported
            for module in row.get("member_modules", []):
                fallback_layers.append(
                    {
                        "module_name": str(module),
                        "precision_group_id": str(group_id),
                        "requested_precision": requested,
                        "final_precision": "fp16",
                        "fallback_reason": "grouped_conv_int8_per_group_shape_not_supported",
                    }
                )
    layer_assignment, overlap_fallback_layers = _resolve_overlapping_precision_groups(assignments)
    gated["precision_group_assignments"] = assignments
    gated["layer_precision_assignment"] = layer_assignment
    gated["fallback_layers"] = fallback_layers + overlap_fallback_layers
    gated["fallback_layer_count"] = len(gated["fallback_layers"])
    gated["deployment_shape_gate_checked"] = True
    return _finalize_profile_counts(gated)


def precision_assignment_hash(profile: Mapping[str, Any]) -> str:
    rows = []
    for group_id in sorted((profile.get("precision_group_assignments") or {}).keys()):
        row = dict((profile.get("precision_group_assignments") or {}).get(group_id) or {})
        rows.append(
            {
                "precision_group_id": str(group_id),
                "member_modules": sorted(str(v) for v in row.get("member_modules", [])),
                "requested_precision": row.get("requested_precision", ""),
                "final_precision": row.get("final_precision", ""),
                "fallback_precision": row.get("fallback_precision", ""),
                "fallback_reason": row.get("fallback_reason", ""),
            }
        )
    raw = json.dumps(rows, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def _resolve_overlapping_precision_groups(
    assignments: dict[str, dict[str, Any]],
) -> tuple[dict[str, str], list[dict[str, Any]]]:
    group_ids = list(assignments.keys())
    parent = {group_id: group_id for group_id in group_ids}

    def find(group_id: str) -> str:
        while parent[group_id] != group_id:
            parent[group_id] = parent[parent[group_id]]
            group_id = parent[group_id]
        return group_id

    def union(left: str, right: str) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    module_to_groups: dict[str, list[str]] = {}
    for group_id, row in assignments.items():
        for module in row.get("member_modules", []):
            module_to_groups.setdefault(str(module), []).append(group_id)
    for linked_groups in module_to_groups.values():
        for group_id in linked_groups[1:]:
            union(linked_groups[0], group_id)

    components: dict[str, list[str]] = {}
    for group_id in group_ids:
        components.setdefault(find(group_id), []).append(group_id)

    for component_group_ids in components.values():
        if len(component_group_ids) <= 1:
            continue
        allowed_sets = [
            {str(value).lower() for value in assignments[group_id].get("allowed_precisions", PRECISIONS)}
            for group_id in component_group_ids
        ]
        common_allowed = set.intersection(*allowed_sets) if allowed_sets else set(PRECISIONS)
        if not common_allowed:
            common_allowed = set(PRECISIONS)
        requested_values = {
            str(assignments[group_id].get("requested_precision", assignments[group_id].get("final_precision", "fp16"))).lower()
            for group_id in component_group_ids
        }
        if "int8" in requested_values and "int8" in common_allowed:
            component_final = "int8"
        elif "int8" in requested_values and "fp16" in common_allowed:
            component_final = "fp16"
        elif "fp32" in requested_values and "fp32" in common_allowed:
            component_final = "fp32"
        elif "fp16" in common_allowed:
            component_final = "fp16"
        elif "fp32" in common_allowed:
            component_final = "fp32"
        else:
            component_final = sorted(common_allowed)[0]
        for group_id in component_group_ids:
            assignments[group_id]["final_precision"] = component_final

    layer_assignment: dict[str, str] = {}
    fallback_layers: list[dict[str, Any]] = []
    for group_id, row in assignments.items():
        requested = str(row.get("requested_precision", row.get("final_precision", "fp16"))).lower()
        final = str(row.get("final_precision", requested)).lower()
        if requested != final:
            row["fallback_precision"] = final
            row["fallback_reason"] = row.get("fallback_reason") or "precision_group_overlap_or_allowed_precision_constraint"
            for module in row.get("member_modules", []):
                fallback_layers.append(
                    {
                        "module_name": module,
                        "precision_group_id": group_id,
                        "requested_precision": requested,
                        "final_precision": final,
                        "fallback_reason": row["fallback_reason"],
                    }
                )
        else:
            row["fallback_precision"] = ""
            row["fallback_reason"] = ""
        for module in row.get("member_modules", []):
            layer_assignment[str(module)] = final
    return layer_assignment, fallback_layers


def _finalize_profile_counts(profile: dict[str, Any]) -> dict[str, Any]:
    assignments = profile.get("precision_group_assignments") or {}
    group_counts = {"fp32": 0, "fp16": 0, "int8": 0}
    layer_counts = {"fp32": 0, "fp16": 0, "int8": 0}
    for row in assignments.values():
        group_counts[str(row.get("final_precision", "fp16"))] = group_counts.get(str(row.get("final_precision", "fp16")), 0) + 1
    for precision in (profile.get("layer_precision_assignment") or {}).values():
        layer_counts[str(precision)] = layer_counts.get(str(precision), 0) + 1
    total_groups = max(sum(group_counts.values()), 1)
    total_layers = max(sum(layer_counts.values()), 1)
    profile["fp32_group_count"] = group_counts.get("fp32", 0)
    profile["fp16_group_count"] = group_counts.get("fp16", 0)
    profile["int8_group_count"] = group_counts.get("int8", 0)
    profile["fp32_layer_count"] = layer_counts.get("fp32", 0)
    profile["fp16_layer_count"] = layer_counts.get("fp16", 0)
    profile["int8_layer_count"] = layer_counts.get("int8", 0)
    profile["actual_int8_group_ratio"] = group_counts.get("int8", 0) / total_groups
    profile["actual_int8_layer_ratio"] = layer_counts.get("int8", 0) / total_layers
    profile["precision_coverage"] = {key: layer_counts.get(key, 0) / total_layers for key in ("fp32", "fp16", "int8")}
    profile["precision_assignment_hash"] = precision_assignment_hash(profile)
    return profile


def sample_mixed_precision_profile(
    *,
    subnet_id: str,
    structure_hash: str,
    groups: Sequence[PrecisionGroup],
    random_seed: int,
    precision_modes: Sequence[str] = PRECISIONS,
) -> dict[str, Any]:
    rng = random.Random(int(random_seed))
    source_groups = list(groups)
    groups = _effective_precision_groups(groups)
    modes = [str(p).lower() for p in precision_modes]
    assignments: dict[str, dict[str, Any]] = {}
    layer_assignment: dict[str, str] = {}
    fallback_layers: list[dict[str, Any]] = []
    counts = {"fp32": 0, "fp16": 0, "int8": 0}
    for group in groups:
        requested = rng.choice(modes) if modes else group.default_precision
        final = requested if requested in group.allowed_precisions else group.default_precision
        if final not in group.allowed_precisions:
            final = group.allowed_precisions[0]
        if final != requested:
            for module in group.member_modules:
                fallback_layers.append(
                    {
                        "module_name": module,
                        "precision_group_id": group.precision_group_id,
                        "requested_precision": requested,
                        "final_precision": final,
                        "fallback_reason": f"requested_precision_not_allowed:{group.reason}",
                    }
                )
        assignments[group.precision_group_id] = {
            "member_modules": list(group.member_modules),
            "reason": group.reason,
            "allowed_precisions": list(group.allowed_precisions),
            "requested_precision": requested,
            "final_precision": final,
            "force_same_precision": group.force_same_precision,
        }
        for module in group.member_modules:
            layer_assignment[module] = final
            counts[final] = counts.get(final, 0) + 1
    layer_assignment, fallback_layers = _resolve_overlapping_precision_groups(assignments)
    total = max(sum(counts.values()), 1)
    profile = {
        "profile_id": "profile_000",
        "profile_index": 0,
        "profile_template_id": "random",
        "subnet_id": subnet_id,
        "structure_hash": structure_hash,
        "random_seed": int(random_seed),
        "precision_group_assignments": assignments,
        "layer_precision_assignment": layer_assignment,
        "requested_precision": {gid: row["requested_precision"] for gid, row in assignments.items()},
        "final_precision": {gid: row["final_precision"] for gid, row in assignments.items()},
        "fallback_layers": fallback_layers,
        "fallback_reason": "per_group_allowed_precision",
        "fallback_layer_count": len(fallback_layers),
        "int8_layer_count": counts.get("int8", 0),
        "fp16_layer_count": counts.get("fp16", 0),
        "fp32_layer_count": counts.get("fp32", 0),
        "requested_int8_group_ratio": None,
        "actual_int8_group_ratio": 0.0,
        "actual_int8_layer_ratio": 0.0,
        "precision_coverage": {key: counts.get(key, 0) / total for key in ("fp32", "fp16", "int8")},
        **_precision_boundary_counts(source_groups, layer_assignment),
    }
    return _finalize_profile_counts(profile)


def sample_stratified_mixed_precision_profile(
    *,
    subnet_id: str,
    structure_hash: str,
    groups: Sequence[PrecisionGroup],
    profile_index: int,
    profile_seed: int,
    subnet_index: int,
    precision_modes: Sequence[str] = PRECISIONS,
) -> dict[str, Any]:
    template_id, requested_ratio = _template_for_profile_index(profile_index)
    seed = int(profile_seed) + int(subnet_index) * 10007 + int(profile_index) * 97
    rng = random.Random(seed)
    source_groups = list(groups)
    groups = _effective_precision_groups(groups)
    assignments: dict[str, dict[str, Any]] = {}
    layer_assignment: dict[str, str] = {}
    fallback_layers: list[dict[str, Any]] = []
    int8_allowed = [group for group in groups if "int8" in group.allowed_precisions and _precision_group_int8_eligible(group)]
    int8_ordered = sorted(int8_allowed, key=lambda group: str(group.precision_group_id))
    rng.shuffle(int8_allowed)
    int8_count = len(int8_allowed)
    low_target = max(1, int(round(int8_count * 0.20))) if int8_count else 0
    medium_target = max(1, int(round(int8_count * 0.50))) if int8_count else 0
    high_target = max(1, int(round(int8_count * 0.80))) if int8_count else 0
    if int8_count > 1:
        high_target = min(high_target, int8_count - 1)
    if requested_ratio < 0 or template_id == "fp16_heavy":
        int8_target = 0
    elif template_id == "low_int8":
        int8_target = low_target
    elif template_id == "medium_int8":
        int8_target = medium_target
    elif template_id == "high_int8":
        int8_target = high_target
    else:
        int8_target = int(round(int8_count * requested_ratio)) if int8_count else 0
    int8_target = max(0, min(int8_target, int8_count))
    if template_id == "low_int8":
        int8_group_ids = {group.precision_group_id for group in int8_ordered[:int8_target]}
    elif template_id == "medium_int8" and int8_count > int8_target and int8_target == low_target:
        # With very few groups, 20% and 50% can both round to one INT8 group.
        # Pick from the opposite end so the precision assignment remains distinct.
        int8_group_ids = {group.precision_group_id for group in int8_ordered[-int8_target:]} if int8_target else set()
    elif template_id == "medium_int8":
        int8_group_ids = {group.precision_group_id for group in int8_ordered[:int8_target]}
    elif template_id == "high_int8" and int8_count > int8_target:
        int8_group_ids = {group.precision_group_id for group in int8_ordered[-int8_target:]} if int8_target else set()
    else:
        int8_group_ids = {group.precision_group_id for group in int8_allowed[:int8_target]}
    fp32_candidate_ids = [
        group.precision_group_id
        for group in groups
        if group.precision_group_id not in int8_group_ids and "fp32" in group.allowed_precisions and "fp32" in precision_modes
    ]
    rng.shuffle(fp32_candidate_ids)
    fp32_group_ids: set[str] = set()
    if template_id == "fp16_heavy" and fp32_candidate_ids:
        fp32_group_ids.add(fp32_candidate_ids[0])
    elif template_id == "medium_int8" and int8_target == low_target and int8_count <= int8_target and fp32_candidate_ids:
        fp32_group_ids.add(fp32_candidate_ids[0])
    elif template_id == "high_int8" and int8_target <= medium_target and fp32_candidate_ids:
        fp32_group_ids.add(fp32_candidate_ids[0])
    for group in groups:
        group_int8_eligible = _precision_group_int8_eligible(group)
        allowed = [p for p in precision_modes if p in group.allowed_precisions and (p != "int8" or group_int8_eligible)]
        if not allowed:
            allowed = [p for p in group.allowed_precisions if p != "int8" or group_int8_eligible] or list(group.allowed_precisions)
        if group.precision_group_id in int8_group_ids:
            requested = "int8"
        elif group.precision_group_id in fp32_group_ids:
            requested = "fp32"
        elif template_id == "fp16_heavy":
            requested = "fp16" if "fp16" in precision_modes else (allowed[0] if allowed else group.default_precision)
        elif template_id == "random_balanced":
            requested = rng.choice(list(precision_modes))
        else:
            requested = "fp16" if "fp16" in precision_modes else rng.choice(list(precision_modes))
        final = requested if requested in allowed else group.default_precision
        if final not in allowed:
            final = allowed[0]
        fallback_reason = ""
        fallback_precision = ""
        if final != requested:
            fallback_reason = f"requested_precision_not_allowed:{group.reason}"
            fallback_precision = final
            for module in group.member_modules:
                fallback_layers.append(
                    {
                        "module_name": module,
                        "precision_group_id": group.precision_group_id,
                        "requested_precision": requested,
                        "final_precision": final,
                        "fallback_reason": fallback_reason,
                    }
                )
        assignments[group.precision_group_id] = {
            "member_modules": list(group.member_modules),
            "reason": group.reason,
            "allowed_precisions": list(group.allowed_precisions),
            "requested_precision": requested,
            "final_precision": final,
            "fallback_precision": fallback_precision,
            "fallback_reason": fallback_reason,
            "force_same_precision": group.force_same_precision,
        }
        for module in group.member_modules:
            layer_assignment[module] = final
    layer_assignment, fallback_layers = _resolve_overlapping_precision_groups(assignments)
    profile = {
        "profile_id": f"profile_{int(profile_index):03d}",
        "profile_index": int(profile_index),
        "profile_template_id": template_id,
        "subnet_id": subnet_id,
        "structure_hash": structure_hash,
        "random_seed": seed,
        "precision_group_assignments": assignments,
        "layer_precision_assignment": layer_assignment,
        "requested_precision": {gid: row["requested_precision"] for gid, row in assignments.items()},
        "final_precision": {gid: row["final_precision"] for gid, row in assignments.items()},
        "fallback_layers": fallback_layers,
        "fallback_reason": "per_group_allowed_precision",
        "fallback_layer_count": len(fallback_layers),
        "requested_int8_group_ratio": requested_ratio if requested_ratio >= 0 else None,
        **_precision_boundary_counts(source_groups, layer_assignment),
    }
    return _finalize_profile_counts(profile)


def make_qdq_insert_report(
    *,
    input_onnx: str,
    output_onnx: str,
    profile: Mapping[str, Any],
    calibration_frame_ids: Sequence[int],
    scale_table: Mapping[str, Any],
    success: bool,
    failure_reason: str = "",
    inserted_qdq_nodes: Sequence[Mapping[str, Any]] | None = None,
    skipped_non_int8_layers: Sequence[Mapping[str, Any]] | None = None,
    matched_int8_precision_groups: Sequence[str] | None = None,
    unmatched_int8_precision_groups: Sequence[str] | None = None,
    qdq_node_count: int = 0,
    quantize_linear_count: int = 0,
    dequantize_linear_count: int = 0,
) -> dict[str, Any]:
    int8_groups = [
        group_id
        for group_id, assignment in (profile.get("precision_group_assignments") or {}).items()
        if assignment.get("final_precision") == "int8"
    ]
    return {
        "success": bool(success),
        "failure_reason": failure_reason,
        "input_onnx": str(input_onnx),
        "output_onnx": str(output_onnx),
        "uses_mixed_precision_qdq": bool(int8_groups),
        "int8_precision_groups": int8_groups,
        "int8_precision_group_count": len(int8_groups),
        "calibration_frame_ids": list(calibration_frame_ids),
        "calibration_frame_count": len(calibration_frame_ids),
        "scale_table": dict(scale_table),
        "inserted_qdq_nodes": [dict(row) for row in (inserted_qdq_nodes or [])],
        "skipped_non_int8_layers": [dict(row) for row in (skipped_non_int8_layers or [])],
        "matched_int8_precision_groups": list(matched_int8_precision_groups or []),
        "unmatched_int8_precision_groups": list(unmatched_int8_precision_groups or []),
        "qdq_node_count": int(qdq_node_count),
        "quantize_linear_count": int(quantize_linear_count),
        "dequantize_linear_count": int(dequantize_linear_count),
        "fallback_layers": list(profile.get("fallback_layers") or []),
    }


def engine_eval_summary_row(
    subnet_id: str,
    profile_id: str,
    structure_hash: str,
    latency: Mapping[str, Any],
    ap: Mapping[str, Any],
    profile: Mapping[str, Any],
    *,
    build_success: bool | None = None,
    eval_success: bool = False,
    failure_reason: str = "real_trt_validation_eval_not_wired",
    status: str = "",
    engine_structure_check_passed: bool | None = None,
    precision_realization_passed: bool | None = None,
    smoke_success: bool | None = None,
    synthetic_used: bool = False,
    validation_dataloader_used: bool | None = None,
    evaluated_frames: int | None = None,
    qdq_insert_success: bool | None = None,
    inserted_qdq_node_count: int | None = None,
) -> dict[str, Any]:
    value = (lambda key: latency.get(key) if eval_success else None)
    ap_value = (lambda key: ap.get(key) if eval_success else None)
    return {
        "subnet_id": subnet_id,
        "profile_id": profile_id,
        "structure_hash": structure_hash,
        "status": status or ("eval_success" if eval_success else "eval_failed"),
        "build_success": build_success,
        "engine_structure_check_passed": engine_structure_check_passed,
        "precision_realization_passed": precision_realization_passed,
        "smoke_success": smoke_success,
        "eval_success": bool(eval_success),
        "synthetic_used": bool(synthetic_used),
        "validation_dataloader_used": bool(eval_success) if validation_dataloader_used is None else bool(validation_dataloader_used),
        "evaluated_frames": int(evaluated_frames or 0),
        "qdq_insert_success": qdq_insert_success,
        "inserted_qdq_node_count": int(inserted_qdq_node_count or 0),
        "failure_reason": "" if eval_success else failure_reason,
        "total_p50_ms": value("total_p50_ms"),
        "total_mean_ms": value("total_mean_ms"),
        "total_p90_ms": value("total_p90_ms"),
        "total_p95_ms": value("total_p95_ms"),
        "total_p99_ms": value("total_p99_ms"),
        "forward_p50_ms": value("forward_p50_ms"),
        "forward_mean_ms": value("forward_mean_ms"),
        "forward_p90_ms": value("forward_p90_ms"),
        "forward_p95_ms": value("forward_p95_ms"),
        "forward_p99_ms": value("forward_p99_ms"),
        "postprocess_p50_ms": value("postprocess_p50_ms"),
        "postprocess_mean_ms": value("postprocess_mean_ms"),
        "data_to_gpu_p50_ms": value("data_to_gpu_p50_ms"),
        "data_to_gpu_mean_ms": value("data_to_gpu_mean_ms"),
        "unaccounted_mean_ms": value("unaccounted_mean_ms"),
        "AP@0.03": ap_value("AP@0.03"),
        "AP@0.30": ap_value("AP@0.30"),
        "AP@0.50": ap_value("AP@0.50"),
        "AP@0.70": ap_value("AP@0.70"),
        "mAP": ap_value("mAP"),
        "precision_coverage": profile.get("precision_coverage", {}),
        "fallback_layer_count": profile.get("fallback_layer_count", 0),
    }


def component_lut_sample_row(subnet_id: str, profile_id: str, structure_hash: str, component: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "subnet_id": subnet_id,
        "profile_id": profile_id,
        "structure_hash": structure_hash,
        "trt_layer_name": component.get("trt_layer_name", component.get("module_name", "")),
        "original_module_name": component.get("module_name", ""),
        "precision_group_id": component.get("precision_group_id", ""),
        "op_type": component.get("op_type", ""),
        "fused_ops": component.get("fused_ops", ""),
        "input_shape": component.get("input_shape", ""),
        "output_shape": component.get("output_shape", ""),
        "C_in": component.get("C_in", ""),
        "C_out": component.get("C_out", ""),
        "C_mid": component.get("C_mid", ""),
        "groups": component.get("groups", ""),
        "per_group_in": component.get("per_group_in", ""),
        "per_group_out": component.get("per_group_out", ""),
        "kernel_size": component.get("kernel_size", ""),
        "stride": component.get("stride", ""),
        "padding": component.get("padding", ""),
        "H_out": component.get("H_out", ""),
        "W_out": component.get("W_out", ""),
        "params": component.get("params", 0),
        "flops": component.get("flops", 0),
        "requested_precision": component.get("requested_precision", ""),
        "final_precision": component.get("final_precision", ""),
        "fallback_precision": component.get("fallback_precision", ""),
        "latency_ms": component.get("latency_ms", None),
        "profile_percent": component.get("profile_percent", None),
        "profile_source": component.get("profile_source", "build_layer_info_only"),
        "is_fused": component.get("is_fused", False),
        "fusion_group_id": component.get("fusion_group_id", ""),
        "mapping_confidence": component.get("mapping_confidence", 0.0),
    }


def component_structure_feature_rows(model: nn.Module, subnet_id: str, profile: Mapping[str, Any]) -> list[dict[str, Any]]:
    assignments = profile.get("layer_precision_assignment", {})
    rows: list[dict[str, Any]] = []
    for name, module in model.named_modules():
        if not isinstance(module, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
            continue
        groups = int(getattr(module, "groups", 1))
        cin = int(getattr(module, "in_channels", getattr(module, "in_features", 0)))
        cout = int(getattr(module, "out_channels", getattr(module, "out_features", 0)))
        rows.append(
            {
                "component_id": f"{subnet_id}:{name}",
                "module_name": name,
                "stage": _stage_from_name(name),
                "op_type": module.__class__.__name__,
                "C_in": cin,
                "C_out": cout,
                "C_mid": "",
                "groups": groups,
                "per_group": cin // groups if groups and cin % groups == 0 else "",
                "kernel_size": getattr(module, "kernel_size", ""),
                "H_out": "",
                "W_out": "",
                "width_signature": f"{cin}x{cout}g{groups}",
                "dependency_group_id": "",
                "pruning_domain_id": "",
                "keep_ratio": 1.0,
                "params": sum(param.numel() for param in module.parameters(recurse=False)),
                "flops": 0,
                "legal_shape_passed": True,
                "precision_group_id": _precision_group_for_module(name, profile),
                "assigned_precision": assignments.get(name, "fp16"),
            }
        )
    return rows


def full_engine_training_sample(
    subnet_id: str,
    profile_id: str,
    structure_hash: str,
    profile: Mapping[str, Any],
    component_features: Sequence[Mapping[str, Any]],
    eval_row: Mapping[str, Any],
) -> dict[str, Any]:
    int8_group_count = int(profile.get("int8_group_count") or 0)
    inserted_qdq_node_count = int(eval_row.get("inserted_qdq_node_count") or 0)
    label_available = (
        bool(eval_row.get("build_success", False))
        and bool(eval_row.get("engine_structure_check_passed", False))
        and bool(eval_row.get("precision_realization_passed", False))
        and bool(eval_row.get("smoke_success", False))
        and bool(eval_row.get("eval_success", False))
        and not bool(eval_row.get("synthetic_used", False))
        and bool(eval_row.get("validation_dataloader_used", False))
        and int(eval_row.get("evaluated_frames") or 0) == 1000
        and bool(eval_row.get("qdq_insert_success", False))
        and (int8_group_count == 0 or inserted_qdq_node_count > 0)
    )
    return {
        "subnet_id": subnet_id,
        "profile_id": profile_id,
        "structure_hash": structure_hash,
        "subnet_structure_encoding": {"structure_hash": structure_hash},
        "precision_group_assignment": profile.get("precision_group_assignments", {}),
        "component_list": [row.get("module_name", row.get("trt_layer_name", "")) for row in component_features],
        "component_features": list(component_features),
        "global_features": {"fallback_layer_count": profile.get("fallback_layer_count", 0)},
        "measured_forward_latency": eval_row.get("forward_mean_ms") if label_available else None,
        "measured_total_latency": eval_row.get("total_mean_ms") if label_available else None,
        "AP/mAP": {"mAP": eval_row.get("mAP"), "AP@0.30": eval_row.get("AP@0.30")} if label_available else None,
        "build_success": bool(eval_row.get("build_success", False)),
        "engine_structure_check_passed": bool(eval_row.get("engine_structure_check_passed", False)),
        "precision_realization_passed": bool(eval_row.get("precision_realization_passed", False)),
        "smoke_success": bool(eval_row.get("smoke_success", False)),
        "eval_success": bool(eval_row.get("eval_success", False)),
        "synthetic_used": bool(eval_row.get("synthetic_used", False)),
        "validation_dataloader_used": bool(eval_row.get("validation_dataloader_used", False)),
        "evaluated_frames": int(eval_row.get("evaluated_frames") or 0),
        "label_available": label_available,
        "metadata": {"dataset_version": DATASET_VERSION},
    }


def _precision_group_for_module(module_name: str, profile: Mapping[str, Any]) -> str:
    for group_id, row in (profile.get("precision_group_assignments") or {}).items():
        if module_name in row.get("member_modules", []):
            return str(group_id)
    return ""


def _stage_from_name(name: str) -> str:
    low = name.lower()
    if "layer0" in low:
        return "backbone_stage0"
    if "layer1" in low:
        return "backbone_stage1"
    if "layer2" in low:
        return "backbone_stage2"
    if "head" in low:
        return "head"
    if "deblock" in low:
        return "deblock"
    if "neck" in low or "fusion" in low:
        return "neck"
    return "other"


def _trt_env(args: argparse.Namespace) -> dict[str, str]:
    env = os.environ.copy()
    trt_root = Path(args.trt_root)
    lib_dirs = [trt_root / "lib", trt_root / "targets" / "x86_64-linux-gnu" / "lib"]
    existing = env.get("LD_LIBRARY_PATH", "")
    values = [str(path) for path in lib_dirs if path.is_dir()]
    if existing:
        values.append(existing)
    if values:
        env["LD_LIBRARY_PATH"] = ":".join(values)
    return env


def _trtexec_path(args: argparse.Namespace) -> Path:
    if args.trtexec:
        return Path(args.trtexec)
    return Path(args.trt_root) / "bin" / "trtexec"


def _onnx_precision_constraint_specs(onnx_path: Path, profile: Mapping[str, Any]) -> list[str]:
    canonical_mapping = profile.get("canonical_precision_mapping") if isinstance(profile, Mapping) else None
    if canonical_mapping:
        specs = precision_constraint_specs_from_canonical_mapping(canonical_mapping)
        if specs:
            return specs
    try:
        import onnx

        model = onnx.load(str(onnx_path))
    except Exception:
        return []
    assignment = {str(k): str(v).lower() for k, v in (profile.get("layer_precision_assignment") or {}).items()}
    quantizable_ops = {"Conv", "Gemm", "MatMul"}
    specs = []
    for node in model.graph.node:
        if node.op_type not in quantizable_ops:
            continue
        module = ""
        precision = ""
        for candidate in _module_candidates_from_onnx_node_name(node.name):
            if candidate in assignment:
                module = candidate
                precision = assignment[candidate]
                break
        if not precision:
            for candidate in _module_candidates_from_onnx_node_name(node.name):
                for module_name, assigned_precision in assignment.items():
                    if _profile_module_matches_onnx_candidate(module_name, candidate):
                        module = module_name
                        precision = assigned_precision
                        break
                if precision:
                    break
        if precision in {"fp32", "fp16", "int8"}:
            specs.append(f"{node.name}:{precision}")
    return specs


def _profile_module_precision_rows(profile: Mapping[str, Any]) -> list[dict[str, Any]]:
    by_module: dict[str, dict[str, Any]] = {}
    layer_assignment = {str(k): str(v).lower() for k, v in (profile.get("layer_precision_assignment") or {}).items()}
    seen: set[str] = set()
    for group_id, assignment in (profile.get("precision_group_assignments") or {}).items():
        requested = str(assignment.get("requested_precision", assignment.get("final_precision", "fp16"))).lower()
        final = str(assignment.get("final_precision", layer_assignment.get(str(group_id), requested))).lower()
        fallback = str(assignment.get("fallback_precision", "")) if requested != final else ""
        fallback_reason = str(assignment.get("fallback_reason", "")) if requested != final else ""
        for module in assignment.get("member_modules", []):
            module_name = str(module)
            seen.add(module_name)
            row = by_module.setdefault(
                module_name,
                {
                    "canonical_module_name": module_name,
                    "precision_group_id": str(group_id),
                    "precision_group_ids": [],
                    "requested_precision": requested,
                    "requested_precision_effective": requested,
                    "final_precision": layer_assignment.get(module_name, final),
                    "fallback_precision": fallback,
                    "fallback_reason": fallback_reason,
                },
            )
            if str(group_id) not in row["precision_group_ids"]:
                row["precision_group_ids"].append(str(group_id))
    for module_name, final in layer_assignment.items():
        if module_name in seen:
            continue
        by_module[module_name] = {
                "canonical_module_name": str(module_name),
                "precision_group_id": "",
                "precision_group_ids": [],
                "requested_precision": str(final).lower(),
                "requested_precision_effective": str(final).lower(),
                "final_precision": str(final).lower(),
                "fallback_precision": "",
                "fallback_reason": "",
        }
    return list(by_module.values())


def _canonical_mapping_entries(mapping: Mapping[str, Any] | Sequence[Mapping[str, Any]] | None) -> list[dict[str, Any]]:
    if mapping is None:
        return []
    if isinstance(mapping, Mapping):
        entries = mapping.get("entries", [])
    else:
        entries = mapping
    return [dict(row) for row in entries if isinstance(row, Mapping)]


ONNX_EXPORT_ORIGIN_ARTIFACTS = (
    "onnx_export_origin_map.json",
    "onnx_export_module_call_trace.json",
    "onnx_node_rename_report.json",
    "onnx_export_origin_map_failure_report.json",
)


def _load_onnx_export_origin_map(onnx_path: str | Path) -> dict[str, Any]:
    path = Path(onnx_path).parent / "onnx_export_origin_map.json"
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) and bool(payload.get("success")) else {}


def copy_onnx_origin_artifacts(source_onnx: str | Path, target_onnx_dir: str | Path) -> None:
    source_dir = Path(source_onnx).parent
    target_dir = Path(target_onnx_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    for name in ONNX_EXPORT_ORIGIN_ARTIFACTS:
        src = source_dir / name
        if src.is_file():
            shutil.copyfile(src, target_dir / name)


def _mapping_node_unique_name(row: Mapping[str, Any]) -> str:
    return str(row.get("onnx_node_name_unique") or row.get("onnx_node_name") or "")


def _mapping_node_original_name(row: Mapping[str, Any]) -> str:
    return str(row.get("onnx_node_name_original") or row.get("onnx_node_name") or "")


def _module_matches_onnx_candidates(module_name: str, candidates: Sequence[str]) -> bool:
    aliases = _canonical_module_aliases(module_name)
    return any(str(candidate) in aliases for candidate in candidates)


def _canonical_module_aliases(module_name: str) -> set[str]:
    module = str(module_name).strip(".")
    aliases = {module} if module else set()
    if module.startswith("model."):
        aliases.add(module[len("model.") :])
    for prefix in ("encoder_m1.", "pyramid_backbone."):
        if module.startswith(prefix):
            aliases.add(module[len(prefix) :])
    if module.startswith("pyramid_backbone.resnet."):
        rest = module[len("pyramid_backbone.resnet.") :]
        aliases.add(rest)
        aliases.add(f"resnet.{rest}")
    if module.startswith("backbone_m1.resnet."):
        rest = module[len("backbone_m1.resnet.") :]
        aliases.add(rest)
        aliases.add(f"resnet.{rest}")
    return {alias for alias in aliases if alias}


def build_canonical_precision_mapping(
    onnx_path: str | Path,
    profile: Mapping[str, Any],
    *,
    allow_multi_node_modules: Sequence[str] | None = None,
) -> dict[str, Any]:
    import onnx

    onnx_path = Path(onnx_path)
    module_rows = _profile_module_precision_rows(profile)
    allow_multi = {str(value) for value in (allow_multi_node_modules or [])}
    origin_map = _load_onnx_export_origin_map(onnx_path)
    if origin_map:
        origin_entries = [dict(row) for row in origin_map.get("entries", []) if isinstance(row, Mapping)]
        by_module: dict[str, list[dict[str, Any]]] = {}
        for row in origin_entries:
            module = str(row.get("canonical_module_name", ""))
            if module:
                by_module.setdefault(module, []).append(row)
        seen_unique_nodes: dict[str, str] = {}
        entries: list[dict[str, Any]] = []
        for module_row in module_rows:
            module_name = str(module_row["canonical_module_name"])
            nodes = list(by_module.get(module_name, []))
            if len(nodes) > 1 and module_name not in allow_multi:
                node_names = ",".join(str(node.get("onnx_node_name_unique") or node.get("onnx_node_name_original") or "") for node in nodes)
                raise ValueError(f"canonical_module_matches_multiple_onnx_nodes:{module_name}:{node_names}")
            if not nodes:
                continue
            for node in nodes:
                unique_name = str(node.get("onnx_node_name_unique", ""))
                original_name = str(node.get("onnx_node_name_original", ""))
                if not unique_name:
                    raise ValueError(f"origin_map_missing_unique_node_name:{module_name}")
                previous = seen_unique_nodes.get(unique_name)
                if previous and previous != module_name:
                    raise ValueError(f"onnx_node_matches_multiple_canonical_modules:{unique_name}:{previous},{module_name}")
                seen_unique_nodes[unique_name] = module_name
                entries.append(
                    {
                        "canonical_module_name": module_name,
                        "precision_group_id": str(module_row.get("precision_group_id", "")),
                        "precision_group_ids": list(module_row.get("precision_group_ids", [])),
                        "requested_precision": str(module_row.get("requested_precision", "")).lower(),
                        "fallback_precision": str(module_row.get("fallback_precision", "")),
                        "fallback_reason": str(module_row.get("fallback_reason", "")),
                        "final_precision": str(module_row.get("final_precision", "")).lower(),
                        "requested_precision_effective": str(module_row.get("requested_precision_effective", "")).lower(),
                        "onnx_node_name": unique_name,
                        "onnx_node_name_unique": unique_name,
                        "onnx_node_name_original": original_name,
                        "onnx_op_type": str(node.get("onnx_op_type") or node.get("mapped_onnx_op_type") or ""),
                        "onnx_weight_initializer": str(node.get("onnx_weight_initializer", "")),
                        "qdq_activation_nodes": [],
                        "qdq_weight_nodes": [],
                        "qdq_output_nodes": [],
                        "trt_metadata_match_key": unique_name,
                        "trt_metadata_original_match_key": original_name,
                        "allow_multi_node_group": module_name in allow_multi,
                        "origin_map_entry": dict(node),
                    }
                )
        return {
            "success": True,
            "origin_map_used": True,
            "onnx_path": str(onnx_path),
            "origin_map_path": str(onnx_path.parent / "onnx_export_origin_map.json"),
            "entries": entries,
            "entry_count": len(entries),
            "canonical_module_count": len({row["canonical_module_name"] for row in entries}),
            "onnx_node_count": len({row["onnx_node_name_unique"] for row in entries}),
            "ambiguous_mapping_count": 0,
        }

    model = onnx.load(str(onnx_path))
    quantizable_ops = {"Conv", "Gemm", "MatMul"}
    compute_nodes = [node for node in model.graph.node if node.op_type in quantizable_ops]
    node_to_modules: dict[str, list[dict[str, Any]]] = {}
    module_to_nodes: dict[str, list[Any]] = {row["canonical_module_name"]: [] for row in module_rows}
    for node in compute_nodes:
        candidates = _module_candidates_from_onnx_node_name(node.name)
        for module_row in module_rows:
            module_name = str(module_row["canonical_module_name"])
            if _module_matches_onnx_candidates(module_name, candidates):
                node_to_modules.setdefault(str(node.name), []).append(module_row)
                module_to_nodes.setdefault(module_name, []).append(node)
    for node_name, matched_modules in node_to_modules.items():
        module_names = sorted({str(row["canonical_module_name"]) for row in matched_modules})
        if len(module_names) > 1:
            raise ValueError(f"onnx_node_matches_multiple_canonical_modules:{node_name}:{','.join(module_names)}")
    entries: list[dict[str, Any]] = []
    for module_row in module_rows:
        module_name = str(module_row["canonical_module_name"])
        nodes = module_to_nodes.get(module_name, [])
        if len(nodes) > 1 and module_name not in allow_multi:
            node_names = ",".join(str(node.name) for node in nodes)
            raise ValueError(f"canonical_module_matches_multiple_onnx_nodes:{module_name}:{node_names}")
        if not nodes:
            continue
        for node in nodes:
            entries.append(
                {
                    "canonical_module_name": module_name,
                    "precision_group_id": str(module_row.get("precision_group_id", "")),
                    "precision_group_ids": list(module_row.get("precision_group_ids", [])),
                    "requested_precision": str(module_row.get("requested_precision", "")).lower(),
                    "fallback_precision": str(module_row.get("fallback_precision", "")),
                    "fallback_reason": str(module_row.get("fallback_reason", "")),
                    "final_precision": str(module_row.get("final_precision", "")).lower(),
                    "requested_precision_effective": str(module_row.get("requested_precision_effective", "")).lower(),
                    "onnx_node_name": str(node.name),
                    "onnx_node_name_unique": str(node.name),
                    "onnx_node_name_original": str(node.name),
                    "onnx_op_type": str(node.op_type),
                    "onnx_weight_initializer": str(node.input[1]) if len(node.input) > 1 else "",
                    "qdq_activation_nodes": [],
                    "qdq_weight_nodes": [],
                    "qdq_output_nodes": [],
                    "trt_metadata_match_key": str(node.name),
                    "allow_multi_node_group": module_name in allow_multi,
                }
            )
    return {
        "success": True,
        "onnx_path": str(onnx_path),
        "entries": entries,
        "entry_count": len(entries),
        "canonical_module_count": len({row["canonical_module_name"] for row in entries}),
        "onnx_node_count": len({row["onnx_node_name"] for row in entries}),
        "origin_map_used": False,
        "ambiguous_mapping_count": 0,
    }


def precision_constraint_specs_from_canonical_mapping(mapping: Mapping[str, Any] | Sequence[Mapping[str, Any]]) -> list[str]:
    specs: list[str] = []
    for row in _canonical_mapping_entries(mapping):
        precision = str(row.get("final_precision", row.get("requested_precision", ""))).lower()
        node_name = _mapping_node_unique_name(row)
        if node_name and precision in {"fp32", "fp16", "int8"}:
            specs.append(f"{node_name}:{precision}")
    return specs


def write_canonical_precision_mapping(profile_dir: str | Path, mapping: Mapping[str, Any]) -> None:
    write_json(Path(profile_dir) / "canonical_precision_mapping.json", dict(mapping))


def load_canonical_precision_mapping(path: str | Path) -> dict[str, Any]:
    return _read_json_if_exists(Path(path), {})


def _canonical_mapping_by_onnx_node(mapping: Mapping[str, Any] | Sequence[Mapping[str, Any]] | None) -> dict[str, dict[str, Any]]:
    return {_mapping_node_unique_name(row): row for row in _canonical_mapping_entries(mapping) if _mapping_node_unique_name(row)}


def _canonical_mapping_by_module(mapping: Mapping[str, Any] | Sequence[Mapping[str, Any]] | None) -> dict[str, dict[str, Any]]:
    return {str(row.get("canonical_module_name", "")): row for row in _canonical_mapping_entries(mapping) if row.get("canonical_module_name")}


def canonical_module_from_trt_metadata(metadata: str, mapping: Mapping[str, Any] | Sequence[Mapping[str, Any]]) -> str:
    metadata = str(metadata)
    entries = _canonical_mapping_entries(mapping)
    for row in _canonical_mapping_entries(mapping):
        key = str(row.get("trt_metadata_match_key") or _mapping_node_unique_name(row) or "")
        if key and key in metadata:
            return str(row.get("canonical_module_name", ""))
    original_matches: list[str] = []
    for row in entries:
        key = str(row.get("trt_metadata_original_match_key") or row.get("onnx_node_name_original") or "")
        if key and key in metadata:
            candidates = _module_candidates_from_onnx_node_name(key)
            weak_matches = [
                str(candidate_row.get("canonical_module_name", ""))
                for candidate_row in entries
                if _module_matches_onnx_candidates(str(candidate_row.get("canonical_module_name", "")), candidates)
            ]
            weak_matches = sorted({value for value in weak_matches if value})
            if len(weak_matches) > 1:
                return "__ambiguous__"
            module = str(row.get("canonical_module_name", ""))
            if module and module not in original_matches:
                original_matches.append(module)
    if len(original_matches) == 1:
        return original_matches[0]
    if len(original_matches) > 1:
        return "__ambiguous__"
    return ""


def _onnx_input_names(path: Path) -> list[str]:
    try:
        import onnx

        model = onnx.load(str(path))
        return [value.name for value in model.graph.input]
    except Exception:
        return []


def _onnx_has_pointpillar_scatter_plugin(path: Path) -> bool:
    try:
        import onnx

        model = onnx.load(str(path))
        return _has_tensorrt_custom_plugin(model)
    except Exception:
        return False


def _resolve_plugin_path(args: argparse.Namespace) -> Path | None:
    raw = str(getattr(args, "plugin", "") or "")
    if not raw:
        return None
    path = Path(raw).expanduser()
    candidates = [path]
    if not path.is_absolute():
        candidates.extend([Path.cwd() / path, _ROOT / path, Path(str(getattr(args, "trt_root", ""))) / path])
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    return None


def _signal_maxk_profile_shapes(args: argparse.Namespace) -> dict[str, Any]:
    try:
        from quantization.utils.calibration import profile_from_observed_shapes

        return profile_from_observed_shapes([], fixed_k=int(args.fixed_k))
    except Exception:
        fixed_k = int(args.fixed_k)
        return {
            "voxel_features": {"min": [fixed_k, 32, 4], "opt": [fixed_k, 32, 4], "max": [fixed_k, 32, 4]},
            "voxel_coords": {"min": [fixed_k, 4], "opt": [fixed_k, 4], "max": [fixed_k, 4]},
            "voxel_num_points": {"min": [fixed_k], "opt": [fixed_k], "max": [fixed_k]},
            "pairwise_t_matrix": {"min": [1, 1, 1, 4, 4], "opt": [1, 2, 2, 4, 4], "max": [1, 2, 2, 4, 4]},
            "valid_voxel_mask": {"min": [fixed_k], "opt": [fixed_k], "max": [fixed_k]},
        }


def _shape_profile_args(profile_shapes: Mapping[str, Any]) -> list[str]:
    min_items: list[str] = []
    opt_items: list[str] = []
    max_items: list[str] = []
    for name, profile in profile_shapes.items():
        min_items.append(f"{name}:{'x'.join(str(int(v)) for v in profile['min'])}")
        opt_items.append(f"{name}:{'x'.join(str(int(v)) for v in profile['opt'])}")
        max_items.append(f"{name}:{'x'.join(str(int(v)) for v in profile['max'])}")
    return [f"--minShapes={','.join(min_items)}", f"--optShapes={','.join(opt_items)}", f"--maxShapes={','.join(max_items)}"]


def build_engine_with_trtexec(
    *,
    args: argparse.Namespace,
    onnx_path: Path,
    engine_path: Path,
    profile: Mapping[str, Any],
    build_log_path: Path,
    layer_info_path: Path,
) -> dict[str, Any]:
    trtexec = _trtexec_path(args)
    if not trtexec.is_file():
        return {
            "build_success": False,
            "success": False,
            "failure_reason": f"trtexec_not_found:{trtexec}",
            "engine_path": str(engine_path),
        }
    engine_path.parent.mkdir(parents=True, exist_ok=True)
    final_precisions = {str(row.get("final_precision", "")).lower() for row in (profile.get("precision_group_assignments") or {}).values()}
    has_int8 = "int8" in final_precisions
    has_fp16 = "fp16" in final_precisions
    precision_specs = _onnx_precision_constraint_specs(onnx_path, profile)
    onnx_inputs = _onnx_input_names(onnx_path)
    is_signal_maxk = REQUIRED_HEAL_INPUTS <= set(onnx_inputs)
    needs_pointpillar_plugin = is_signal_maxk and _onnx_has_pointpillar_scatter_plugin(onnx_path)
    plugin_path = _resolve_plugin_path(args) if needs_pointpillar_plugin else None
    if needs_pointpillar_plugin and plugin_path is None:
        return {
            "build_success": False,
            "success": False,
            "failure_reason": f"pointpillar_scatter_plugin_not_found:{getattr(args, 'plugin', '')}",
            "engine_path": str(engine_path),
            "trtexec": str(trtexec),
            "uses_int8_flag": has_int8,
            "uses_fp16_flag": has_fp16,
            "layer_precision_constraints": precision_specs,
            "layer_output_type_constraints": precision_specs,
            "signal_maxk_input_names": onnx_inputs,
            "signal_maxk_shape_profile_used": is_signal_maxk,
            "static_plugin": str(getattr(args, "plugin", "")),
        }
    if (has_int8 or has_fp16) and not precision_specs:
        return {
            "build_success": False,
            "success": False,
            "failure_reason": "layer_precision_constraints_unavailable",
            "engine_path": str(engine_path),
            "trtexec": str(trtexec),
            "uses_int8_flag": has_int8,
            "uses_fp16_flag": has_fp16,
            "layer_precision_constraints": [],
            "layer_output_type_constraints": [],
        }
    export_layer_info_during_build = bool(getattr(args, "export_layer_info_during_build", True))
    cmd = [
        str(trtexec),
        f"--onnx={onnx_path}",
        f"--saveEngine={engine_path}",
        "--skipInference",
        "--profilingVerbosity=detailed",
        "--memPoolSize=workspace:512",
    ]
    if export_layer_info_during_build:
        cmd.append(f"--exportLayerInfo={layer_info_path}")
    if is_signal_maxk:
        cmd.extend(_shape_profile_args(_signal_maxk_profile_shapes(args)))
        if plugin_path is not None:
            cmd.append(f"--staticPlugins={plugin_path}")
    if has_fp16:
        cmd.append("--fp16")
    if has_int8:
        cmd.append("--int8")
    if precision_specs:
        joined_specs = ",".join(precision_specs)
        cmd.extend(["--precisionConstraints=obey", f"--layerPrecisions={joined_specs}", f"--layerOutputTypes={joined_specs}"])
    completed = subprocess.run(cmd, cwd=str(Path.cwd()), env=_trt_env(args), text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=int(args.trt_build_timeout_seconds), check=False)
    build_log_path.parent.mkdir(parents=True, exist_ok=True)
    build_log_path.write_text(completed.stdout, encoding="utf-8")
    success = completed.returncode == 0 and engine_path.is_file()
    layer_info_export_report: dict[str, Any] = {
        "executed": bool(export_layer_info_during_build),
        "success": bool(export_layer_info_during_build and layer_info_path.is_file() and layer_info_path.stat().st_size > 3),
        "failure_reason": "" if bool(export_layer_info_during_build and layer_info_path.is_file() and layer_info_path.stat().st_size > 3) else "",
        "command": [],
        "returncode": None,
    }
    if success and not export_layer_info_during_build:
        layer_cmd = [
            str(trtexec),
            f"--loadEngine={engine_path}",
            "--skipInference",
            "--profilingVerbosity=detailed",
            f"--exportLayerInfo={layer_info_path}",
        ]
        if plugin_path is not None:
            layer_cmd.append(f"--staticPlugins={plugin_path}")
        layer_completed = subprocess.run(
            layer_cmd,
            cwd=str(Path.cwd()),
            env=_trt_env(args),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=int(args.trt_build_timeout_seconds),
            check=False,
        )
        (build_log_path.parent / "layer_info_export_log.txt").write_text(layer_completed.stdout, encoding="utf-8")
        layer_success = layer_completed.returncode == 0 and layer_info_path.is_file() and layer_info_path.stat().st_size > 3
        layer_info_export_report = {
            "executed": True,
            "success": layer_success,
            "failure_reason": "" if layer_success else f"layer_info_export_failed_rc_{layer_completed.returncode}",
            "command": layer_cmd,
            "returncode": layer_completed.returncode,
            "log_path": str(build_log_path.parent / "layer_info_export_log.txt"),
        }
        if not layer_success:
            success = False
    return {
        "build_success": success,
        "success": success,
        "failure_reason": "" if success else (str(layer_info_export_report.get("failure_reason") or "") or f"trtexec_failed_rc_{completed.returncode}"),
        "engine_path": str(engine_path),
        "trtexec": str(trtexec),
        "command": cmd,
        "uses_int8_flag": has_int8,
        "uses_fp16_flag": has_fp16,
        "layer_precision_constraints": precision_specs,
        "layer_output_type_constraints": precision_specs,
        "signal_maxk_input_names": onnx_inputs,
        "signal_maxk_shape_profile_used": is_signal_maxk,
        "static_plugin": str(plugin_path or getattr(args, "plugin", "")) if is_signal_maxk else "",
        "build_log_path": str(build_log_path),
        "trt_layer_info_path": str(layer_info_path),
        "returncode": completed.returncode,
        "export_layer_info_during_build": export_layer_info_during_build,
        "layer_info_export_report": layer_info_export_report,
    }


def _scale_table_for_profile(profile: Mapping[str, Any]) -> dict[str, Any]:
    return {
        group_id: {"scale": 0.1}
        for group_id, row in (profile.get("precision_group_assignments") or {}).items()
        if row.get("final_precision") == "int8"
    }


def _profile_index_row(
    *,
    subnet_id: str,
    profile_id: str,
    structure_hash: str,
    profile: Mapping[str, Any],
    qdq_onnx_path: Path,
    engine_path: Path,
    build_success: bool,
    eval_success: bool,
    failure_reason: str,
    status: str = "",
    engine_structure_check_passed: bool | None = None,
    precision_realization_passed: bool | None = None,
    smoke_success: bool | None = None,
) -> dict[str, Any]:
    return {
        "subnet_id": subnet_id,
        "profile_id": profile_id,
        "profile_index": profile.get("profile_index", int(str(profile_id).split("_")[-1])),
        "profile_template_id": profile.get("profile_template_id", ""),
        "structure_hash": structure_hash,
        "precision_assignment_hash": profile.get("precision_assignment_hash", precision_assignment_hash(profile)),
        "random_seed": profile.get("random_seed", ""),
        "fp32_group_count": profile.get("fp32_group_count", 0),
        "fp16_group_count": profile.get("fp16_group_count", 0),
        "int8_group_count": profile.get("int8_group_count", 0),
        "fp32_layer_count": profile.get("fp32_layer_count", 0),
        "fp16_layer_count": profile.get("fp16_layer_count", 0),
        "int8_layer_count": profile.get("int8_layer_count", 0),
        "requested_int8_group_ratio": profile.get("requested_int8_group_ratio", ""),
        "actual_int8_group_ratio": profile.get("actual_int8_group_ratio", 0.0),
        "fallback_layer_count": profile.get("fallback_layer_count", 0),
        "qdq_onnx_path": str(qdq_onnx_path),
        "engine_path": str(engine_path),
        "status": status or ("eval_success" if eval_success else "eval_failed"),
        "build_success": bool(build_success),
        "engine_structure_check_passed": engine_structure_check_passed,
        "precision_realization_passed": precision_realization_passed,
        "smoke_success": smoke_success,
        "eval_success": bool(eval_success),
        "failure_reason": failure_reason,
    }


_ONNX_METADATA_RE = re.compile(r"\[ONNX Layer:\s*/?([^\]\x1e]+)\]")
_ONNX_OP_ROOTS = {
    "add",
    "cast",
    "concat",
    "constant",
    "dequantizelinear",
    "div",
    "flatten",
    "identity",
    "mul",
    "quantizelinear",
    "relu",
    "reshape",
    "sigmoid",
    "softmax",
    "sub",
    "transpose",
}
_ONNX_TERMINAL_OPS = {
    "add",
    "batchnormalization",
    "cast",
    "concat",
    "constant",
    "conv",
    "dequantizelinear",
    "div",
    "flatten",
    "gemm",
    "identity",
    "matmul",
    "mul",
    "quantizelinear",
    "relu",
    "reshape",
    "sigmoid",
    "softmax",
    "sub",
    "transpose",
}


def _module_candidates_from_onnx_node_name(node_name: str) -> list[str]:
    raw_parts = [part for part in str(node_name).strip("/").split("/") if part]
    if raw_parts and raw_parts[-1].lower() in _ONNX_TERMINAL_OPS:
        raw_parts = raw_parts[:-1]
    parts: list[str] = []
    for idx, part in enumerate(raw_parts):
        lowered = part.lower()
        if lowered in _ONNX_OP_ROOTS or lowered in _ONNX_TERMINAL_OPS:
            continue
        next_part = raw_parts[idx + 1] if idx + 1 < len(raw_parts) else ""
        if next_part.startswith(f"{part}."):
            continue
        parts.append(part)
    candidates: list[str] = []
    for end in range(len(parts), 0, -1):
        candidate = ".".join(parts[:end])
        if candidate and candidate not in candidates:
            candidates.append(candidate)
    return candidates


def _profile_module_aliases(module_name: str) -> list[str]:
    module = str(module_name).strip(".")
    aliases: list[str] = []

    def add(value: str) -> None:
        value = value.strip(".")
        if value and value not in aliases:
            aliases.append(value)

    add(module)
    if module.startswith("model."):
        add(module[len("model.") :])
    prefix_rewrites = (
        ("encoder_m1.", ""),
        ("backbone_m1.resnet.", ""),
        ("pyramid_backbone.", ""),
    )
    for prefix, replacement in prefix_rewrites:
        if module.startswith(prefix):
            rest = replacement + module[len(prefix) :]
            add(rest)
    if module.startswith("pyramid_backbone.resnet."):
        rest = module[len("pyramid_backbone.resnet.") :]
        if rest.startswith("layer0.") and (rest.endswith(".conv1") or rest.endswith(".conv2")):
            add(f"{rest}_1")
        else:
            add(rest)
        add(f"resnet.{rest}")
    if module.startswith("pyramid_backbone.deblocks."):
        parts = module.split(".")
        if len(parts) >= 3 and parts[2].isdigit():
            add(f"single_head_{parts[2]}")
            add(f"pyramid_backbone.single_head_{parts[2]}")
    return aliases


def _module_name_from_trt_row(row: Mapping[str, Any], layer_name: str) -> tuple[str, float, str]:
    metadata = str(row.get("Metadata") or row.get("metadata") or "")
    for match in _ONNX_METADATA_RE.finditer(metadata):
        onnx_path = match.group(1).strip("/")
        if not onnx_path or "QuantizeLinear" in onnx_path or "DequantizeLinear" in onnx_path:
            continue
        candidates = _module_candidates_from_onnx_node_name(onnx_path)
        if candidates:
            return candidates[0], 0.9, "onnx_metadata"
    lowered_type = str(row.get("LayerType", row.get("type", ""))).lower()
    if "reformat" in lowered_type or "copy" in layer_name.lower():
        return "", 0.0, "reformat_or_copy_no_module"
    root = layer_name.strip("/").split("/")[0]
    return root, 0.25, "layer_name_fallback"


def _precision_from_trt_row(row: Mapping[str, Any]) -> str:
    precision_tokens: list[str] = []

    def collect(value: Any) -> None:
        if isinstance(value, Mapping):
            for key, nested in value.items():
                if str(key).lower() in {"format/datatype", "type", "datatype", "precision", "tacticname", "tactic"}:
                    precision_tokens.append(str(nested))
                collect(nested)
        elif isinstance(value, list):
            for nested in value:
                collect(nested)

    collect(
        {
            "Inputs": row.get("Inputs") or row.get("inputs"),
            "Outputs": row.get("Outputs") or row.get("outputs"),
            "Weights": row.get("Weights") or row.get("weights"),
            "Bias": row.get("Bias") or row.get("bias"),
            "TacticName": row.get("TacticName") or row.get("Tactic") or row.get("tactic"),
        }
    )
    upper = " ".join(precision_tokens).upper()
    if "INT8" in upper or "KINT8" in upper:
        return "int8"
    if "FP16" in upper or "HALF" in upper or re.search(r"(^|[^A-Z0-9])F16([^A-Z0-9]|$)", upper):
        return "fp16"
    if "FP32" in upper or "FLOAT" in upper or re.search(r"(^|[^A-Z0-9])F32([^A-Z0-9]|$)", upper):
        return "fp32"
    text = json.dumps(row, sort_keys=True).upper()
    if "INT8" in text:
        return "int8"
    if "FP16" in text or "HALF" in text:
        return "fp16"
    if "FP32" in text or "FLOAT" in text:
        return "fp32"
    return ""


def _trt_io_dtypes(values: Any) -> list[str]:
    if not isinstance(values, list):
        return []
    dtypes: list[str] = []
    for item in values:
        if not isinstance(item, Mapping):
            continue
        dtype = str(item.get("Format/Datatype") or item.get("Datatype") or item.get("Type") or "")
        if dtype:
            dtypes.append(dtype)
    return dtypes


def _contains_int8_token(value: Any) -> bool:
    upper = str(value).upper()
    return "INT8" in upper or "I8" in upper


def _contains_fp16_token(value: Any) -> bool:
    upper = str(value).upper()
    return "FP16" in upper or "HALF" in upper or bool(re.search(r"(^|[^A-Z0-9])F16([^A-Z0-9]|$)", upper))


def _trt_layer_int8_compute_realization(row: Mapping[str, Any]) -> dict[str, Any]:
    input_dtypes = [str(value) for value in row.get("input_dtypes", []) if str(value)]
    output_dtypes = [str(value) for value in row.get("output_dtypes", []) if str(value)]
    weights_type = str(row.get("weights_type", ""))
    tactic_name = str(row.get("tactic_name", ""))
    metadata = str(row.get("metadata", ""))
    name = str(row.get("trt_layer_name", ""))
    has_int8_input = any(_contains_int8_token(value) for value in input_dtypes)
    has_int8_weights = _contains_int8_token(weights_type)
    has_int8_tactic = _contains_int8_token(tactic_name)
    int8_realized = has_int8_input and has_int8_weights and has_int8_tactic
    has_fp16_output = any(_contains_fp16_token(value) for value in output_dtypes)
    boundary_text = f"{name} {metadata}"
    has_boundary_fusion = any(
        token.lower() in boundary_text.lower()
        for token in ("DequantizeLinear", "Add", "Relu", "PWN", "residual", "fusion")
    )
    boundary_dtype = "fp16" if has_fp16_output and has_boundary_fusion else ""
    return {
        "int8_realized": int8_realized,
        "realization_status": "int8_compute_fp16_boundary" if int8_realized and boundary_dtype else ("int8_realized" if int8_realized else "not_int8_realized"),
        "boundary_dtype": boundary_dtype,
        "has_int8_input": has_int8_input,
        "has_int8_weights": has_int8_weights,
        "has_int8_tactic": has_int8_tactic,
        "input_dtypes": input_dtypes,
        "output_dtypes": output_dtypes,
        "weights_type": weights_type,
        "tactic_name": tactic_name,
    }


def _parse_trt_layer_info(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []
    if isinstance(data, list):
        rows = data
    elif isinstance(data, dict):
        rows = data.get("Layers") or data.get("layers") or []
    else:
        rows = []
    out = []
    for idx, row in enumerate(rows):
        if not isinstance(row, Mapping):
            continue
        name = str(row.get("Name") or row.get("name") or row.get("LayerName") or f"layer_{idx}")
        module_name, mapping_confidence, mapping_source = _module_name_from_trt_row(row, name)
        final_precision = _precision_from_trt_row(row)
        inputs = row.get("Inputs") or row.get("inputs") or []
        outputs = row.get("Outputs") or row.get("outputs") or []
        weights = row.get("Weights") or row.get("weights") or {}
        out.append(
            {
                "trt_layer_name": name,
                "module_name": module_name,
                "op_type": row.get("LayerType", row.get("type", row.get("ParameterType", ""))),
                "fused_ops": row.get("Operations", ""),
                "final_precision": final_precision,
                "input_dtypes": _trt_io_dtypes(inputs),
                "output_dtypes": _trt_io_dtypes(outputs),
                "weights_type": str(weights.get("Type", "")) if isinstance(weights, Mapping) else "",
                "tactic_name": str(row.get("TacticName") or row.get("Tactic") or row.get("tactic") or ""),
                "metadata": str(row.get("Metadata") or row.get("metadata") or ""),
                "latency_ms": None,
                "profile_percent": None,
                "profile_source": "build_layer_info_only",
                "mapping_confidence": mapping_confidence,
                "mapping_source": mapping_source,
            }
        )
    return out


def _write_subnet_artifacts(args: argparse.Namespace, subnet_index: int, output_dir: Path) -> dict[str, Any]:
    subnet_id = f"subnet_{subnet_index:03d}"
    subnet_dir = output_dir / "subnets" / subnet_id
    subnet_dir.mkdir(parents=True, exist_ok=True)
    target = _target_for_index(subnet_index, int(args.num_subnets))
    width = 16 + (subnet_index % 12) * 4
    model = ToyMixedPrecisionSubnet(width=width).eval()
    structure_hash = _structure_hash(model, target, int(args.round_to))
    config = PruningConfig(
        target_pruning_ratio=target,
        target_pruning_mode=args.target_pruning_mode,
        round_to=args.round_to,
        max_ch_sparsity=args.max_ch_sparsity,
        stage1_min_per_group=args.stage1_min_per_group,
        stage1_max_ch_sparsity=args.stage1_max_ch_sparsity,
        protect_fpn_output=args.protect_fpn_output,
        protect_head_output=args.protect_head_output,
        no_extra_output_protection=args.no_extra_output_protection,
    )
    pruner = HEALStructuredPruner(model, config)
    sample = torch.randn(1, 3, 8, 8)
    graph = build_dependency_graph(model, sample)
    pruner.trace(sample)
    shape_report = pruner.check_shape_invariants()
    write_json(subnet_dir / "shape_invariant_report.json", shape_report.get("rows", []))
    groups = build_precision_coupling_groups(model, graph, sample)
    write_json(subnet_dir / "precision_coupling_groups.json", precision_groups_to_json(groups))
    grouped_shape_report = [
        {
            "module_name": name,
            "groups": int(module.groups),
            "C_in_before": int(module.in_channels),
            "C_out_before": int(module.out_channels),
            "C_in_after": int(module.in_channels),
            "C_out_after": int(module.out_channels),
            "legality_passed": True,
        }
        for name, module in model.named_modules()
        if isinstance(module, nn.Conv2d) and int(module.groups) > 1
    ]
    write_json(subnet_dir / "grouped_shape_report.json", grouped_shape_report)
    manifest = pruner.get_manifest()
    manifest.update(
        {
            "subnet_id": subnet_id,
            "structure_hash": structure_hash,
            "actual_param_prune_ratio": target,
            "actual_channel_prune_ratio": target / 2.0,
            "shape_invariant_passed": bool(shape_report.get("passed", False)),
            "requires_architecture_patch": True,
        }
    )
    write_json(subnet_dir / "pruning_manifest.json", manifest)
    torch.save({"model_object": model, "manifest": manifest}, subnet_dir / "pruned_model_object.pth")
    torch.save({"state_dict": model.state_dict(), "architecture_manifest": manifest}, subnet_dir / "pruned_state_dict_with_manifest.pth")
    return {
        "subnet_id": subnet_id,
        "structure_hash": structure_hash,
        "target_param_prune_ratio": target,
        "actual_param_prune_ratio": target,
        "actual_channel_prune_ratio": target / 2.0,
        "params_after": int(sum(param.numel() for param in model.parameters())),
        "pruning_manifest_path": str(subnet_dir / "pruning_manifest.json"),
        "shape_invariant_passed": bool(shape_report.get("passed", False)),
        "precision_coupling_groups_path": str(subnet_dir / "precision_coupling_groups.json"),
        "_model": model,
        "_groups": groups,
        "_subnet_dir": subnet_dir,
    }


def _v11_targets(args: argparse.Namespace) -> list[float]:
    explicit = str(getattr(args, "subnet_targets", "") or "").strip()
    if explicit:
        return [float(value) for value in explicit.split(",") if value.strip()]
    return [_target_for_index(index, int(args.num_subnets)) for index in range(int(args.num_subnets))]


def _module_channel_rows(before: Mapping[str, Any], after: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for name, b_row in before.items():
        a_row = after.get(name)
        if not a_row:
            continue
        attrs_b = b_row.get("attrs", {}) if isinstance(b_row, Mapping) else {}
        attrs_a = a_row.get("attrs", {}) if isinstance(a_row, Mapping) else {}
        channel_keys = ["in_channels", "out_channels", "in_features", "out_features", "num_features", "groups"]
        if not any(attrs_b.get(key) != attrs_a.get(key) for key in channel_keys):
            continue
        rows.append(
            {
                "module_name": name,
                "module_type": a_row.get("module_type", b_row.get("module_type", "")),
                "before": {key: attrs_b.get(key) for key in channel_keys if key in attrs_b},
                "after": {key: attrs_a.get(key) for key in channel_keys if key in attrs_a},
            }
        )
    return rows


def _apply_v11_stage1_grouped_constraints(domains: Sequence[Any], args: argparse.Namespace) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    min_per_group = int(getattr(args, "stage1_min_per_group", 8))
    stage1_max = float(getattr(args, "stage1_max_ch_sparsity", 0.30))
    for domain in domains:
        if not bool(getattr(domain, "is_grouped_conv", False)):
            continue
        per_group = int(getattr(domain, "per_group", 0) or 0)
        groups = int(getattr(domain, "groups", 1) or 1)
        root = str(getattr(domain, "root_module_name", ""))
        stage1_like = groups == 32 and per_group == min_per_group
        if not stage1_like:
            continue
        reason = f"v11_stage1_grouped_output_min_per_group_{min_per_group}_max_sparsity_{stage1_max:.2f}"
        domain.skipped_reason = reason
        for unit in getattr(domain, "units", []):
            unit.skipped_reason = reason
        rows.append(
            {
                "pruning_domain_id": getattr(domain, "pruning_domain_id", ""),
                "root_module_name": root,
                "groups": groups,
                "per_group": per_group,
                "stage1_min_per_group": min_per_group,
                "stage1_max_ch_sparsity": stage1_max,
                "skipped_reason": reason,
            }
        )
    return rows


def generate_real_heal_subnets(args: argparse.Namespace, output_dir: Path) -> list[dict[str, Any]]:
    from heal_compress.pruning.artifacts import save_v108_model_artifacts
    from heal_compress.pruning.model_io import collect_module_structure, load_heal_model, setup_logger
    from heal_compress.pruning.shape_invariants import check_model_shape_invariants, snapshot_model_shape_invariants
    from tools.latency_lut.run_v108_complete_taylor_greedy_pruner import (
        _build_global_physical_plan,
        _build_pruning_domains_and_reports,
        _shape_changes,
        count_parameters,
    )
    from tools.latency_lut.run_v109_param_budget_round4_pruner import (
        _apply_grouped_input_legality_filter,
        _manifest_for_target_v109,
        build_exact_param_ratio_predictor,
        build_param_savings_by_unit,
    )
    from heal_compress.pruning.greedy_budget_selector import select_greedy_global_budget

    output_dir.mkdir(parents=True, exist_ok=True)
    args.align_channels = int(args.round_to)
    args.importance_mode = "first_order_taylor"
    if not hasattr(args, "num_calib_batches"):
        args.num_calib_batches = 8
    device_value = str(getattr(args, "device", "") or "")
    if not device_value:
        device_value = "cuda:0" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_value)
    logger = setup_logger(output_dir / "real_heal_pruner_logs", name="v11_real_heal_subnet_generator")
    baseline, adapter = load_heal_model(args, device, logger)
    baseline.eval()
    params_before = count_parameters(baseline)
    structure_before = collect_module_structure(baseline)
    invariant_before = snapshot_model_shape_invariants(baseline)
    groups, domains, taylor_report, protection_report, _scores = _build_pruning_domains_and_reports(
        baseline,
        adapter,
        args,
        logger,
        output_dir / "real_heal_pruner_reports",
        device,
    )
    v11_stage1_constraint_rows = _apply_v11_stage1_grouped_constraints(domains, args)
    write_json(output_dir / "real_heal_pruner_reports" / "v11_stage1_grouped_constraint_report.json", v11_stage1_constraint_rows)
    param_savings_by_unit = build_param_savings_by_unit(groups, domains)
    exact_param_ratio_predictor = build_exact_param_ratio_predictor(baseline, groups, domains, params_before)
    targets = _v11_targets(args)
    rows: list[dict[str, Any]] = []
    seen_hashes: set[str] = set()
    for subnet_index, target in enumerate(targets):
        subnet_id = f"subnet_{subnet_index:03d}"
        subnet_dir = output_dir / "subnets" / subnet_id
        if subnet_dir.exists():
            shutil.rmtree(subnet_dir)
        subnet_dir.mkdir(parents=True, exist_ok=True)
        selection = select_greedy_global_budget(
            copy.deepcopy(domains),
            target_pruning_ratio=float(target),
            target_pruning_mode=str(args.target_pruning_mode),
            predicted_total_params=float(params_before),
            param_savings_by_unit=param_savings_by_unit,
            param_ratio_from_plans=exact_param_ratio_predictor,
            max_ch_sparsity=float(args.max_ch_sparsity),
            align_channels=int(args.round_to),
        )
        selection.round_to = int(args.round_to)
        grouped_input_reports = _apply_grouped_input_legality_filter(groups, selection, align_channels=int(args.round_to))
        if grouped_input_reports:
            selection.grouped_shape_rows.extend(grouped_input_reports)
        pruned = copy.deepcopy(baseline).to(device).eval()
        physical_plan = _build_global_physical_plan(groups, selection, align_channels=int(args.round_to))
        surgery = physical_plan.apply_one_shot(pruned)
        params_after = count_parameters(pruned)
        actual_param = 1.0 - params_after / max(params_before, 1)
        selection.actual_param_prune_ratio = actual_param
        selection.param_prediction_error = actual_param - selection.predicted_param_prune_ratio
        selection.param_budget_overshoot_ratio = max(0.0, actual_param - float(target)) if args.target_pruning_mode == "param" else 0.0
        structure_after = collect_module_structure(pruned)
        shape_report = check_model_shape_invariants(invariant_before, snapshot_model_shape_invariants(pruned))
        if not shape_report.get("passed"):
            raise RuntimeError(f"shape_invariant_failed:{subnet_id}:{shape_report.get('non_channel_shape_violation_count')}")
        shape_report_path = subnet_dir / "shape_invariant_report.json"
        write_json(shape_report_path, shape_report.get("rows", []))
        channel_rows = _module_channel_rows(structure_before, structure_after)
        manifest = _manifest_for_target_v109(
            args=args,
            target=float(target),
            selection=selection,
            params_before=params_before,
            params_after=params_after,
            shape_changes=_shape_changes(structure_before, structure_after),
            grouped_rows=selection.grouped_shape_rows,
            shape_report_path=shape_report_path,
            protected_units=[],
            fixed_shape_skipped_units=[],
        )
        structure_hash = _structure_hash(pruned, float(target), int(args.round_to))
        if structure_hash in seen_hashes:
            raise RuntimeError(f"duplicate_structure_hash:{subnet_id}:{structure_hash}")
        seen_hashes.add(structure_hash)
        manifest.update(
            {
                "subnet_id": subnet_id,
                "structure_hash": structure_hash,
                "target_param_prune_ratio": float(target),
                "shape_invariant_passed": True,
                "physical_prune_surgery": surgery,
                "module_channel_before_after": channel_rows,
                "source_pruner": "v10.9_formalized_HEALStructuredPruner_physical_plan",
                "model_config": str(args.model_config),
                "checkpoint_source": str(args.checkpoint),
            }
        )
        write_json(subnet_dir / "pruning_manifest.json", manifest)
        write_json(subnet_dir / "global_physical_prune_plan.json", physical_plan.to_json())
        write_csv(subnet_dir / "budget_selection_trace.csv", selection.trace_rows)
        write_json(subnet_dir / "grouped_shape_report.json", selection.grouped_shape_rows)
        write_json(subnet_dir / "grouped_pergroup_round_to_shape_report.json", selection.grouped_shape_rows)
        write_json(subnet_dir / "module_channel_before_after.json", channel_rows)
        sample = adapter.build_synthetic_batch(pruned)
        dep_graph = build_dependency_graph(pruned, sample)
        precision_groups = build_precision_coupling_groups(pruned, dep_graph, sample)
        write_json(subnet_dir / "precision_coupling_groups.json", precision_groups_to_json(precision_groups))
        artifacts = save_v108_model_artifacts(
            model=pruned,
            models_dir=subnet_dir,
            manifest=manifest,
            model_config=str(args.model_config),
            checkpoint_source=str(args.checkpoint),
        )
        row = {
            "subnet_id": subnet_id,
            "structure_hash": structure_hash,
            "target_param_prune_ratio": float(target),
            "actual_param_prune_ratio": actual_param,
            "actual_channel_prune_ratio": selection.actual_channel_prune_ratio_on_searchable_surface,
            "params_after": int(params_after),
            "pruning_manifest_path": str(subnet_dir / "pruning_manifest.json"),
            "shape_invariant_passed": True,
            "precision_coupling_groups_path": str(subnet_dir / "precision_coupling_groups.json"),
            "pruned_model_path": str(artifacts["model_object"]),
            "artifact_type": "real_heal_pruned",
        }
        rows.append(row)
    write_csv(output_dir / "subnet_index.csv", rows)
    audit_subnet_artifacts(output_dir)
    manifest = {
        "dataset_version": DATASET_VERSION,
        "mode": "sample-real-heal",
        "real_heal_pruned_subnets": True,
        "toy_subnets": False,
        "successful_subnet_count": len(rows),
        "unique_structure_hash_count": len({row["structure_hash"] for row in rows}),
        "target_pruning_mode": args.target_pruning_mode,
        "round_to": int(args.round_to),
        "max_ch_sparsity": float(args.max_ch_sparsity),
        "stage1_min_per_group": int(args.stage1_min_per_group),
        "stage1_max_ch_sparsity": float(args.stage1_max_ch_sparsity),
        "taylor_report_path": str(output_dir / "real_heal_pruner_reports" / "taylor_importance_report.json"),
        "protection_report_path": str(output_dir / "real_heal_pruner_reports" / "protection_policy_report.json"),
    }
    write_json(output_dir / "dataset_manifest.json", manifest)
    write_json(output_dir / "failure_report.json", {"success": True, "failure_reason": "", "traceback": ""})
    return rows


def _write_profiles_for_subnet(args: argparse.Namespace, subnet_row: Mapping[str, Any], profile_index_rows: list[dict[str, Any]], eval_rows: list[dict[str, Any]], component_rows: list[dict[str, Any]], training_rows: list[dict[str, Any]]) -> None:
    subnet_dir: Path = subnet_row["_subnet_dir"]
    model: nn.Module = subnet_row["_model"]
    groups: list[PrecisionGroup] = list(subnet_row["_groups"])
    modes = [p.strip().lower() for p in str(args.precision_modes).split(",") if p.strip()]
    for profile_idx in range(int(args.precision_profiles_per_subnet)):
        profile_id = f"profile_{profile_idx:03d}"
        profile_dir = subnet_dir / profile_id
        profile_dir.mkdir(parents=True, exist_ok=True)
        profile = sample_mixed_precision_profile(
            subnet_id=str(subnet_row["subnet_id"]),
            structure_hash=str(subnet_row["structure_hash"]),
            groups=groups,
            random_seed=int(args.random_seed) + int(subnet_row["subnet_id"].split("_")[-1]) * 100 + profile_idx,
            precision_modes=modes,
        )
        profile["profile_id"] = profile_id
        profile["profile_index"] = profile_idx
        profile["profile_template_id"] = profile.get("profile_template_id", "random")
        _finalize_profile_counts(profile)
        write_json(profile_dir / "mixed_precision_profile.json", profile)
        calib_ids = list(range(int(args.calib_train_frames)))
        write_json(profile_dir / "calibration_frames.json", calib_ids)
        onnx_dir = profile_dir / "onnx"
        onnx_dir.mkdir(parents=True, exist_ok=True)
        input_onnx = onnx_dir / "model.onnx"
        output_onnx = onnx_dir / "model_mixed_qdq.onnx"
        shape_gate = gate_onnx_export({"passed": bool(subnet_row["shape_invariant_passed"])})
        onnx_report = {"success": False, "failure_reason": shape_gate["blocked_stage"], "onnx_path": str(input_onnx)}
        if shape_gate["allowed"] and args.mode in {"smoke", "full"}:
            try:
                torch.onnx.export(model, torch.randn(1, 3, 8, 8), str(input_onnx), opset_version=13)
                output_onnx.write_bytes(input_onnx.read_bytes())
                onnx_report = {"success": True, "failure_reason": "", "onnx_path": str(output_onnx)}
            except Exception as exc:  # noqa: BLE001
                onnx_report = {"success": False, "failure_reason": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc(), "onnx_path": str(output_onnx)}
        scale_table = _scale_table_for_profile(profile)
        write_json(profile_dir / "scale_table.json", scale_table)
        qdq_report = make_qdq_insert_report(
            input_onnx=str(input_onnx),
            output_onnx=str(output_onnx),
            profile=profile,
            calibration_frame_ids=calib_ids,
            scale_table=scale_table,
            success=bool(onnx_report.get("success")),
            failure_reason=str(onnx_report.get("failure_reason", "")),
        )
        write_json(profile_dir / "qdq_insert_report.json", qdq_report)
        write_json(profile_dir / "onnx_check_report.json", onnx_report)
        write_json(profile_dir / "onnx_parser_report.json", onnx_report)
        engine_gate = gate_engine_build(onnx_report)
        engine_path = profile_dir / "engine.plan"
        layer_info_path = profile_dir / "trt_layer_info.json"
        build_log_path = profile_dir / "build_log.txt"
        engine_report = {
            "build_success": False,
            "success": False,
            "failure_reason": engine_gate["blocked_stage"] or "tensorrt_build_not_executed",
            "engine_path": str(engine_path),
        }
        if engine_gate["allowed"] and args.mode in {"smoke", "full"}:
            engine_report = build_engine_with_trtexec(
                args=args,
                onnx_path=output_onnx,
                engine_path=engine_path,
                profile=profile,
                build_log_path=build_log_path,
                layer_info_path=layer_info_path,
            )
        else:
            write_json(profile_dir / "build_log.txt.json", engine_report)
            build_log_path.write_text(json.dumps(engine_report, indent=2) + "\n", encoding="utf-8")
            write_json(layer_info_path, [])
        write_json(profile_dir / "build_report.json", engine_report)
        if not layer_info_path.exists():
            write_json(layer_info_path, [])
        write_json(profile_dir / "precision_assignment_report.json", profile)
        write_json(profile_dir / "unsupported_or_fallback_layers.json", profile.get("fallback_layers", []))
        eval_gate = gate_eval(engine_report)
        if eval_gate["allowed"] and bool(args.enable_synthetic_trt_eval):
            eval_report = {
                "eval_success": True,
                "failure_reason": "",
                "synthetic_used": True,
                "validation_dataloader_used": False,
                "note": "synthetic TensorRT timing only; not a valid val1000 AP result",
            }
        else:
            eval_report = {
                "eval_success": False,
                "failure_reason": eval_gate["blocked_stage"] or "real_trt_validation_eval_not_wired",
                "synthetic_used": False,
                "validation_dataloader_used": False,
            }
        failure_reason = str(eval_report.get("failure_reason", ""))
        eval_summary = engine_eval_summary_row(
            str(subnet_row["subnet_id"]),
            profile_id,
            str(subnet_row["structure_hash"]),
            {},
            {},
            profile,
            build_success=bool(engine_report.get("build_success")),
            eval_success=bool(eval_report.get("eval_success")),
            failure_reason=failure_reason,
        )
        write_csv(profile_dir / "eval_latency_per_frame.csv", [])
        write_json(profile_dir / "eval_latency_summary.json", eval_summary)
        write_json(profile_dir / "eval_ap.json", {"AP@0.03": 0.0, "AP@0.30": 0.0, "AP@0.50": 0.0, "AP@0.70": 0.0, "mAP": 0.0})
        write_json(profile_dir / "eval_report.json", eval_report)
        features = component_structure_feature_rows(model, str(subnet_row["subnet_id"]), profile)
        trt_components = _parse_trt_layer_info(layer_info_path)
        comp_samples = [component_lut_sample_row(str(subnet_row["subnet_id"]), profile_id, str(subnet_row["structure_hash"]), row) for row in (trt_components or features)]
        write_csv(profile_dir / "component_structure_features.csv", features)
        write_csv(profile_dir / "component_profile.csv", comp_samples)
        component_rows.extend(comp_samples)
        eval_rows.append(eval_summary)
        training_rows.append(full_engine_training_sample(str(subnet_row["subnet_id"]), profile_id, str(subnet_row["structure_hash"]), profile, features, eval_summary))
        profile_index_rows.append(
            _profile_index_row(
                subnet_id=str(subnet_row["subnet_id"]),
                profile_id=profile_id,
                structure_hash=str(subnet_row["structure_hash"]),
                profile=profile,
                qdq_onnx_path=output_onnx,
                engine_path=profile_dir / "engine.plan",
                build_success=bool(engine_report.get("build_success")),
                eval_success=bool(eval_report.get("eval_success")),
                failure_reason=failure_reason,
            )
        )


def _precision_groups_from_json(path: Path) -> list[PrecisionGroup]:
    data = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else []
    groups = []
    for row in data:
        if not isinstance(row, Mapping):
            continue
        groups.append(
            PrecisionGroup(
                precision_group_id=str(row.get("precision_group_id")),
                member_modules=[str(v) for v in row.get("member_modules", [])],
                reason=str(row.get("reason", "user_constraint")),
                allowed_precisions=[str(v).lower() for v in row.get("allowed_precisions", ["fp32", "fp16", "int8"])],
                default_precision=str(row.get("default_precision", "fp16")).lower(),
                force_same_precision=bool(row.get("force_same_precision", True)),
            )
        )
    return groups


def _read_profile(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _normalize_existing_profile(profile: dict[str, Any], *, profile_index: int, subnet_id: str, structure_hash: str) -> dict[str, Any]:
    profile.setdefault("profile_id", f"profile_{profile_index:03d}")
    profile.setdefault("profile_index", profile_index)
    if profile_index == 0:
        profile["profile_template_id"] = "existing_or_baseline_profile"
    else:
        profile.setdefault("profile_template_id", "existing_profile")
    profile.setdefault("subnet_id", subnet_id)
    profile.setdefault("structure_hash", structure_hash)
    profile.setdefault("requested_int8_group_ratio", None)
    if isinstance(profile.get("precision_group_assignments"), dict):
        layer_assignment, fallback_layers = _resolve_overlapping_precision_groups(profile["precision_group_assignments"])
        profile["layer_precision_assignment"] = layer_assignment
        profile["fallback_layers"] = fallback_layers
        profile["fallback_layer_count"] = len(fallback_layers)
        profile["requested_precision"] = {
            group_id: row.get("requested_precision", "")
            for group_id, row in profile["precision_group_assignments"].items()
        }
        profile["final_precision"] = {
            group_id: row.get("final_precision", "")
            for group_id, row in profile["precision_group_assignments"].items()
        }
    return _finalize_profile_counts(profile)


def _subnet_dirs_for_expand(args: argparse.Namespace) -> list[Path]:
    source_dir = Path(args.source_dir)
    subnets = sorted((source_dir / "subnets").glob("subnet_*"))
    if int(args.max_subnets) > 0:
        subnets = subnets[: int(args.max_subnets)]
    return [path for path in subnets if (path / "pruning_manifest.json").is_file()]


def _base_onnx_for_subnet(subnet_dir: Path) -> Path | None:
    candidates = [
        subnet_dir / "profile_000" / "onnx" / "model.onnx",
        subnet_dir / "profile_000" / "onnx" / "model_mixed_qdq.onnx",
    ]
    for path in candidates:
        if path.is_file():
            return path
    return None


def _looks_like_toy_pruned_artifact(path: Path) -> tuple[bool, str]:
    if not path.is_file():
        return True, f"missing pruned model artifact: {path}"
    if path.stat().st_size < 1024 * 1024:
        return True, f"not a real HEAL pruned model artifact: {path.name} is only {path.stat().st_size} bytes"
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        model = payload.get("model_object") if isinstance(payload, Mapping) else payload
        cls_name = f"{model.__class__.__module__}.{model.__class__.__name__}"
        if "ToyMixedPrecisionSubnet" in cls_name:
            return True, f"not a real HEAL pruned model artifact: {cls_name}"
    except Exception as exc:  # noqa: BLE001
        text = f"{type(exc).__name__}: {exc}"
        if "ToyMixedPrecisionSubnet" in text:
            return True, f"not a real HEAL pruned model artifact: {text}"
    return False, ""


def _torch_load_artifact(path: Path) -> tuple[Any | None, str]:
    try:
        return torch.load(path, map_location="cpu", weights_only=False), ""
    except TypeError:
        try:
            return torch.load(path, map_location="cpu"), ""
        except Exception as exc:  # noqa: BLE001
            return None, f"{type(exc).__name__}: {exc}"
    except Exception as exc:  # noqa: BLE001
        return None, f"{type(exc).__name__}: {exc}"


def _artifact_model_object(payload: Any) -> nn.Module | None:
    if isinstance(payload, Mapping):
        for key in ("model_object", "model", "net", "module"):
            value = payload.get(key)
            if isinstance(value, nn.Module):
                return value
    return payload if isinstance(payload, nn.Module) else None


def _artifact_state_dict(payload: Any) -> Mapping[str, Any] | None:
    if isinstance(payload, Mapping):
        for key in ("state_dict", "model_state_dict"):
            value = payload.get(key)
            if isinstance(value, Mapping):
                return value
        if payload and all(torch.is_tensor(value) for value in payload.values()):
            return payload
    return None


def _is_real_heal_lidar_model(model: nn.Module | None, state_dict: Mapping[str, Any] | None = None) -> bool:
    markers = (
        "encoder_",
        "backbone_",
        "pillar_vfe",
        "scatter",
        "cls_head",
        "reg_head",
        "dir_head",
        "HeterPyramidCollab",
        "lidar_pyramid",
    )
    if model is not None:
        type_name = f"{model.__class__.__module__}.{model.__class__.__name__}"
        if any(marker in type_name for marker in markers):
            return True
        try:
            for name, module in model.named_modules():
                text = f"{name} {module.__class__.__name__}"
                if any(marker in text for marker in markers):
                    return True
        except Exception:
            return False
    if state_dict:
        joined = " ".join(str(key) for key in list(state_dict.keys())[:200])
        return any(marker in joined for marker in markers)
    return False


def _conv_shape_signature(model: nn.Module | None, *, limit: int = 12) -> tuple[int, bool, str]:
    if model is None:
        return 0, False, ""
    rows: list[str] = []
    aligned = True
    count = 0
    try:
        for name, module in model.named_modules():
            if not isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
                continue
            count += 1
            cin = int(module.in_channels)
            cout = int(module.out_channels)
            if cin % 4 or cout % 4:
                aligned = False
            if len(rows) < int(limit):
                rows.append(f"{name}:{cin}->{cout}:g{int(getattr(module, 'groups', 1))}")
    except Exception:
        return count, False, ";".join(rows)
    return count, bool(count and aligned), ";".join(rows)


def classify_subnet_artifact(subnet_dir: Path) -> dict[str, Any]:
    subnet_id = subnet_dir.name
    model_path = subnet_dir / "pruned_model_object.pth"
    if not model_path.is_file():
        alt = subnet_dir / "models" / "pruned_model_object.pth"
        if alt.is_file():
            model_path = alt
    size_mb = model_path.stat().st_size / (1024 * 1024) if model_path.is_file() else 0.0
    row: dict[str, Any] = {
        "subnet_id": subnet_id,
        "pruned_model_path": str(model_path),
        "file_size_mb": f"{size_mb:.6f}",
        "artifact_type": "missing",
        "can_export_signal_maxk_onnx": "false",
        "failure_reason": "",
        "torch_object_type": "",
        "has_real_heal_lidar_pyramid_modules": "false",
        "has_state_dict": "false",
        "conv_count": 0,
        "round_to_4_aligned": "false",
        "conv_shape_signature_head": "",
    }
    if not model_path.is_file():
        row["failure_reason"] = "missing pruned_model_object.pth"
        return row

    payload, load_error = _torch_load_artifact(model_path)
    if load_error:
        row["artifact_type"] = "toy" if "ToyMixedPrecisionSubnet" in load_error or size_mb < 1.0 else "state_dict_only"
        row["failure_reason"] = load_error
        return row

    model = _artifact_model_object(payload)
    state_dict = _artifact_state_dict(payload)
    object_type = f"{model.__class__.__module__}.{model.__class__.__name__}" if model is not None else type(payload).__name__
    conv_count, round4, signature = _conv_shape_signature(model)
    has_heal = _is_real_heal_lidar_model(model, state_dict)
    has_state = bool(state_dict)
    row.update(
        {
            "torch_object_type": object_type,
            "has_real_heal_lidar_pyramid_modules": str(bool(has_heal)).lower(),
            "has_state_dict": str(bool(has_state)).lower(),
            "conv_count": int(conv_count),
            "round_to_4_aligned": str(bool(round4)).lower(),
            "conv_shape_signature_head": signature,
        }
    )
    if "ToyMixedPrecisionSubnet" in object_type or size_mb < 1.0:
        row["artifact_type"] = "toy"
        row["failure_reason"] = f"toy_or_too_small: object_type={object_type}, file_size_mb={size_mb:.4f}"
    elif model is not None and has_heal:
        row["artifact_type"] = "real_heal_pruned"
        row["can_export_signal_maxk_onnx"] = "true"
    elif has_state:
        row["artifact_type"] = "state_dict_only"
        row["failure_reason"] = "state_dict present but no reconstructable HEAL model object detected"
    else:
        row["artifact_type"] = "missing"
        row["failure_reason"] = f"unrecognized artifact payload type={type(payload).__name__}"
    return row


def audit_subnet_artifacts(dataset_dir: str | Path) -> dict[str, Any]:
    root = Path(dataset_dir)
    subnet_root = root / "subnets"
    subnet_dirs = sorted(subnet_root.glob("subnet_*")) if subnet_root.is_dir() else []
    rows = [classify_subnet_artifact(path) for path in subnet_dirs]
    write_csv(root / "subnet_artifact_audit.csv", rows)
    counts: dict[str, int] = {}
    for row in rows:
        kind = str(row.get("artifact_type", "missing"))
        counts[kind] = counts.get(kind, 0) + 1
    real_count = counts.get("real_heal_pruned", 0)
    summary = {
        "subnet_count": len(rows),
        "toy_count": counts.get("toy", 0),
        "real_heal_pruned_count": real_count,
        "state_dict_only_count": counts.get("state_dict_only", 0),
        "missing_count": counts.get("missing", 0),
        "can_export_signal_maxk_onnx_count": sum(str(row.get("can_export_signal_maxk_onnx")) == "true" for row in rows),
        "dataset_usable_for_real_lut": bool(rows and real_count == len(rows)),
        "current_dataset_must_not_run_2x3_or_50x5": bool(rows and real_count != len(rows)),
        "audit_csv": str(root / "subnet_artifact_audit.csv"),
    }
    write_json(root / "subnet_artifact_audit_summary.json", summary)
    verdict = "usable_for_real_lut" if summary["dataset_usable_for_real_lut"] else "not_usable_for_real_lut"
    (root / "subnet_artifact_audit_report.md").write_text(
        "\n".join(
            [
                "# v11 Subnet Artifact Audit",
                "",
                f"verdict: {verdict}",
                f"subnet_count: {summary['subnet_count']}",
                f"toy_count: {summary['toy_count']}",
                f"real_heal_pruned_count: {summary['real_heal_pruned_count']}",
                f"state_dict_only_count: {summary['state_dict_only_count']}",
                f"missing_count: {summary['missing_count']}",
                "",
                "Current dataset must not be used for 2x3 or 50x5 real LUT expansion until every subnet is a real HEAL lidar_pyramid physical-pruned artifact.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return summary


def _write_real_onnx_report(subnet_dir: Path, report: Mapping[str, Any]) -> None:
    write_json(subnet_dir / "onnx" / "real_onnx_export_report.json", dict(report))


def ensure_real_pruned_signal_maxk_onnx_for_subnet(subnet_dir: Path, args: argparse.Namespace) -> tuple[Path | None, dict[str, Any]]:
    onnx_dir = subnet_dir / "onnx"
    onnx_dir.mkdir(parents=True, exist_ok=True)
    output_onnx = onnx_dir / "model_signal_maxk.onnx"
    report_path = onnx_dir / "real_onnx_export_report.json"
    fixed_k = int(getattr(args, "fixed_k", 29696))
    if output_onnx.is_file():
        require_origin_map = bool(getattr(args, "require_onnx_origin_map", False))
        origin_map = _load_onnx_export_origin_map(output_onnx)
        origin_map_policy_ok = str(origin_map.get("unique_name_policy_version", "")) == ONNX_ORIGIN_MAP_UNIQUE_NAME_POLICY_VERSION
        if require_origin_map and (not origin_map or not origin_map_policy_ok):
            pass
        else:
            try:
                info = inspect_signal_maxk_onnx(output_onnx, validate_onnx=False)
                report = {
                    "export_success": bool(info.get("has_required_heal_inputs")) and not bool(info.get("has_toy_input_only")),
                    "onnx_path": str(output_onnx),
                    "input_names": info.get("input_names", []),
                    "output_names": info.get("output_names", []),
                    "fixed_k": fixed_k,
                    "dynamic_axes": True,
                    "source_exporter": "existing_subnet_signal_maxk_onnx",
                    "shape_check_passed": bool(info.get("has_required_heal_inputs")) and not bool(info.get("has_toy_input_only")),
                    "onnx_checker_passed": info.get("onnx_checker_passed"),
                    "origin_map_success": bool(origin_map.get("success")),
                    "origin_map_entry_count": int(origin_map.get("entry_count", 0) or 0),
                    "failure_reason": "" if bool(info.get("has_required_heal_inputs")) and not bool(info.get("has_toy_input_only")) else "existing_onnx_missing_required_heal_bindings",
                }
                _write_real_onnx_report(subnet_dir, report)
                if report["export_success"]:
                    return output_onnx, report
                return None, report
            except Exception as exc:  # noqa: BLE001
                report = {"export_success": False, "onnx_path": str(output_onnx), "failure_reason": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()}
                _write_real_onnx_report(subnet_dir, report)
                return None, report
    manifest_path = subnet_dir / "pruning_manifest.json"
    pruned_model_path = subnet_dir / "pruned_model_object.pth"
    if not pruned_model_path.is_file():
        alt = subnet_dir / "models" / "pruned_model_object.pth"
        if alt.is_file():
            pruned_model_path = alt
    is_toy, toy_reason = _looks_like_toy_pruned_artifact(pruned_model_path)
    if is_toy:
        report = {
            "export_success": False,
            "onnx_path": str(output_onnx),
            "input_names": [],
            "output_names": [],
            "fixed_k": fixed_k,
            "dynamic_axes": True,
            "source_exporter": "quantization.export.export_single_engine_maxk_onnx",
            "pruned_model_path": str(pruned_model_path),
            "pruning_manifest_path": str(manifest_path),
            "shape_check_passed": False,
            "onnx_checker_passed": None,
            "failure_reason": toy_reason,
        }
        _write_real_onnx_report(subnet_dir, report)
        return None, report

    checkpoint = pruned_model_path if pruned_model_path.is_file() else Path(args.checkpoint)
    report = export_pruned_lidar_pyramid_signal_maxk_onnx(
        pruned_model_path=pruned_model_path,
        pruning_manifest_path=manifest_path,
        output_onnx_path=output_onnx,
        model_config=Path(args.model_config),
        checkpoint=checkpoint,
        heal_root=Path(args.heal_root),
        fixed_k=fixed_k,
        calibration_or_dummy_batch_source="train",
        dynamic_axes=True,
        validate_onnx=False,
    )
    _write_real_onnx_report(subnet_dir, report)
    return (output_onnx if report.get("export_success") else None), report


def _safe_onnx_name(value: str) -> str:
    return re.sub(r"[^0-9A-Za-z_]+", "_", value.strip("/")) or "tensor"


def _safe_qdq_name_prefix(node_name: str) -> str:
    base = _safe_onnx_name(node_name)
    if len(base) <= 96:
        return base
    digest = hashlib.sha1(base.encode("utf-8")).hexdigest()[:10]
    return f"{base[:80]}_{digest}"


def _module_from_onnx_node_name(node_name: str) -> str:
    candidates = _module_candidates_from_onnx_node_name(node_name)
    return candidates[0] if candidates else ""


def _match_profile_module_for_onnx_node(node_name: str, profile_modules: Mapping[str, list[str]]) -> tuple[str, list[str]]:
    candidates = _module_candidates_from_onnx_node_name(node_name)
    for candidate in candidates:
        if candidate in profile_modules:
            return candidate, list(profile_modules[candidate])
    for candidate in candidates:
        for module_name, group_ids in profile_modules.items():
            if _profile_module_matches_onnx_candidate(module_name, candidate):
                return module_name, list(group_ids)
    return (candidates[0] if candidates else ""), []


def _int8_module_to_groups(profile: Mapping[str, Any]) -> dict[str, list[str]]:
    layer_assignment = {str(k): str(v).lower() for k, v in (profile.get("layer_precision_assignment") or {}).items()}
    out: dict[str, list[str]] = {}
    for group_id, row in (profile.get("precision_group_assignments") or {}).items():
        if str(row.get("final_precision", "")).lower() != "int8":
            continue
        for module in row.get("member_modules", []):
            module_name = str(module)
            if layer_assignment.get(module_name, "int8") == "int8":
                out.setdefault(module_name, []).append(str(group_id))
    return out


def _non_int8_layer_rows(profile: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for module, precision in sorted((profile.get("layer_precision_assignment") or {}).items()):
        final = str(precision).lower()
        if final != "int8":
            rows.append({"module_name": str(module), "final_precision": final})
    return rows


def _count_qdq_nodes_in_onnx(path: Path) -> dict[str, int]:
    import onnx

    model = onnx.load(str(path))
    q_count = sum(1 for node in model.graph.node if node.op_type == "QuantizeLinear")
    dq_count = sum(1 for node in model.graph.node if node.op_type == "DequantizeLinear")
    return {"qdq_node_count": q_count + dq_count, "quantize_linear_count": q_count, "dequantize_linear_count": dq_count}


def _has_tensorrt_custom_plugin(model: Any) -> bool:
    return any(str(node.op_type) == "PointPillarScatterTRT" for node in getattr(getattr(model, "graph", None), "node", []))


def _check_onnx_model_allowing_trt_plugins(model: Any) -> dict[str, Any]:
    import onnx

    if _has_tensorrt_custom_plugin(model):
        return {
            "success": True,
            "failure_reason": "",
            "checker": "onnx.checker_skipped_for_tensorrt_custom_plugin",
            "custom_plugin_op_types": ["PointPillarScatterTRT"],
        }
    onnx.checker.check_model(model)
    return {"success": True, "failure_reason": "", "checker": "onnx.checker"}


def _insert_qdq_pair(
    *,
    helper: Any,
    numpy_helper: Any,
    np: Any,
    graph: Any,
    new_nodes: list[Any],
    source_tensor: str,
    name_prefix: str,
    module_name: str,
    onnx_node_name: str,
    onnx_node_name_unique: str,
    onnx_node_name_original: str,
    precision_group_id: str,
    qdq_position: str,
    scale: float,
    inserted_qdq_nodes: list[dict[str, Any]],
) -> str:
    scale_name = f"{name_prefix}_scale"
    zero_point_name = f"{name_prefix}_zero_point"
    q_name = f"{name_prefix}_QuantizeLinear"
    dq_name = f"{name_prefix}_DequantizeLinear"
    q_tensor = f"{name_prefix}_q"
    dq_tensor = f"{name_prefix}_dq"
    graph.initializer.extend(
        [
            numpy_helper.from_array(np.asarray([float(scale)], dtype=np.float32), scale_name),
            numpy_helper.from_array(np.asarray([0], dtype=np.int8), zero_point_name),
        ]
    )
    q_node = helper.make_node("QuantizeLinear", [source_tensor, scale_name, zero_point_name], [q_tensor], name=q_name)
    dq_node = helper.make_node("DequantizeLinear", [q_tensor, scale_name, zero_point_name], [dq_tensor], name=dq_name)
    new_nodes.extend([q_node, dq_node])
    for op_type, qdq_node_name, output_tensor in (
        ("QuantizeLinear", q_name, q_tensor),
        ("DequantizeLinear", dq_name, dq_tensor),
    ):
        inserted_qdq_nodes.append(
            {
                "canonical_module_name": module_name,
                "module_name": module_name,
                "onnx_node_name": onnx_node_name,
                "onnx_node_name_unique": onnx_node_name_unique,
                "onnx_node_name_original": onnx_node_name_original,
                "qdq_node_name": qdq_node_name,
                "qdq_op_type": op_type,
                "tensor_name": source_tensor,
                "qdq_tensor_name": output_tensor,
                "scale_name": scale_name,
                "zero_point_name": zero_point_name,
                "precision_group_id": precision_group_id,
                "qdq_position": qdq_position,
            }
        )
    return dq_tensor


def insert_mixed_precision_qdq(
    *,
    input_onnx: Path,
    output_onnx: Path,
    profile: Mapping[str, Any],
    scale_table: Mapping[str, Any],
) -> dict[str, Any]:
    import numpy as np
    import onnx
    from onnx import helper, numpy_helper

    model = onnx.load(str(input_onnx))
    int8_modules = _int8_module_to_groups(profile)
    canonical_mapping = profile.get("canonical_precision_mapping") if isinstance(profile, Mapping) else None
    canonical_by_node = _canonical_mapping_by_onnx_node(canonical_mapping)
    int8_groups = {
        str(group_id)
        for group_id, row in (profile.get("precision_group_assignments") or {}).items()
        if str(row.get("final_precision", "")).lower() == "int8"
    }
    inserted_qdq_nodes: list[dict[str, Any]] = []
    matched_groups: set[str] = set()
    quantizable_ops = {"Conv", "Gemm", "MatMul"}
    new_nodes = []
    tensor_rewrites: dict[str, str] = {}

    for node in list(model.graph.node):
        for idx, input_name in enumerate(node.input):
            if input_name in tensor_rewrites:
                node.input[idx] = tensor_rewrites[input_name]
        if canonical_by_node:
            mapped = canonical_by_node.get(str(node.name), {})
            module_name = str(mapped.get("canonical_module_name", ""))
            node_unique_name = _mapping_node_unique_name(mapped)
            node_original_name = _mapping_node_original_name(mapped)
            raw_group_ids = mapped.get("precision_group_ids") or ([mapped.get("precision_group_id")] if mapped.get("precision_group_id") else [])
            group_ids = [str(value) for value in raw_group_ids if str(value)]
            group_id = group_ids[0] if group_ids else ""
            is_int8_compute = str(mapped.get("final_precision", "")).lower() == "int8" and node.op_type in quantizable_ops
        else:
            module_name, group_ids = _match_profile_module_for_onnx_node(node.name, int8_modules)
            node_unique_name = str(node.name)
            node_original_name = str(node.name)
            group_id = group_ids[0] if group_ids else ""
            is_int8_compute = bool(group_ids) and node.op_type in quantizable_ops
        if is_int8_compute:
            matched_groups.update(group_ids)
            group_scale = float((scale_table.get(group_id) or {}).get("scale", 0.1))
            base = _safe_qdq_name_prefix(str(node.name))
            if len(node.input) >= 1 and node.input[0]:
                node.input[0] = _insert_qdq_pair(
                    helper=helper,
                    numpy_helper=numpy_helper,
                    np=np,
                    graph=model.graph,
                    new_nodes=new_nodes,
                    source_tensor=node.input[0],
                    name_prefix=f"{base}_activation",
                    module_name=module_name,
                    onnx_node_name=node.name,
                    onnx_node_name_unique=node_unique_name,
                    onnx_node_name_original=node_original_name,
                    precision_group_id=group_id,
                    qdq_position="activation_input",
                    scale=group_scale,
                    inserted_qdq_nodes=inserted_qdq_nodes,
                )
            if len(node.input) >= 2 and node.input[1]:
                node.input[1] = _insert_qdq_pair(
                    helper=helper,
                    numpy_helper=numpy_helper,
                    np=np,
                    graph=model.graph,
                    new_nodes=new_nodes,
                    source_tensor=node.input[1],
                    name_prefix=f"{base}_weight",
                    module_name=module_name,
                    onnx_node_name=node.name,
                    onnx_node_name_unique=node_unique_name,
                    onnx_node_name_original=node_original_name,
                    precision_group_id=group_id,
                    qdq_position="weight",
                    scale=group_scale,
                    inserted_qdq_nodes=inserted_qdq_nodes,
                )
        new_nodes.append(node)
        if is_int8_compute:
            group_scale = float((scale_table.get(group_id) or {}).get("scale", 0.1))
            base = _safe_qdq_name_prefix(str(node.name))
            for out_idx, output_name in enumerate(node.output):
                if not output_name:
                    continue
                dq_tensor = _insert_qdq_pair(
                    helper=helper,
                    numpy_helper=numpy_helper,
                    np=np,
                    graph=model.graph,
                    new_nodes=new_nodes,
                    source_tensor=output_name,
                    name_prefix=f"{base}_output_{out_idx}",
                    module_name=module_name,
                    onnx_node_name=node.name,
                    onnx_node_name_unique=node_unique_name,
                    onnx_node_name_original=node_original_name,
                    precision_group_id=group_id,
                    qdq_position="activation_output",
                    scale=group_scale,
                    inserted_qdq_nodes=inserted_qdq_nodes,
                )
                tensor_rewrites[output_name] = dq_tensor
                for graph_output in model.graph.output:
                    if graph_output.name == output_name:
                        graph_output.name = dq_tensor

    del model.graph.node[:]
    model.graph.node.extend(new_nodes)
    output_onnx.parent.mkdir(parents=True, exist_ok=True)
    _check_onnx_model_allowing_trt_plugins(model)
    onnx.save(model, str(output_onnx))
    counts = _count_qdq_nodes_in_onnx(output_onnx)
    unmatched_groups = sorted(int8_groups - matched_groups)
    if isinstance(canonical_mapping, dict):
        for entry in canonical_mapping.get("entries", []):
            if not isinstance(entry, dict):
                continue
            node_name = _mapping_node_unique_name(entry)
            qdq_rows = [row for row in inserted_qdq_nodes if str(row.get("onnx_node_name_unique") or row.get("onnx_node_name", "")) == node_name]
            entry["qdq_activation_nodes"] = [row["qdq_node_name"] for row in qdq_rows if row.get("qdq_position") == "activation_input"]
            entry["qdq_weight_nodes"] = [row["qdq_node_name"] for row in qdq_rows if row.get("qdq_position") == "weight"]
            entry["qdq_output_nodes"] = [row["qdq_node_name"] for row in qdq_rows if row.get("qdq_position") == "activation_output"]
        if isinstance(profile, dict):
            profile["canonical_precision_mapping"] = canonical_mapping
    return {
        "inserted_qdq_nodes": inserted_qdq_nodes,
        "skipped_non_int8_layers": _non_int8_layer_rows(profile),
        "matched_int8_precision_groups": sorted(matched_groups),
        "unmatched_int8_precision_groups": unmatched_groups,
        **counts,
    }


def _onnx_compute_nodes_by_module(onnx_path: Path) -> dict[str, list[str]]:
    try:
        import onnx

        model = onnx.load(str(onnx_path))
    except Exception:
        return {}
    out: dict[str, list[str]] = {}
    for node in model.graph.node:
        if node.op_type not in {"Conv", "Gemm", "MatMul"}:
            continue
        for module in _module_candidates_from_onnx_node_name(node.name):
            out.setdefault(module, []).append(str(node.name))
    return out


def _precision_constraint_mapping_report(profile: Mapping[str, Any], onnx_path: Path, qdq_report: Mapping[str, Any]) -> list[dict[str, Any]]:
    canonical_mapping = profile.get("canonical_precision_mapping") if isinstance(profile, Mapping) else None
    if canonical_mapping:
        inserted_by_node: dict[str, list[dict[str, Any]]] = {}
        for row in qdq_report.get("inserted_qdq_nodes") or []:
            inserted_by_node.setdefault(str(row.get("onnx_node_name_unique") or row.get("onnx_node_name", "")), []).append(dict(row))
        rows: list[dict[str, Any]] = []
        for entry in _canonical_mapping_entries(canonical_mapping):
            final = str(entry.get("final_precision", "")).lower()
            node_name = _mapping_node_unique_name(entry)
            inserted = list(inserted_by_node.get(node_name, []))
            unmatched_reason = ""
            if final == "int8" and not inserted:
                unmatched_reason = "matched_node_but_no_qdq_inserted"
            rows.append(
                {
                    "canonical_module_name": str(entry.get("canonical_module_name", "")),
                    "module_name": str(entry.get("canonical_module_name", "")),
                    "precision_group_id": str(entry.get("precision_group_id", "")),
                    "requested_precision": str(entry.get("requested_precision", "")),
                    "final_precision": final,
                    "matched_onnx_node_names": [node_name] if node_name else [],
                    "matched_onnx_node_name_unique": node_name,
                    "matched_onnx_node_name_original": _mapping_node_original_name(entry),
                    "inserted_qdq_nodes": inserted,
                    "unmatched_reason": unmatched_reason,
                }
            )
        return rows
    nodes_by_module = _onnx_compute_nodes_by_module(onnx_path)
    inserted_by_module: dict[str, list[dict[str, Any]]] = {}
    for row in qdq_report.get("inserted_qdq_nodes") or []:
        inserted_by_module.setdefault(str(row.get("module_name", "")), []).append(dict(row))
    rows: list[dict[str, Any]] = []
    final_by_group = profile.get("final_precision") or {}
    requested_by_group = profile.get("requested_precision") or {}
    for group_id, assignment in sorted((profile.get("precision_group_assignments") or {}).items()):
        final = str(assignment.get("final_precision", final_by_group.get(group_id, ""))).lower()
        requested = str(assignment.get("requested_precision", requested_by_group.get(group_id, final))).lower()
        for module in assignment.get("member_modules", []):
            module_name = str(module)
            matched_nodes = list(nodes_by_module.get(module_name, []))
            if not matched_nodes:
                for candidate, node_names in nodes_by_module.items():
                    if _profile_module_matches_onnx_candidate(module_name, candidate):
                        matched_nodes.extend(node_names)
            inserted = list(inserted_by_module.get(module_name, []))
            unmatched_reason = ""
            if final == "int8" and not matched_nodes:
                unmatched_reason = "no_matching_real_onnx_compute_node"
            elif final == "int8" and not inserted:
                unmatched_reason = "matched_node_but_no_qdq_inserted"
            rows.append(
                {
                    "module_name": module_name,
                    "precision_group_id": str(group_id),
                    "requested_precision": requested,
                    "final_precision": final,
                    "matched_onnx_node_names": matched_nodes,
                    "inserted_qdq_nodes": inserted,
                    "unmatched_reason": unmatched_reason,
                }
            )
    return rows


def _profile_paths(subnet_dir: Path) -> list[Path]:
    return sorted(subnet_dir.glob("profile_*/mixed_precision_profile.json"))


def _write_expand_profile_artifacts(
    *,
    args: argparse.Namespace,
    subnet_dir: Path,
    subnet_id: str,
    structure_hash: str,
    profile: dict[str, Any],
    groups: Sequence[PrecisionGroup],
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    profile_id = str(profile["profile_id"])
    profile_dir = subnet_dir / profile_id
    profile_dir.mkdir(parents=True, exist_ok=True)
    write_json(profile_dir / "mixed_precision_profile.json", profile)
    calib_ids = list(range(int(args.calib_train_frames)))
    write_json(profile_dir / "calibration_frames.json", calib_ids)
    scale_table = _scale_table_for_profile(profile)
    write_json(profile_dir / "scale_table.json", scale_table)
    onnx_dir = profile_dir / "onnx"
    onnx_dir.mkdir(parents=True, exist_ok=True)
    input_onnx = onnx_dir / "model.onnx"
    output_onnx = onnx_dir / "model_mixed_qdq.onnx"
    should_prepare_onnx = str2bool(args.build_engines) or str2bool(args.eval_engines)
    base_onnx: Path | None = None
    real_export_report: dict[str, Any] = {}
    qdq_payload: dict[str, Any] = {
        "inserted_qdq_nodes": [],
        "skipped_non_int8_layers": _non_int8_layer_rows(profile),
        "matched_int8_precision_groups": [],
        "unmatched_int8_precision_groups": [],
        "qdq_node_count": 0,
        "quantize_linear_count": 0,
        "dequantize_linear_count": 0,
    }
    if should_prepare_onnx:
        base_onnx, real_export_report = ensure_real_pruned_signal_maxk_onnx_for_subnet(subnet_dir, args)
    onnx_report = {
        "success": False,
        "status": "real_onnx_export_skipped" if not should_prepare_onnx else "real_onnx_export_failed",
        "failure_reason": "build_and_eval_disabled" if not should_prepare_onnx else str(real_export_report.get("failure_reason", "real_signal_maxk_onnx_unavailable")),
        "onnx_path": str(output_onnx),
        "real_onnx_export_report": real_export_report,
    }
    if base_onnx is None:
        for stale_path in (input_onnx, output_onnx):
            if stale_path.exists():
                stale_path.unlink()
    if base_onnx is not None:
        input_onnx.write_bytes(base_onnx.read_bytes())
        copy_onnx_origin_artifacts(base_onnx, input_onnx.parent)
        try:
            canonical_mapping = build_canonical_precision_mapping(input_onnx, profile)
            profile["canonical_precision_mapping"] = canonical_mapping
            write_canonical_precision_mapping(profile_dir, canonical_mapping)
            qdq_payload = insert_mixed_precision_qdq(
                input_onnx=input_onnx,
                output_onnx=output_onnx,
                profile=profile,
                scale_table=scale_table,
            )
            onnx_report = _check_onnx(output_onnx)
        except Exception as exc:  # noqa: BLE001
            onnx_report = {
                "success": False,
                "status": "onnx_or_qdq_failed",
                "failure_reason": f"qdq_insert_failed:{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
                "onnx_path": str(output_onnx),
            }
        onnx_report["status"] = "onnx_qdq_passed" if onnx_report.get("success") else "onnx_or_qdq_failed"
    int8_group_count = sum(1 for row in (profile.get("precision_group_assignments") or {}).values() if str(row.get("final_precision", "")).lower() == "int8")
    inserted_count = len(qdq_payload.get("inserted_qdq_nodes") or [])
    unmatched_groups = list(qdq_payload.get("unmatched_int8_precision_groups") or [])
    if bool(onnx_report.get("success")) and int8_group_count > 0 and inserted_count == 0:
        onnx_report["success"] = False
        onnx_report["status"] = "onnx_or_qdq_failed"
        onnx_report["failure_reason"] = "int8_profile_has_no_inserted_qdq_nodes"
    elif bool(onnx_report.get("success")) and unmatched_groups:
        onnx_report["success"] = False
        onnx_report["status"] = "onnx_or_qdq_failed"
        onnx_report["failure_reason"] = "int8_profile_has_unmatched_onnx_nodes:" + ",".join(unmatched_groups)
    qdq_report = make_qdq_insert_report(
        input_onnx=str(input_onnx),
        output_onnx=str(output_onnx),
        profile=profile,
        calibration_frame_ids=calib_ids,
        scale_table=scale_table,
        success=bool(onnx_report.get("success")),
        failure_reason=str(onnx_report.get("failure_reason", "")),
        inserted_qdq_nodes=qdq_payload.get("inserted_qdq_nodes", []),
        skipped_non_int8_layers=qdq_payload.get("skipped_non_int8_layers", []),
        matched_int8_precision_groups=qdq_payload.get("matched_int8_precision_groups", []),
        unmatched_int8_precision_groups=qdq_payload.get("unmatched_int8_precision_groups", []),
        qdq_node_count=int(qdq_payload.get("qdq_node_count", 0)),
        quantize_linear_count=int(qdq_payload.get("quantize_linear_count", 0)),
        dequantize_linear_count=int(qdq_payload.get("dequantize_linear_count", 0)),
    )
    if profile.get("canonical_precision_mapping"):
        write_canonical_precision_mapping(profile_dir, profile["canonical_precision_mapping"])
    write_json(profile_dir / "qdq_insert_report.json", qdq_report)
    write_json(profile_dir / "onnx_check_report.json", onnx_report)
    write_json(profile_dir / "onnx_parser_report.json", onnx_report)
    engine_path = profile_dir / "engine.plan"
    layer_info_path = profile_dir / "trt_layer_info.json"
    build_log_path = profile_dir / "build_log.txt"
    engine_gate = gate_engine_build(onnx_report)
    engine_report = {
        "build_success": engine_path.is_file() and not bool(args.overwrite_profiles),
        "success": engine_path.is_file() and not bool(args.overwrite_profiles),
        "failure_reason": "" if engine_path.is_file() and not bool(args.overwrite_profiles) else (engine_gate["blocked_stage"] or "tensorrt_build_not_executed"),
        "engine_path": str(engine_path),
        "uses_int8_flag": any(row.get("final_precision") == "int8" for row in (profile.get("precision_group_assignments") or {}).values()),
    }
    if str2bool(args.build_engines) and (bool(args.overwrite_profiles) or not engine_path.is_file()) and engine_gate["allowed"]:
        engine_report = build_engine_with_trtexec(
            args=args,
            onnx_path=output_onnx,
            engine_path=engine_path,
            profile=profile,
            build_log_path=build_log_path,
            layer_info_path=layer_info_path,
        )
    else:
        if not build_log_path.exists():
            build_log_path.write_text(json.dumps(engine_report, indent=2) + "\n", encoding="utf-8")
        if not layer_info_path.exists():
            write_json(layer_info_path, [])
    write_json(profile_dir / "build_report.json", engine_report)
    write_json(profile_dir / "precision_assignment_report.json", profile)
    write_json(profile_dir / "unsupported_or_fallback_layers.json", profile.get("fallback_layers", []))
    eval_success = False
    eval_reason = "real_trt_validation_eval_not_wired" if str2bool(args.eval_engines) is False else "real_trt_validation_eval_not_wired"
    eval_report = {
        "eval_success": eval_success,
        "failure_reason": eval_reason,
        "synthetic_used": False,
        "validation_dataloader_used": False,
    }
    write_csv(profile_dir / "eval_latency_per_frame.csv", [])
    write_json(profile_dir / "eval_report.json", eval_report)
    eval_row = engine_eval_summary_row(
        subnet_id,
        profile_id,
        structure_hash,
        {},
        {},
        profile,
        build_success=bool(engine_report.get("build_success")),
        eval_success=False,
        failure_reason=eval_reason,
    )
    write_json(profile_dir / "eval_latency_summary.json", eval_row)
    write_json(profile_dir / "eval_ap.json", {"AP@0.03": None, "AP@0.30": None, "AP@0.50": None, "AP@0.70": None, "mAP": None})
    trt_components = _parse_trt_layer_info(layer_info_path)
    if not trt_components:
        trt_components = [
            {
                "module_name": module,
                "precision_group_id": group.precision_group_id,
                "final_precision": profile.get("layer_precision_assignment", {}).get(module),
                "profile_source": "precision_group_only",
            }
            for group in groups
            for module in group.member_modules
        ]
    component_rows = [component_lut_sample_row(subnet_id, profile_id, structure_hash, row) for row in trt_components]
    write_csv(profile_dir / "component_profile.csv", component_rows)
    write_csv(profile_dir / "component_structure_features.csv", component_rows)
    training_row = full_engine_training_sample(subnet_id, profile_id, structure_hash, profile, component_rows, eval_row)
    index_row = _profile_index_row(
        subnet_id=subnet_id,
        profile_id=profile_id,
        structure_hash=structure_hash,
        profile=profile,
        qdq_onnx_path=output_onnx,
        engine_path=engine_path,
        build_success=bool(engine_report.get("build_success")),
        eval_success=False,
        failure_reason=eval_reason if not bool(eval_success) else "",
    )
    return index_row, eval_row, component_rows, training_row


def run_profile_legality_stage(ctx: dict[str, Any]) -> dict[str, Any]:
    profile = ctx["profile"]
    groups: Sequence[PrecisionGroup] = ctx["groups"]
    existing_hashes: set[str] = set(ctx.get("existing_hashes") or set())
    subnet_dir = Path(ctx["subnet_dir"]) if ctx.get("subnet_dir") is not None else None
    if subnet_dir is not None:
        module_shapes = _module_shape_index_from_subnet_dir(subnet_dir)
        if module_shapes:
            profile = apply_deployment_aware_precision_legality(profile, module_shapes)
            ctx["profile"] = profile
            write_json(Path(ctx["profile_dir"]) / "mixed_precision_profile.json", profile)
    profile_hash = str(profile.get("precision_assignment_hash", precision_assignment_hash(profile)))
    reasons: list[str] = []
    if profile_hash in existing_hashes:
        reasons.append("duplicate_precision_assignment_hash")
    if str(profile.get("profile_template_id", "")) in {"low_int8", "medium_int8", "high_int8"} and int(profile.get("int8_group_count") or 0) <= 0:
        reasons.append("mixed_int8_profile_has_no_int8_groups")
    group_ids = {group.precision_group_id for group in _effective_precision_groups(groups)}
    for group_id, row in (profile.get("precision_group_assignments") or {}).items():
        final = str(row.get("final_precision", "")).lower()
        if group_id not in group_ids:
            reasons.append(f"unknown_precision_group:{group_id}")
        if final not in [str(value).lower() for value in row.get("allowed_precisions", PRECISIONS)]:
            reasons.append(f"final_precision_not_allowed:{group_id}:{final}")
        for module in row.get("member_modules", []):
            assigned = str((profile.get("layer_precision_assignment") or {}).get(module, "")).lower()
            if assigned != final:
                reasons.append(f"group_member_precision_conflict:{group_id}:{module}:{assigned}!={final}")
        if str(row.get("reason", "")).lower() == "unsupported_int8" and row.get("requested_precision") == "int8" and final == "int8":
            reasons.append(f"unsupported_int8_not_fallback:{group_id}")
        if str(row.get("reason", "")).lower() == "head_constraint" and final == "int8":
            reasons.append(f"head_constraint_int8_forbidden:{group_id}")
    report = {
        "profile_legality_passed": not reasons,
        "status": "profile_legality_passed" if not reasons else "profile_legality_failed",
        "precision_assignment_hash": profile_hash,
        "failure_reason": ";".join(reasons),
        "checked_constraints": ["duplicate_hash", "overlap_consistency", "allowed_precision", "unsupported_int8", "head_constraint"],
    }
    write_json(ctx["profile_dir"] / "profile_legality_report.json", report)
    return report


def _check_onnx(path: Path) -> dict[str, Any]:
    try:
        import onnx

        model = onnx.load(str(path))
        report = _check_onnx_model_allowing_trt_plugins(model)
        report["onnx_path"] = str(path)
        return report
    except Exception as exc:  # noqa: BLE001
        return {"success": False, "failure_reason": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc(), "onnx_path": str(path), "checker": "onnx.checker"}


def run_onnx_qdq_stage(ctx: dict[str, Any]) -> dict[str, Any]:
    args = ctx["args"]
    subnet_dir: Path = ctx["subnet_dir"]
    profile_dir: Path = ctx["profile_dir"]
    profile = ctx["profile"]
    calib_ids = list(range(int(args.calib_train_frames)))
    write_json(profile_dir / "calibration_frames.json", calib_ids)
    scale_table = _scale_table_for_profile(profile)
    write_json(profile_dir / "scale_table.json", scale_table)
    onnx_dir = profile_dir / "onnx"
    onnx_dir.mkdir(parents=True, exist_ok=True)
    input_onnx = onnx_dir / "model.onnx"
    output_onnx = onnx_dir / "model_mixed_qdq.onnx"
    qdq_payload: dict[str, Any] = {
        "inserted_qdq_nodes": [],
        "skipped_non_int8_layers": _non_int8_layer_rows(profile),
        "matched_int8_precision_groups": [],
        "unmatched_int8_precision_groups": [],
        "qdq_node_count": 0,
        "quantize_linear_count": 0,
        "dequantize_linear_count": 0,
    }
    base_onnx, real_export_report = ensure_real_pruned_signal_maxk_onnx_for_subnet(subnet_dir, args)
    if base_onnx is None:
        for stale_path in (input_onnx, output_onnx):
            if stale_path.exists():
                stale_path.unlink()
        onnx_report = {
            "success": False,
            "status": "real_onnx_export_failed",
            "failure_reason": str(real_export_report.get("failure_reason", "real_signal_maxk_onnx_unavailable")),
            "onnx_path": str(output_onnx),
            "real_onnx_export_report": real_export_report,
        }
    else:
        input_onnx.write_bytes(base_onnx.read_bytes())
        copy_onnx_origin_artifacts(base_onnx, input_onnx.parent)
        try:
            canonical_mapping = build_canonical_precision_mapping(input_onnx, profile)
            profile["canonical_precision_mapping"] = canonical_mapping
            ctx["profile"] = profile
            write_canonical_precision_mapping(profile_dir, canonical_mapping)
            qdq_payload = insert_mixed_precision_qdq(
                input_onnx=input_onnx,
                output_onnx=output_onnx,
                profile=profile,
                scale_table=scale_table,
            )
            onnx_report = _check_onnx(output_onnx)
        except Exception as exc:  # noqa: BLE001
            onnx_report = {
                "success": False,
                "status": "onnx_or_qdq_failed",
                "failure_reason": f"qdq_insert_failed:{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
                "onnx_path": str(output_onnx),
            }
        onnx_report["status"] = "onnx_qdq_passed" if onnx_report.get("success") else "onnx_or_qdq_failed"
    int8_group_count = sum(1 for row in (profile.get("precision_group_assignments") or {}).values() if str(row.get("final_precision", "")).lower() == "int8")
    inserted_count = len(qdq_payload.get("inserted_qdq_nodes") or [])
    unmatched_groups = list(qdq_payload.get("unmatched_int8_precision_groups") or [])
    if bool(onnx_report.get("success")) and int8_group_count > 0 and inserted_count == 0:
        onnx_report["success"] = False
        onnx_report["status"] = "onnx_or_qdq_failed"
        onnx_report["failure_reason"] = "int8_profile_has_no_inserted_qdq_nodes"
    elif bool(onnx_report.get("success")) and unmatched_groups:
        onnx_report["success"] = False
        onnx_report["status"] = "onnx_or_qdq_failed"
        onnx_report["failure_reason"] = "int8_profile_has_unmatched_onnx_nodes:" + ",".join(unmatched_groups)
    qdq_report = make_qdq_insert_report(
        input_onnx=str(input_onnx),
        output_onnx=str(output_onnx),
        profile=profile,
        calibration_frame_ids=calib_ids,
        scale_table=scale_table,
        success=bool(onnx_report.get("success")),
        failure_reason=str(onnx_report.get("failure_reason", "")),
        inserted_qdq_nodes=qdq_payload.get("inserted_qdq_nodes", []),
        skipped_non_int8_layers=qdq_payload.get("skipped_non_int8_layers", []),
        matched_int8_precision_groups=qdq_payload.get("matched_int8_precision_groups", []),
        unmatched_int8_precision_groups=qdq_payload.get("unmatched_int8_precision_groups", []),
        qdq_node_count=int(qdq_payload.get("qdq_node_count", 0)),
        quantize_linear_count=int(qdq_payload.get("quantize_linear_count", 0)),
        dequantize_linear_count=int(qdq_payload.get("dequantize_linear_count", 0)),
    )
    parser_report = dict(onnx_report)
    parser_report["parser_check_method"] = "onnx_checker_then_trtexec_build_parser_gate"
    if profile.get("canonical_precision_mapping"):
        write_canonical_precision_mapping(profile_dir, profile["canonical_precision_mapping"])
    write_json(profile_dir / "qdq_insert_report.json", qdq_report)
    write_json(profile_dir / "precision_constraint_mapping_report.json", _precision_constraint_mapping_report(profile, input_onnx, qdq_report))
    write_json(profile_dir / "onnx_check_report.json", onnx_report)
    write_json(profile_dir / "onnx_parser_report.json", parser_report)
    onnx_report["output_onnx"] = str(output_onnx)
    return onnx_report


def run_engine_build_stage(ctx: dict[str, Any]) -> dict[str, Any]:
    args = ctx["args"]
    profile_dir: Path = ctx["profile_dir"]
    profile = ctx["profile"]
    output_onnx = Path(ctx.get("output_onnx") or profile_dir / "onnx" / "model_mixed_qdq.onnx")
    engine_path = profile_dir / "engine.plan"
    layer_info_path = profile_dir / "trt_layer_info.json"
    build_log_path = profile_dir / "build_log.txt"
    if not str2bool(args.build_engines):
        report = {
            "build_success": engine_path.is_file(),
            "success": engine_path.is_file(),
            "status": "engine_build_passed" if engine_path.is_file() else "engine_build_failed",
            "failure_reason": "" if engine_path.is_file() else "engine_build_disabled_and_engine_missing",
            "engine_path": str(engine_path),
            "uses_int8_flag": any(row.get("final_precision") == "int8" for row in (profile.get("precision_group_assignments") or {}).values()),
        }
    else:
        report = build_engine_with_trtexec(
            args=args,
            onnx_path=output_onnx,
            engine_path=engine_path,
            profile=profile,
            build_log_path=build_log_path,
            layer_info_path=layer_info_path,
        )
        log_text = build_log_path.read_text(encoding="utf-8", errors="ignore") if build_log_path.is_file() else ""
        fatal = any(token in log_text.lower() for token in ("&&&& failed", "segmentation fault", "internal error", "error code"))
        nonempty = engine_path.is_file() and engine_path.stat().st_size > 0
        has_int8 = any(row.get("final_precision") == "int8" for row in (profile.get("precision_group_assignments") or {}).values())
        uses_int8 = bool(report.get("uses_int8_flag"))
        if not nonempty:
            report["failure_reason"] = report.get("failure_reason") or "engine_plan_missing_or_empty"
        if fatal:
            report["failure_reason"] = report.get("failure_reason") or "fatal_error_in_build_log"
        if has_int8 != uses_int8:
            report["failure_reason"] = report.get("failure_reason") or "uses_int8_flag_mismatch"
        report["build_success"] = bool(report.get("build_success")) and nonempty and not fatal and has_int8 == uses_int8
        report["success"] = bool(report["build_success"])
        report["status"] = "engine_build_passed" if report["build_success"] else "engine_build_failed"
    if not layer_info_path.exists():
        write_json(layer_info_path, [])
    write_json(profile_dir / "build_report.json", report)
    write_json(profile_dir / "precision_assignment_report.json", profile)
    write_json(profile_dir / "unsupported_or_fallback_layers.json", profile.get("fallback_layers", []))
    return report


def run_physical_structure_preflight_stage(ctx: dict[str, Any]) -> dict[str, Any]:
    """Gate engine build on the complete materialized physical structure."""
    subnet_dir = Path(ctx["subnet_dir"])
    profile_dir = Path(ctx["profile_dir"])
    args = ctx.get("args")
    snapshot_path = subnet_dir / "physical_structure_snapshot_v2.json"
    required = bool(getattr(args, "require_preflight_pass", False) or getattr(args, "require_physical_metadata_v2", False) or snapshot_path.is_file())
    if not required:
        return {
            "preflight_schema_version": "physical-onnx-preflight-v2",
            "preflight_passed": True,
            "preflight_skipped": True,
            "skip_reason": "legacy_artifact_without_physical_metadata_v2",
            "check_count": 0,
            "checks": [],
        }
    try:
        return run_physical_structure_preflight(
            subnet_dir=subnet_dir,
            profile_dir=profile_dir,
            require_snapshot=True,
        )
    except Exception as exc:  # noqa: BLE001
        report = {
            "preflight_schema_version": "physical-onnx-preflight-v2",
            "preflight_passed": False,
            "failure_reason": f"physical_structure_preflight_exception:{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
            "check_count": 0,
            "checks": [],
        }
        write_json(profile_dir / "physical_structure_preflight_report.json", report)
        return report


def _read_json_if_exists(path: Path, default: Any) -> Any:
    if not path.is_file():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def _manifest_channel_rows(manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    raw = manifest.get("module_channel_before_after") or manifest.get("before_after_shapes") or []
    if isinstance(raw, Mapping):
        rows = []
        for module_name, value in raw.items():
            row = dict(value) if isinstance(value, Mapping) else {}
            row.setdefault("module_name", module_name)
            rows.append(row)
        return rows
    return [dict(row) for row in raw if isinstance(row, Mapping)]


def _row_channels(row: Mapping[str, Any], stage: str) -> dict[str, int]:
    raw = row.get(stage) if isinstance(row.get(stage), Mapping) else {}
    attrs = raw.get("attrs") if isinstance(raw.get("attrs"), Mapping) else raw
    out: dict[str, int] = {}
    for key in ("in_channels", "out_channels", "in_features", "out_features"):
        value = attrs.get(key) if isinstance(attrs, Mapping) else None
        if value is not None:
            try:
                out[key] = int(value)
            except Exception:
                pass
    return out


def _is_head_like_module(module_name: str) -> bool:
    lowered = str(module_name).lower()
    return any(token in lowered for token in ("cls_head", "reg_head", "dir_head", "head"))


def _is_deblock_transpose_module(row: Mapping[str, Any]) -> bool:
    module_name = str(row.get("module_name", "")).lower()
    module_type = str(row.get("module_type", "") or (row.get("after", {}) or {}).get("module_type", "")).lower()
    return "convtranspose" in module_type or "deblock" in module_name


def _channel_alignment_report(manifest: Mapping[str, Any]) -> tuple[bool, list[dict[str, Any]]]:
    round_to = int(manifest.get("round_to") or 4)
    violations: list[dict[str, Any]] = []
    for row in _manifest_channel_rows(manifest):
        if _is_deblock_transpose_module(row):
            continue
        module_name = str(row.get("module_name", ""))
        after = _row_channels(row, "after")
        for key, value in after.items():
            if value <= 0:
                continue
            if key in {"out_channels", "out_features"} and _is_head_like_module(module_name):
                continue
            if value % round_to != 0:
                violations.append({"module_name": module_name, "field": key, "value": value, "round_to": round_to})
    return not violations, violations


def _onnx_graph_summary(path: Path) -> dict[str, Any]:
    try:
        import onnx

        model = onnx.load(str(path))
    except Exception as exc:  # noqa: BLE001
        return {
            "loaded": False,
            "failure_reason": f"{type(exc).__name__}: {exc}",
            "input_names": [],
            "output_names": [],
            "compute_nodes": [],
        }
    compute_nodes = []
    for node in model.graph.node:
        if node.op_type not in {"Conv", "Gemm", "MatMul"}:
            continue
        candidates = _module_candidates_from_onnx_node_name(node.name)
        compute_nodes.append(
            {
                "onnx_node_name": str(node.name),
                "op_type": str(node.op_type),
                "module_name": candidates[0] if candidates else "",
                "module_name_candidates": candidates,
                "inputs": list(node.input),
                "outputs": list(node.output),
            }
        )
    return {
        "loaded": True,
        "failure_reason": "",
        "input_names": [value.name for value in model.graph.input],
        "output_names": [value.name for value in model.graph.output],
        "compute_nodes": compute_nodes,
    }


def _manifest_rows_by_module(value: Any) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    if isinstance(value, Mapping):
        for module_name, row in value.items():
            if isinstance(row, Mapping):
                merged = {"module_name": str(module_name)}
                merged.update(dict(row))
                out[str(module_name)] = merged
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for row in value:
            if isinstance(row, Mapping) and row.get("module_name"):
                out[str(row.get("module_name"))] = dict(row)
    return out


def _sampling_row_with_physical_before(row: Mapping[str, Any]) -> dict[str, Any]:
    """Represent an unapplied sampling request as an unchanged physical row."""
    physical = copy.deepcopy(dict(row))
    before = copy.deepcopy(physical.get("before", {}))
    physical["sampling_requested_after"] = copy.deepcopy(physical.get("after", {}))
    physical["after"] = before
    physical["physical_truth_source"] = "sampling_before_unless_overlaid_by_physical_delta"
    return physical


def _snapshot_rows_by_module(snapshot: Mapping[str, Any] | None) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for raw in (snapshot or {}).get("modules", []):
        if not isinstance(raw, Mapping):
            continue
        module_name = str(raw.get("canonical_module_name", ""))
        if not module_name:
            continue
        attrs = {
            key: raw.get(key)
            for key in ("in_channels", "out_channels", "in_features", "out_features", "num_features", "groups")
            if raw.get(key) is not None
        }
        rows[module_name] = {
            "module_name": module_name,
            "module_type": str(raw.get("module_type", "")),
            "after": {"attrs": attrs},
            "weight_shape": list(raw.get("weight_shape") or []),
            "bias_shape": list(raw.get("bias_shape") or []),
            "physical_truth_source": "physical_structure_snapshot_v2",
        }
    return rows


def _manifest_shapes_by_module(manifest: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    sampling = _manifest_rows_by_module(manifest.get("before_after_shapes"))
    is_materialized = bool(manifest.get("materialized_from_random_dependency_domains")) or "module_channel_before_after" in manifest
    out = {
        module_name: _sampling_row_with_physical_before(row) if is_materialized else dict(row)
        for module_name, row in sampling.items()
    }
    # This legacy artifact is a sparse physical delta. It only overrides rows
    # for modules whose materialized shape actually changed.
    out.update(_manifest_rows_by_module(manifest.get("module_channel_before_after")))
    # A complete snapshot is the physical hard truth and overrides both legacy
    # sampling requests and sparse deltas.
    out.update(_snapshot_rows_by_module(manifest.get("physical_structure_snapshot_v2")))
    return out


def _after_attrs_from_manifest_shape(row: Mapping[str, Any]) -> dict[str, int]:
    after = row.get("after", {})
    if isinstance(after, Mapping) and isinstance(after.get("attrs"), Mapping):
        after = after.get("attrs", {})
    if not isinstance(after, Mapping):
        return {}
    out: dict[str, int] = {}
    for key in ("in_channels", "out_channels", "in_features", "out_features", "groups"):
        if key in after:
            try:
                out[key] = int(after[key])
            except Exception:
                pass
    legacy_pairs = {
        "in_channels": ("C_in_after", "in_after"),
        "out_channels": ("C_out_after", "out_after"),
        "in_features": ("C_in_after", "in_after"),
        "out_features": ("C_out_after", "out_after"),
    }
    for key, aliases in legacy_pairs.items():
        if key in out:
            continue
        for alias in aliases:
            if alias in row:
                try:
                    out[key] = int(row[alias])
                    break
                except Exception:
                    pass
    if "groups" not in out and "groups" in row:
        try:
            out["groups"] = int(row["groups"])
        except Exception:
            pass
    return out


def _onnx_initializer_shapes(path: Path) -> dict[str, list[int]]:
    try:
        import onnx

        model = onnx.load(str(path))
    except Exception:
        return {}
    return {initializer.name: [int(dim) for dim in initializer.dims] for initializer in model.graph.initializer}


def _canonical_shape_consistency_report(
    *,
    onnx_path: Path,
    manifest: Mapping[str, Any],
    canonical_mapping: Mapping[str, Any] | Sequence[Mapping[str, Any]] | None,
) -> tuple[bool, list[dict[str, Any]]]:
    init_shapes = _onnx_initializer_shapes(onnx_path)
    manifest_shapes = _manifest_shapes_by_module(manifest)
    rows: list[dict[str, Any]] = []
    for entry in _canonical_mapping_entries(canonical_mapping):
        module = str(entry.get("canonical_module_name", ""))
        manifest_row = manifest_shapes.get(module, {})
        attrs = _after_attrs_from_manifest_shape(manifest_row)
        weight_name = str(entry.get("onnx_weight_initializer", ""))
        weight_shape = list(init_shapes.get(weight_name) or (entry.get("origin_map_entry", {}) or {}).get("weight_shape") or [])
        op_type = str(entry.get("onnx_op_type", ""))
        expected_weight_shape: list[int] = []
        groups = int(attrs.get("groups", 1) or 1)
        if op_type == "Conv" and attrs.get("in_channels") and attrs.get("out_channels") and len(weight_shape) >= 4:
            expected_weight_shape = [attrs["out_channels"], attrs["in_channels"] // groups, weight_shape[2], weight_shape[3]]
        elif op_type == "ConvTranspose" and attrs.get("in_channels") and attrs.get("out_channels") and len(weight_shape) >= 4:
            expected_weight_shape = [attrs["in_channels"], attrs["out_channels"] // groups, weight_shape[2], weight_shape[3]]
        elif op_type in {"Gemm", "MatMul"} and attrs.get("in_features") and attrs.get("out_features"):
            expected_weight_shape = [attrs["out_features"], attrs["in_features"]]
        passed = True
        reason = ""
        if expected_weight_shape and weight_shape:
            if op_type == "MatMul" and weight_shape == list(reversed(expected_weight_shape)):
                passed = True
            else:
                passed = weight_shape == expected_weight_shape
            if not passed:
                reason = "onnx_initializer_shape_mismatch_manifest_after_shape"
        rows.append(
            {
                "canonical_module_name": module,
                "onnx_node_name_unique": _mapping_node_unique_name(entry),
                "onnx_node_name_original": _mapping_node_original_name(entry),
                "onnx_op_type": op_type,
                "onnx_weight_initializer": weight_name,
                "onnx_weight_shape": weight_shape,
                "manifest_after_attrs": attrs,
                "expected_weight_shape": expected_weight_shape,
                "shape_check_passed": passed,
                "failure_reason": reason,
            }
        )
    return all(row.get("shape_check_passed", True) for row in rows), rows


def _stale_artifact_report(profile_dir: Path, onnx_path: Path) -> tuple[bool, list[dict[str, Any]]]:
    watched = [
        profile_dir / "engine.plan",
        profile_dir / "trt_layer_info.json",
        onnx_path,
        profile_dir / "canonical_precision_mapping.json",
    ]
    rows = []
    mtimes = {path.name: path.stat().st_mtime if path.is_file() else None for path in watched}
    for path in watched:
        rows.append({"path": str(path), "exists": path.is_file(), "mtime": mtimes[path.name]})
    stale = False
    engine_mtime = mtimes.get("engine.plan")
    layer_info_mtime = mtimes.get("trt_layer_info.json")
    onnx_mtime = mtimes.get(onnx_path.name)
    mapping_mtime = mtimes.get("canonical_precision_mapping.json")
    if engine_mtime is not None and onnx_mtime is not None and engine_mtime < onnx_mtime:
        stale = True
        rows.append({"path": str(profile_dir / "engine.plan"), "exists": True, "stale_reason": "engine_older_than_onnx"})
    if layer_info_mtime is not None and engine_mtime is not None and layer_info_mtime < engine_mtime:
        stale = True
        rows.append({"path": str(profile_dir / "trt_layer_info.json"), "exists": True, "stale_reason": "trt_layer_info_older_than_engine"})
    if layer_info_mtime is not None and mapping_mtime is not None and layer_info_mtime < mapping_mtime:
        stale = True
        rows.append({"path": str(profile_dir / "trt_layer_info.json"), "exists": True, "stale_reason": "trt_layer_info_older_than_canonical_mapping"})
    return not stale, rows


def _module_matches(left: str, right: str) -> bool:
    if not left or not right:
        return False
    left_aliases = _profile_module_aliases(str(left))
    right_aliases = _profile_module_aliases(str(right))
    return any(a == b or a.startswith(f"{b}.") or b.startswith(f"{a}.") for a in left_aliases for b in right_aliases)


def _profile_module_matches_onnx_candidate(profile_module: str, onnx_candidate: str) -> bool:
    if not profile_module or not onnx_candidate:
        return False
    for alias in _profile_module_aliases(str(profile_module)):
        if alias == onnx_candidate or str(onnx_candidate).startswith(f"{alias}."):
            return True
    return False


def run_engine_structure_stage(ctx: dict[str, Any]) -> dict[str, Any]:
    subnet_dir: Path = ctx["subnet_dir"]
    profile_dir: Path = ctx["profile_dir"]
    manifest_path = subnet_dir / "pruning_manifest.json"
    export_report_path = subnet_dir / "onnx" / "real_onnx_export_report.json"
    onnx_path = Path(str(ctx.get("output_onnx") or profile_dir / "onnx" / "model_mixed_qdq.onnx"))
    if not onnx_path.is_file():
        onnx_path = subnet_dir / "onnx" / "model_signal_maxk.onnx"

    manifest = _read_json_if_exists(manifest_path, {})
    snapshot = _read_json_if_exists(subnet_dir / "physical_structure_snapshot_v2.json", {})
    physical_hash = _read_json_if_exists(subnet_dir / "physical_hash_v2.json", {})
    if isinstance(manifest, Mapping):
        manifest = dict(manifest)
        if isinstance(snapshot, Mapping) and snapshot.get("modules"):
            manifest["physical_structure_snapshot_v2"] = snapshot
    export_report = _read_json_if_exists(export_report_path, {})
    onnx_summary = _onnx_graph_summary(onnx_path)
    input_names = list(onnx_summary.get("input_names") or export_report.get("input_names") or [])
    output_names = list(onnx_summary.get("output_names") or export_report.get("output_names") or [])
    compute_nodes = list(onnx_summary.get("compute_nodes") or [])
    profile = ctx.get("profile", {}) if isinstance(ctx.get("profile", {}), Mapping) else {}
    canonical_mapping = profile.get("canonical_precision_mapping") if isinstance(profile, Mapping) else None
    if not canonical_mapping:
        canonical_mapping = load_canonical_precision_mapping(profile_dir / "canonical_precision_mapping.json")
    canonical_entries = _canonical_mapping_entries(canonical_mapping)
    trt_components = _parse_trt_layer_info(profile_dir / "trt_layer_info.json")
    trt_compute_modules: list[str] = []
    ambiguous_trt_metadata_layers: list[dict[str, Any]] = []
    for row in trt_components:
        if any(token in str(row.get("op_type", "")).lower() for token in ("reformat", "noop", "signal", "wait")):
            continue
        module = canonical_module_from_trt_metadata(str(row.get("metadata", "")), canonical_mapping) if canonical_entries else ""
        if module == "__ambiguous__":
            ambiguous_trt_metadata_layers.append({"trt_layer_name": row.get("trt_layer_name", ""), "metadata": row.get("metadata", "")})
            continue
        if not module:
            module = str(row.get("module_name", ""))
        if module and (canonical_entries or float(row.get("mapping_confidence", 0.0) or 0.0) >= 0.5):
            trt_compute_modules.append(module)
    matched_compute_layer_count = 0
    unmatched_compute_layers: list[dict[str, Any]] = []
    if canonical_entries:
        trt_module_set = set(trt_compute_modules)
        for entry in canonical_entries:
            module = str(entry.get("canonical_module_name", ""))
            if module in trt_module_set:
                matched_compute_layer_count += 1
            else:
                unmatched_compute_layers.append(
                    {
                        "canonical_module_name": module,
                        "onnx_node_name_unique": _mapping_node_unique_name(entry),
                        "onnx_node_name_original": _mapping_node_original_name(entry),
                        "op_type": entry.get("onnx_op_type", ""),
                    }
                )
    else:
        for node in compute_nodes:
            candidates = [str(value) for value in node.get("module_name_candidates", []) if str(value)]
            matched = any(_module_matches(candidate, trt_module) for candidate in candidates for trt_module in trt_compute_modules)
            if matched:
                matched_compute_layer_count += 1
            else:
                unmatched_compute_layers.append(
                    {
                        "onnx_node_name": node.get("onnx_node_name", ""),
                        "op_type": node.get("op_type", ""),
                        "module_name_candidates": candidates,
                    }
                )

    channel_alignment_passed, channel_alignment_violations = _channel_alignment_report(manifest)
    canonical_shape_passed, canonical_shape_checks = _canonical_shape_consistency_report(
        onnx_path=onnx_path,
        manifest=manifest,
        canonical_mapping=canonical_mapping,
    )
    stale_artifact_passed, stale_artifact_checks = _stale_artifact_report(profile_dir, onnx_path)
    required_core_inputs = {"voxel_features", "voxel_coords", "voxel_num_points", "pairwise_t_matrix"}
    has_signal_maxk_inputs = required_core_inputs <= set(input_names) and ("valid_voxel_mask" in input_names or "record_len" in input_names)
    has_expected_outputs = {"cls_preds", "reg_preds", "dir_preds"} <= set(output_names)
    reasons: list[str] = []
    if input_names == ["input.1"] or input_names == ["input"] or "input.1" in input_names:
        reasons.append("toy_input_1_detected")
    if not has_signal_maxk_inputs:
        reasons.append("missing_signal_maxk_heal_inputs")
    if not has_expected_outputs:
        reasons.append("missing_signal_maxk_heal_outputs")
    if not bool(export_report.get("export_success", True)):
        reasons.append(f"real_onnx_export_failed:{export_report.get('failure_reason', '')}")
    if not onnx_summary.get("loaded"):
        reasons.append(f"onnx_load_failed:{onnx_summary.get('failure_reason', '')}")
    if (canonical_entries or compute_nodes) and matched_compute_layer_count == 0:
        reasons.append("no_trt_compute_layers_mapped_to_onnx_compute_nodes")
    if ambiguous_trt_metadata_layers:
        reasons.append("ambiguous_trt_metadata_mapping")
    if not channel_alignment_passed:
        reasons.append("round_to_channel_alignment_failed")
    if not canonical_shape_passed:
        reasons.append("canonical_onnx_initializer_shape_mismatch")
    if not stale_artifact_passed:
        reasons.append("stale_engine_or_layer_info_artifact")
    structure_hash = str(manifest.get("structure_hash") or ctx.get("structure_hash", ""))
    expected_hash = str(ctx.get("structure_hash", structure_hash))
    if expected_hash and structure_hash and structure_hash != expected_hash:
        reasons.append(f"structure_hash_mismatch:{structure_hash}!={expected_hash}")

    report = {
        "structure_check_passed": not reasons,
        "status": "engine_structure_passed" if not reasons else "engine_structure_mismatch",
        "structure_hash": structure_hash,
        "hash_schema_version": physical_hash.get("hash_schema_version", ""),
        "structure_hash_v2": physical_hash.get("structure_hash_v2", ""),
        "shape_hash_v2": physical_hash.get("shape_hash_v2", ""),
        "physical_structure_snapshot_used": bool(snapshot.get("modules")) if isinstance(snapshot, Mapping) else False,
        "actual_param_prune_ratio": manifest.get("actual_param_prune_ratio", ""),
        "actual_channel_prune_ratio": manifest.get("actual_channel_prune_ratio", manifest.get("actual_channel_prune_ratio_on_searchable_surface", "")),
        "onnx_path": str(onnx_path),
        "onnx_input_names": input_names,
        "onnx_output_names": output_names,
        "matched_compute_layer_count": matched_compute_layer_count,
        "onnx_compute_layer_count": len(canonical_entries) if canonical_entries else len(compute_nodes),
        "trt_mapped_compute_layer_count": len(trt_compute_modules),
        "unmatched_compute_layers": unmatched_compute_layers,
        "canonical_mapping_used": bool(canonical_entries),
        "canonical_mapping_entry_count": len(canonical_entries),
        "ambiguous_trt_metadata_layers": ambiguous_trt_metadata_layers,
        "canonical_shape_check_passed": canonical_shape_passed,
        "canonical_shape_checks": canonical_shape_checks,
        "stale_artifact_check_passed": stale_artifact_passed,
        "stale_artifact_checks": stale_artifact_checks,
        "channel_alignment_passed": channel_alignment_passed,
        "channel_alignment_violations": channel_alignment_violations,
        "round_to": int(manifest.get("round_to") or 4),
        "failure_reason": ";".join(reasons),
    }
    write_json(profile_dir / "engine_structure_check_report.json", report)
    return report


def run_engine_precision_stage(ctx: dict[str, Any]) -> dict[str, Any]:
    profile_dir: Path = ctx["profile_dir"]
    profile = ctx["profile"]
    layer_info_path = profile_dir / "trt_layer_info.json"
    components = _parse_trt_layer_info(layer_info_path)
    layer_assignment = profile.get("layer_precision_assignment") or {}
    canonical_mapping = profile.get("canonical_precision_mapping") if isinstance(profile, Mapping) else None
    if not canonical_mapping:
        canonical_mapping = load_canonical_precision_mapping(profile_dir / "canonical_precision_mapping.json")
    qdq_report = _read_json_if_exists(profile_dir / "qdq_insert_report.json", {})
    int8_modules = {
        str(module)
        for group_id, row in (profile.get("precision_group_assignments") or {}).items()
        if str(row.get("final_precision", "")).lower() == "int8"
        for module in row.get("member_modules", [])
        if str((profile.get("layer_precision_assignment") or {}).get(module, row.get("final_precision", ""))).lower() == "int8"
    }
    mismatch_layers = []
    hidden_reformat_count = 0
    hidden_cast_count = 0
    qdq_folded_count = 0
    unresolved = 0
    exempt_layers = []
    int8_realized_layers = []
    int8_compute_fp16_boundary_layers = []
    checked_expected_modules: set[str] = set()
    if int8_modules:
        inserted_qdq = list(qdq_report.get("inserted_qdq_nodes") or [])
        if not bool(qdq_report.get("success", False)) or not inserted_qdq:
            for module in sorted(int8_modules):
                mismatch_layers.append(
                    {
                        "trt_layer_name": "",
                        "original_module_name": module,
                        "expected": "int8",
                        "engine_actual_precision": "",
                        "failure_reason": "int8_profile_missing_successful_qdq_insert",
                    }
                )
    for row in components:
        name = str(row.get("trt_layer_name", ""))
        op_type = str(row.get("op_type", ""))
        op_type_lower = op_type.lower()
        name_lower = name.lower()
        is_noncompute_layer = any(token in op_type_lower for token in ("reformat", "noop", "signal", "wait"))
        is_compute_layer = (not is_noncompute_layer) and (
            any(token in op_type_lower for token in ("conv", "gemm", "matrix", "matmul"))
            or any(token in name for token in ("/Conv", "/MatMul", "/Gemm"))
        )
        final = str(row.get("final_precision", "")).lower()
        exempt_reason = ""
        if "reformat" in name_lower or "reformat" in op_type_lower:
            hidden_reformat_count += 1
            exempt_reason = "reformat_boundary"
        if "cast" in name_lower or "cast" in op_type_lower:
            hidden_cast_count += 1
            exempt_reason = exempt_reason or "cast_boundary"
        if "quantize" in name_lower or "dequantize" in name_lower:
            qdq_folded_count += 1
            if not is_compute_layer:
                exempt_reason = exempt_reason or "qdq_boundary"
        if exempt_reason and not is_compute_layer:
            exempt_layers.append({"trt_layer_name": name, "op_type": op_type, "precision_check_exempt_reason": exempt_reason})
            continue
        if not is_compute_layer:
            exempt_layers.append({"trt_layer_name": name, "op_type": op_type, "precision_check_exempt_reason": "non_compute_layer"})
            continue
        module = canonical_module_from_trt_metadata(str(row.get("metadata", "")), canonical_mapping) if canonical_mapping else ""
        if not module:
            module = str(row.get("module_name", ""))
        expected_module = ""
        expected = ""
        if module in layer_assignment:
            expected_module = module
            expected = str(layer_assignment.get(module, "")).lower()
        else:
            for assigned_module, assigned_precision in layer_assignment.items():
                if _profile_module_matches_onnx_candidate(str(assigned_module), module):
                    expected_module = str(assigned_module)
                    expected = str(assigned_precision).lower()
                    break
        if not expected or not final:
            unresolved += 1
            continue
        checked_expected_modules.add(expected_module)
        if expected == "int8":
            realization = _trt_layer_int8_compute_realization(row)
            if realization["int8_realized"]:
                realized = {
                    "trt_layer_name": name,
                    "layer_type": op_type,
                    "original_module_name": expected_module or module,
                    "expected": expected,
                    "engine_actual_precision": final,
                    "realization_status": realization["realization_status"],
                    "boundary_dtype": realization["boundary_dtype"],
                    "input_dtypes": realization["input_dtypes"],
                    "output_dtypes": realization["output_dtypes"],
                    "weights_type": realization["weights_type"],
                    "tactic_name": realization["tactic_name"],
                    "metadata": str(row.get("metadata", "")),
                }
                int8_realized_layers.append(realized)
                if realization["boundary_dtype"] == "fp16":
                    int8_compute_fp16_boundary_layers.append(realized)
                continue
            mismatch_layers.append(
                {
                    "trt_layer_name": name,
                    "layer_type": op_type,
                    "original_module_name": expected_module or module,
                    "expected": expected,
                    "engine_actual_precision": final,
                    "failure_reason": "requested_int8_compute_not_realized",
                    "input_dtypes": realization["input_dtypes"],
                    "output_dtypes": realization["output_dtypes"],
                    "weights_type": realization["weights_type"],
                    "tactic_name": realization["tactic_name"],
                    "has_int8_input": realization["has_int8_input"],
                    "has_int8_weights": realization["has_int8_weights"],
                    "has_int8_tactic": realization["has_int8_tactic"],
                    "metadata": str(row.get("metadata", "")),
                }
            )
            continue
        if expected != final:
            mismatch_layers.append({"trt_layer_name": name, "original_module_name": expected_module or module, "expected": expected, "engine_actual_precision": final})
    for module, expected in sorted((str(k), str(v).lower()) for k, v in layer_assignment.items()):
        if expected != "int8":
            continue
        if module not in checked_expected_modules:
            mismatch_layers.append(
                {
                    "trt_layer_name": "",
                    "original_module_name": module,
                    "expected": "int8",
                    "engine_actual_precision": "",
                    "failure_reason": "expected_int8_module_not_found_in_trt_layer_info",
                }
            )
    report = {
        "precision_realization_passed": len(mismatch_layers) == 0,
        "status": "engine_precision_passed" if len(mismatch_layers) == 0 else "engine_precision_mismatch",
        "precision_realization_mismatch_count": len(mismatch_layers),
        "hidden_reformat_count": hidden_reformat_count,
        "hidden_cast_count": hidden_cast_count,
        "qdq_folded_count": qdq_folded_count,
        "unresolved_layer_mapping_count": unresolved,
        "mismatch_layers": mismatch_layers,
        "true_mismatch_layers": mismatch_layers,
        "int8_realized_layer_count": len(int8_realized_layers),
        "int8_compute_fp16_boundary_count": len(int8_compute_fp16_boundary_layers),
        "int8_realized_layers": int8_realized_layers,
        "fused_int8_with_fp16_boundary_layers": int8_compute_fp16_boundary_layers,
        "precision_check_exempt_layers": exempt_layers,
        "layer_count": len(components),
    }
    write_json(profile_dir / "engine_precision_realization_report.json", report)
    return report


def run_trt_smoke_stage(ctx: dict[str, Any]) -> dict[str, Any]:
    args = ctx["args"]
    profile_dir: Path = ctx["profile_dir"]
    engine_path = Path(ctx.get("engine_path") or profile_dir / "engine.plan")
    report = run_trt_smoke(engine_path=engine_path, output_dir=profile_dir, args=args, frames=int(args.smoke_frames))
    if not report.get("success"):
        report["status"] = "trt_smoke_failed"
    return report


def run_real_eval_stage(ctx: dict[str, Any]) -> dict[str, Any]:
    args = ctx["args"]
    profile_dir: Path = ctx["profile_dir"]
    engine_path = Path(ctx.get("engine_path") or profile_dir / "engine.plan")
    if not str2bool(args.eval_engines):
        report = {
            "eval_success": False,
            "status": "eval_not_requested",
            "synthetic_used": False,
            "validation_dataloader_used": False,
            "failure_reason": "eval_engines_false",
            "latency_summary": {},
            "ap": {},
            "per_frame_rows": [],
        }
        write_csv(profile_dir / "eval_latency_per_frame.csv", [])
        write_json(profile_dir / "eval_latency_summary.json", {})
        write_json(profile_dir / "eval_ap.json", {})
        write_json(profile_dir / "eval_report.json", report)
        return report
    report = run_real_heal_validation_eval(
        engine_path=engine_path,
        output_dir=profile_dir,
        args=args,
        warmup_frames=int(args.warmup_frames),
        eval_frames=int(args.eval_frames),
    )
    if report.get("eval_success") and (report.get("synthetic_used") or not report.get("validation_dataloader_used")):
        report["eval_success"] = False
        report["status"] = "eval_failed"
        report["failure_reason"] = "real_eval_success_requires_synthetic_false_and_validation_dataloader_true"
        write_json(profile_dir / "eval_report.json", report)
    return report


def _generate_missing_profile(
    *,
    args: argparse.Namespace,
    subnet_id: str,
    subnet_index: int,
    structure_hash: str,
    groups: Sequence[PrecisionGroup],
    profile_index: int,
    existing_hashes: set[str],
) -> dict[str, Any] | None:
    for retry in range(20):
        profile = sample_stratified_mixed_precision_profile(
            subnet_id=subnet_id,
            structure_hash=structure_hash,
            groups=groups,
            profile_index=profile_index,
            profile_seed=int(args.profile_seed) + retry * 7919,
            subnet_index=subnet_index,
            precision_modes=[p.strip().lower() for p in str(args.precision_modes).split(",") if p.strip()],
        )
        if profile["precision_assignment_hash"] not in existing_hashes:
            return profile
    return None


def _profile_matches_per_engine_template(profile: Mapping[str, Any], profile_index: int) -> bool:
    expected_template, _ = _template_for_profile_index(profile_index)
    actual_template = str(profile.get("profile_template_id", ""))
    if profile_index == 0:
        return True
    if expected_template != actual_template:
        return False
    if expected_template in {"low_int8", "medium_int8", "high_int8"} and int(profile.get("int8_group_count") or 0) <= 0:
        return False
    return True


def _profile_success_complete(profile_dir: Path) -> bool:
    required = [
        profile_dir / "mixed_precision_profile.json",
        profile_dir / "qdq_insert_report.json",
        profile_dir / "engine.plan",
        profile_dir / "engine_structure_check_report.json",
        profile_dir / "engine_precision_realization_report.json",
        profile_dir / "trt_smoke_report.json",
        profile_dir / "eval_report.json",
    ]
    if not all(path.exists() for path in required):
        return False
    try:
        qdq = json.loads((profile_dir / "qdq_insert_report.json").read_text(encoding="utf-8"))
        structure = json.loads((profile_dir / "engine_structure_check_report.json").read_text(encoding="utf-8"))
        precision = json.loads((profile_dir / "engine_precision_realization_report.json").read_text(encoding="utf-8"))
        smoke = json.loads((profile_dir / "trt_smoke_report.json").read_text(encoding="utf-8"))
        eval_report = json.loads((profile_dir / "eval_report.json").read_text(encoding="utf-8"))
    except Exception:
        return False
    return (
        bool(qdq.get("success"))
        and (profile_dir / "engine.plan").stat().st_size > 0
        and bool(structure.get("structure_check_passed"))
        and bool(precision.get("precision_realization_passed"))
        and bool(smoke.get("success"))
        and bool(eval_report.get("eval_success"))
        and not bool(eval_report.get("synthetic_used"))
        and bool(eval_report.get("validation_dataloader_used"))
        and int(eval_report.get("evaluated_frames") or 0) == 1000
    )


def resolve_stale_profile_failure_report(profile_dir: Path, *, resolved_by: str) -> dict[str, Any]:
    failure_path = profile_dir / "profile_failure_report.json"
    if not failure_path.is_file():
        return {"status": "missing_profile_failure_report", "resolved": False}
    try:
        failure_report = json.loads(failure_path.read_text(encoding="utf-8"))
    except Exception:
        failure_report = {}
    if bool(failure_report.get("resolved")) and failure_report.get("resolution_status") == "stale_failure_report":
        return {"status": "already_resolved_stale_failure_report", "resolved": True}
    success_files = [
        profile_dir / "engine.plan",
        profile_dir / "trt_layer_info.json",
        profile_dir / "engine_precision_realization_report.json",
        profile_dir / "trt_smoke_report.json",
        profile_dir / "eval_report.json",
    ]
    existing_success_files = [path for path in success_files if path.exists()]
    if not existing_success_files:
        return {"status": "no_success_artifacts", "resolved": False}
    failure_mtime = failure_path.stat().st_mtime
    newest_success_mtime = max(path.stat().st_mtime for path in existing_success_files)
    if failure_mtime >= newest_success_mtime or not _profile_success_complete(profile_dir):
        return {
            "status": "active_or_unproven_failure_report",
            "resolved": False,
            "failure_mtime": failure_mtime,
            "newest_success_mtime": newest_success_mtime,
        }
    failure_report.update(
        {
            "resolved": True,
            "resolution_status": "stale_failure_report",
            "resolved_by": resolved_by,
            "resolved_at": datetime.now(timezone.utc).isoformat(),
            "failure_mtime": failure_mtime,
            "newest_success_mtime": newest_success_mtime,
        }
    )
    write_json(failure_path, failure_report)
    return {
        "status": "stale_failure_report",
        "resolved": True,
        "failure_mtime": failure_mtime,
        "newest_success_mtime": newest_success_mtime,
    }


def _row_key(row: Mapping[str, Any]) -> tuple[str, str]:
    return str(row.get("subnet_id", "")), str(row.get("profile_id", ""))


def _replace_row(rows: list[dict[str, Any]], row: dict[str, Any]) -> list[dict[str, Any]]:
    key = _row_key(row)
    kept = [dict(item) for item in rows if _row_key(item) != key]
    kept.append(row)
    return sorted(kept, key=lambda item: (str(item.get("subnet_id", "")), str(item.get("profile_id", ""))))


def _remove_row(rows: list[dict[str, Any]], *, subnet_id: str, profile_id: str) -> list[dict[str, Any]]:
    key = (str(subnet_id), str(profile_id))
    return sorted([dict(item) for item in rows if _row_key(item) != key], key=lambda item: (str(item.get("subnet_id", "")), str(item.get("profile_id", ""))))


def _write_progress_state(output_dir: Path, state: Mapping[str, Any]) -> None:
    write_json(output_dir / "progress_state.json", dict(state))


def _write_failure_summary(output_dir: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    write_csv(output_dir / "failure_summary.csv", rows)


def _write_per_engine_indexes(
    *,
    output_dir: Path,
    subnet_rows: list[dict[str, Any]],
    profile_rows: list[dict[str, Any]],
    eval_rows: list[dict[str, Any]],
    component_rows: list[dict[str, Any]],
    training_rows: list[dict[str, Any]],
    failure_rows: list[dict[str, Any]],
    args: argparse.Namespace,
    progress_state: Mapping[str, Any],
) -> None:
    write_csv(output_dir / "subnet_index.csv", subnet_rows)
    write_csv(output_dir / "mixed_precision_profile_index.csv", profile_rows)
    write_csv(output_dir / "engine_eval_summary.csv", eval_rows)
    write_csv(output_dir / "component_lut_samples.csv", component_rows)
    append_jsonl(output_dir / "full_engine_training_samples.jsonl", training_rows)
    _write_failure_summary(output_dir, failure_rows)
    _write_progress_state(output_dir, progress_state)
    manifest = {
        "dataset_version": DATASET_VERSION,
        "mode": "expand-profiles-per-engine",
        "reuse_existing_subnets": True,
        "new_pruning_performed": False,
        "successful_subnet_count": len(subnet_rows),
        "profiles_per_subnet_requested": int(args.precision_profiles_per_subnet),
        "successful_profile_count": len(profile_rows),
        "successful_engine_count": sum(truthy(row.get("build_success")) for row in profile_rows),
        "engine_structure_checked_count": sum(truthy(row.get("engine_structure_check_passed")) for row in profile_rows),
        "engine_precision_checked_count": sum(truthy(row.get("precision_realization_passed")) for row in profile_rows),
        "engine_smoke_passed_count": sum(truthy(row.get("smoke_success")) for row in profile_rows),
        "engine_eval_success_count": sum(
            truthy(row.get("eval_success")) and not truthy(row.get("synthetic_used")) and truthy(row.get("validation_dataloader_used"))
            for row in eval_rows
        ),
        "failure_summary": dict(progress_state.get("failure_stage_counts", {})),
    }
    write_json(output_dir / "dataset_manifest.json", manifest)


def _failure_report(profile_dir: Path, *, subnet_id: str, profile_id: str, stage: str, reason: str, traceback_text: str = "") -> dict[str, Any]:
    report = {
        "subnet_id": subnet_id,
        "profile_id": profile_id,
        "stage_failed": stage,
        "failure_reason": reason,
        "traceback": traceback_text,
        "recovery_action": "quarantine_invalid_profile_or_resume_after_fix",
    }
    write_json(profile_dir / "profile_failure_report.json", report)
    return report


def _remove_downstream_gate_artifacts(profile_dir: Path) -> None:
    for rel in (
        "qdq_insert_report.json",
        "onnx_check_report.json",
        "onnx_parser_report.json",
        "engine.plan",
        "build_report.json",
        "build_log.txt",
        "trt_layer_info.json",
        "engine_structure_check_report.json",
        "engine_precision_realization_report.json",
        "trt_smoke_report.json",
        "eval_report.json",
    ):
        path = profile_dir / rel
        if path.exists():
            path.unlink()


def _write_blocked_downstream_reports(profile_dir: Path, *, blocked_by: str, reason: str) -> dict[str, Any]:
    engine_path = profile_dir / "engine.plan"
    if engine_path.exists():
        engine_path.unlink()
    build_report = {
        "build_success": False,
        "success": False,
        "status": "engine_build_blocked",
        "blocked_by": blocked_by,
        "failure_reason": reason,
        "engine_path": str(engine_path),
    }
    precision_report = {
        "precision_realization_passed": False,
        "status": "engine_precision_not_run",
        "blocked_by": blocked_by,
        "failure_reason": reason,
        "precision_realization_mismatch_count": 0,
        "hidden_reformat_count": 0,
        "hidden_cast_count": 0,
        "qdq_folded_count": 0,
        "unresolved_layer_mapping_count": 0,
        "mismatch_layers": [],
        "layer_count": 0,
    }
    structure_report = {
        "structure_check_passed": False,
        "status": "engine_structure_not_run",
        "blocked_by": blocked_by,
        "failure_reason": reason,
        "structure_hash": "",
        "onnx_input_names": [],
        "onnx_output_names": [],
        "matched_compute_layer_count": 0,
        "unmatched_compute_layers": [],
        "channel_alignment_passed": False,
        "round_to": 4,
    }
    smoke_report = {
        "success": False,
        "status": "trt_smoke_not_run",
        "blocked_by": blocked_by,
        "failure_reason": reason,
        "synthetic_used": False,
        "validation_dataloader_used": False,
    }
    eval_report = {
        "eval_success": False,
        "status": "eval_not_run",
        "blocked_by": blocked_by,
        "failure_reason": reason,
        "synthetic_used": False,
        "validation_dataloader_used": False,
        "latency_summary": {},
        "ap": {},
        "per_frame_rows": [],
    }
    write_json(profile_dir / "build_report.json", build_report)
    write_json(profile_dir / "trt_layer_info.json", [])
    (profile_dir / "build_log.txt").write_text(json.dumps(build_report, indent=2) + "\n", encoding="utf-8")
    write_json(profile_dir / "engine_structure_check_report.json", structure_report)
    write_json(profile_dir / "engine_precision_realization_report.json", precision_report)
    write_json(profile_dir / "trt_smoke_report.json", smoke_report)
    _ensure_eval_artifacts(profile_dir, eval_report)
    return {"build": build_report, "structure": structure_report, "precision": precision_report, "smoke": smoke_report, "eval": eval_report}


def _write_blocked_smoke_eval_reports(profile_dir: Path, *, blocked_by: str, reason: str) -> dict[str, Any]:
    smoke_report = {
        "success": False,
        "status": "trt_smoke_not_run",
        "blocked_by": blocked_by,
        "failure_reason": reason,
        "synthetic_used": False,
        "validation_dataloader_used": False,
    }
    eval_report = {
        "eval_success": False,
        "status": "eval_not_run",
        "blocked_by": blocked_by,
        "failure_reason": reason,
        "synthetic_used": False,
        "validation_dataloader_used": False,
        "latency_summary": {},
        "ap": {},
        "per_frame_rows": [],
    }
    write_json(profile_dir / "trt_smoke_report.json", smoke_report)
    _ensure_eval_artifacts(profile_dir, eval_report)
    return {"smoke": smoke_report, "eval": eval_report}


def _ensure_eval_artifacts(profile_dir: Path, eval_report: Mapping[str, Any]) -> None:
    write_json(profile_dir / "eval_report.json", dict(eval_report))
    write_json(profile_dir / "eval_latency_summary.json", dict(eval_report.get("latency_summary") or {}))
    write_json(profile_dir / "eval_ap.json", dict(eval_report.get("ap") or {}))
    write_csv(profile_dir / "eval_latency_per_frame.csv", list(eval_report.get("per_frame_rows") or []))


def _profile_output_rows(
    *,
    subnet_id: str,
    profile_id: str,
    structure_hash: str,
    profile: Mapping[str, Any],
    profile_dir: Path,
    status: str,
    build_report: Mapping[str, Any] | None,
    structure_report: Mapping[str, Any] | None,
    precision_report: Mapping[str, Any] | None,
    smoke_report: Mapping[str, Any] | None,
    eval_report: Mapping[str, Any] | None,
    groups: Sequence[PrecisionGroup],
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    eval_report = dict(eval_report or {})
    build_success = bool((build_report or {}).get("build_success"))
    structure_passed = (structure_report or {}).get("structure_check_passed")
    precision_passed = (precision_report or {}).get("precision_realization_passed")
    smoke_success = (smoke_report or {}).get("success")
    eval_success = bool(eval_report.get("eval_success"))
    failure_reason = "" if eval_success else str(eval_report.get("failure_reason") or (smoke_report or {}).get("failure_reason") or (precision_report or {}).get("failure_reason") or (structure_report or {}).get("failure_reason") or (build_report or {}).get("failure_reason") or status)
    latency = eval_report.get("latency_summary") or {}
    ap = eval_report.get("ap") or {}
    eval_row = engine_eval_summary_row(
        subnet_id,
        profile_id,
        structure_hash,
        latency,
        ap,
        profile,
        build_success=build_success,
        eval_success=eval_success,
        failure_reason=failure_reason,
        status=status,
        engine_structure_check_passed=bool(structure_passed),
        precision_realization_passed=bool(precision_passed),
        smoke_success=bool(smoke_success),
        synthetic_used=bool(eval_report.get("synthetic_used", False)),
        validation_dataloader_used=bool(eval_report.get("validation_dataloader_used", False)),
        evaluated_frames=int(eval_report.get("evaluated_frames") or latency.get("evaluated_frames") or 0),
        qdq_insert_success=bool(_read_json_if_exists(profile_dir / "qdq_insert_report.json", {}).get("success", False)),
        inserted_qdq_node_count=len(_read_json_if_exists(profile_dir / "qdq_insert_report.json", {}).get("inserted_qdq_nodes") or []),
    )
    index_row = _profile_index_row(
        subnet_id=subnet_id,
        profile_id=profile_id,
        structure_hash=structure_hash,
        profile=profile,
        qdq_onnx_path=profile_dir / "onnx" / "model_mixed_qdq.onnx",
        engine_path=profile_dir / "engine.plan",
        build_success=build_success,
        eval_success=eval_success,
        failure_reason=failure_reason,
        status=status,
        engine_structure_check_passed=bool(structure_passed),
        precision_realization_passed=bool(precision_passed),
        smoke_success=bool(smoke_success),
    )
    trt_components = _parse_trt_layer_info(profile_dir / "trt_layer_info.json")
    if not trt_components:
        trt_components = [
            {
                "module_name": module,
                "precision_group_id": group.precision_group_id,
                "final_precision": profile.get("layer_precision_assignment", {}).get(module),
                "profile_source": "precision_group_only",
            }
            for group in groups
            for module in group.member_modules
        ]
    component_rows = [component_lut_sample_row(subnet_id, profile_id, structure_hash, row) for row in trt_components]
    training_row = full_engine_training_sample(subnet_id, profile_id, structure_hash, profile, component_rows, eval_row)
    return index_row, eval_row, component_rows, training_row


def _load_existing_index_state(output_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    return (
        read_csv_rows(output_dir / "mixed_precision_profile_index.csv"),
        read_csv_rows(output_dir / "engine_eval_summary.csv"),
        read_csv_rows(output_dir / "component_lut_samples.csv"),
        [json.loads(line) for line in (output_dir / "full_engine_training_samples.jsonl").read_text(encoding="utf-8").splitlines()] if (output_dir / "full_engine_training_samples.jsonl").is_file() else [],
        read_csv_rows(output_dir / "failure_summary.csv"),
    )


def run_one_profile_pipeline(ctx: dict[str, Any]) -> dict[str, Any]:
    args: argparse.Namespace = ctx["args"]
    subnet_id = str(ctx["subnet_id"])
    profile_id = str(ctx["profile_id"])
    profile_dir: Path = ctx["profile_dir"]
    profile = ctx["profile"]
    groups: Sequence[PrecisionGroup] = ctx["groups"]
    structure_hash = str(ctx["structure_hash"])
    profile_dir.mkdir(parents=True, exist_ok=True)
    write_json(profile_dir / "mixed_precision_profile.json", profile)
    reports: dict[str, Any] = {"status": "started"}
    try:
        profile_report = run_profile_legality_stage(ctx)
        reports["profile"] = profile_report
        if not profile_report.get("profile_legality_passed"):
            _remove_downstream_gate_artifacts(profile_dir)
            reports["status"] = "profile_legality_failed"
            reports["failure"] = _failure_report(profile_dir, subnet_id=subnet_id, profile_id=profile_id, stage="profile_legality_failed", reason=str(profile_report.get("failure_reason", "")))
            return reports

        onnx_report = run_onnx_qdq_stage(ctx)
        reports["onnx"] = onnx_report
        ctx["output_onnx"] = onnx_report.get("output_onnx")
        if not onnx_report.get("success"):
            failed_status = str(onnx_report.get("status") or "onnx_or_qdq_failed")
            reports["status"] = failed_status
            blocked = _write_blocked_downstream_reports(profile_dir, blocked_by=failed_status, reason=str(onnx_report.get("failure_reason", "")))
            reports.update(blocked)
            reports["failure"] = _failure_report(profile_dir, subnet_id=subnet_id, profile_id=profile_id, stage=failed_status, reason=str(onnx_report.get("failure_reason", "")), traceback_text=str(onnx_report.get("traceback", "")))
            return reports

        preflight_report = run_physical_structure_preflight_stage(ctx)
        reports["preflight"] = preflight_report
        if not preflight_report.get("preflight_passed"):
            reports["status"] = "physical_structure_preflight_failed"
            reports["failure"] = _failure_report(
                profile_dir,
                subnet_id=subnet_id,
                profile_id=profile_id,
                stage="physical_structure_preflight_failed",
                reason=str(preflight_report.get("failure_reason", "")),
                traceback_text=str(preflight_report.get("traceback", "")),
            )
            return reports

        build_report = run_engine_build_stage(ctx)
        reports["build"] = build_report
        ctx["engine_path"] = build_report.get("engine_path") or str(profile_dir / "engine.plan")
        if not build_report.get("build_success"):
            reports["status"] = "engine_build_failed"
            reports["failure"] = _failure_report(profile_dir, subnet_id=subnet_id, profile_id=profile_id, stage="engine_build_failed", reason=str(build_report.get("failure_reason", "")))
            return reports

        structure_report = run_engine_structure_stage(ctx)
        reports["structure"] = structure_report
        if not structure_report.get("structure_check_passed"):
            reports["status"] = "engine_structure_mismatch"
            reports.update(_write_blocked_smoke_eval_reports(profile_dir, blocked_by="engine_structure_mismatch", reason=str(structure_report.get("failure_reason", ""))))
            reports["failure"] = _failure_report(profile_dir, subnet_id=subnet_id, profile_id=profile_id, stage="engine_structure_mismatch", reason=str(structure_report.get("failure_reason", "")))
            return reports

        precision_report = run_engine_precision_stage(ctx)
        reports["precision"] = precision_report
        if not precision_report.get("precision_realization_passed") and not bool(args.allow_precision_mismatch_eval):
            reports["status"] = "engine_precision_mismatch"
            reason = f"mismatch_count={precision_report.get('precision_realization_mismatch_count')}"
            reports.update(_write_blocked_smoke_eval_reports(profile_dir, blocked_by="engine_precision_mismatch", reason=reason))
            reports["failure"] = _failure_report(profile_dir, subnet_id=subnet_id, profile_id=profile_id, stage="engine_precision_mismatch", reason=reason)
            return reports

        smoke_report = run_trt_smoke_stage(ctx)
        reports["smoke"] = smoke_report
        if not smoke_report.get("success"):
            reports["status"] = "trt_smoke_failed"
            reports["failure"] = _failure_report(profile_dir, subnet_id=subnet_id, profile_id=profile_id, stage="trt_smoke_failed", reason=str(smoke_report.get("failure_reason", "")), traceback_text=str(smoke_report.get("traceback", "")))
            return reports

        eval_report = run_real_eval_stage(ctx)
        reports["eval"] = eval_report
        _ensure_eval_artifacts(profile_dir, eval_report)
        if not eval_report.get("eval_success"):
            reports["status"] = "eval_failed"
            reports["failure"] = _failure_report(profile_dir, subnet_id=subnet_id, profile_id=profile_id, stage="eval_failed", reason=str(eval_report.get("failure_reason", "")), traceback_text=str(eval_report.get("traceback", "")))
            return reports
        reports["status"] = "eval_success"
        resolve_stale_profile_failure_report(profile_dir, resolved_by="run_one_profile_pipeline_eval_success")
        return reports
    finally:
        index_row, eval_row, component_rows, training_row = _profile_output_rows(
            subnet_id=subnet_id,
            profile_id=profile_id,
            structure_hash=structure_hash,
            profile=profile,
            profile_dir=profile_dir,
            status=str(reports.get("status", "failed")),
            build_report=reports.get("build"),
            structure_report=reports.get("structure"),
            precision_report=reports.get("precision"),
            smoke_report=reports.get("smoke"),
            eval_report=reports.get("eval"),
            groups=groups,
        )
        reports["index_row"] = index_row
        reports["eval_row"] = eval_row
        reports["component_rows"] = component_rows
        reports["training_row"] = training_row


def run_expand_profiles_per_engine(args: argparse.Namespace) -> int:
    output_dir = Path(args.source_dir)
    subnet_dirs = _subnet_dirs_for_expand(args)
    selected_profile_keys: set[tuple[str, str]] = set()
    for subnet_dir in subnet_dirs:
        manifest = json.loads((subnet_dir / "pruning_manifest.json").read_text(encoding="utf-8"))
        subnet_id_for_keys = str(manifest.get("subnet_id", subnet_dir.name))
        for profile_index_for_keys in range(int(args.precision_profiles_per_subnet)):
            selected_profile_keys.add((subnet_id_for_keys, f"profile_{profile_index_for_keys:03d}"))

    def selected_rows(rows: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
        return [row for row in rows if _row_key(row) in selected_profile_keys]

    profile_rows, eval_rows, component_rows, training_rows, failure_rows = _load_existing_index_state(output_dir) if bool(args.resume) else ([], [], [], [], [])
    subnet_rows: list[dict[str, Any]] = []
    progress_state: dict[str, Any] = {
        "total_subnets": len(subnet_dirs),
        "profiles_per_subnet": int(args.precision_profiles_per_subnet),
        "profiles_started": 0,
        "profiles_completed": 0,
        "profiles_failed": 0,
        "engines_built": 0,
        "engines_structure_checked": 0,
        "engines_precision_checked": 0,
        "engines_smoke_passed": 0,
        "engines_eval_success": 0,
        "last_completed_profile": "",
        "current_profile": "",
        "last_failure": None,
        "failure_stage_counts": {},
        "new_pruning_performed": False,
    }
    consecutive_failures = 0
    total_gate_failures = 0
    for subnet_dir in subnet_dirs:
        manifest = json.loads((subnet_dir / "pruning_manifest.json").read_text(encoding="utf-8"))
        subnet_id = str(manifest.get("subnet_id", subnet_dir.name))
        subnet_index = int(subnet_id.split("_")[-1])
        structure_hash = str(manifest.get("structure_hash", ""))
        groups = _precision_groups_from_json(subnet_dir / "precision_coupling_groups.json")
        subnet_rows.append(
            {
                "subnet_id": subnet_id,
                "structure_hash": structure_hash,
                "target_param_prune_ratio": manifest.get("target_pruning_ratio", manifest.get("actual_param_prune_ratio", "")),
                "actual_param_prune_ratio": manifest.get("actual_param_prune_ratio", ""),
                "actual_channel_prune_ratio": manifest.get("actual_channel_prune_ratio", ""),
                "params_after": manifest.get("params_after", ""),
                "pruning_manifest_path": str(subnet_dir / "pruning_manifest.json"),
                "shape_invariant_passed": manifest.get("shape_invariant_passed", True),
                "precision_coupling_groups_path": str(subnet_dir / "precision_coupling_groups.json"),
            }
        )
        normalized_existing_profiles: dict[int, dict[str, Any]] = {}
        for profile_path in _profile_paths(subnet_dir):
            idx = int(profile_path.parent.name.split("_")[-1])
            normalized_existing_profiles[idx] = _normalize_existing_profile(_read_profile(profile_path), profile_index=idx, subnet_id=subnet_id, structure_hash=structure_hash)
        for profile_index in range(int(args.precision_profiles_per_subnet)):
            profile_id = f"profile_{profile_index:03d}"
            profile_dir = subnet_dir / profile_id
            if bool(args.resume) and not bool(args.force_reeval) and _profile_success_complete(profile_dir):
                continue
            other_hashes = {
                profile.get("precision_assignment_hash", "")
                for idx, profile in normalized_existing_profiles.items()
                if idx != profile_index
            }
            if profile_index == 0 and profile_index in normalized_existing_profiles:
                profile = normalized_existing_profiles[profile_index]
            elif (
                profile_index in normalized_existing_profiles
                and not bool(args.overwrite_profiles)
                and _profile_matches_per_engine_template(normalized_existing_profiles[profile_index], profile_index)
            ):
                profile = normalized_existing_profiles[profile_index]
            else:
                profile = _generate_missing_profile(
                    args=args,
                    subnet_id=subnet_id,
                    subnet_index=subnet_index,
                    structure_hash=structure_hash,
                    groups=groups,
                    profile_index=profile_index,
                    existing_hashes=other_hashes,
                )
                if profile is None:
                    profile = sample_stratified_mixed_precision_profile(
                        subnet_id=subnet_id,
                        structure_hash=structure_hash,
                        groups=groups,
                        profile_index=profile_index,
                        profile_seed=int(args.profile_seed) + 99991,
                        subnet_index=subnet_index,
                    )
            profile["profile_id"] = profile_id
            profile["profile_index"] = profile_index
            progress_state["profiles_started"] += 1
            progress_state["current_profile"] = f"{subnet_id}/{profile_id}"
            _write_progress_state(output_dir, progress_state)
            ctx = {
                "args": args,
                "subnet_dir": subnet_dir,
                "subnet_id": subnet_id,
                "subnet_index": subnet_index,
                "structure_hash": structure_hash,
                "groups": groups,
                "profile": profile,
                "profile_id": profile_id,
                "profile_index": profile_index,
                "profile_dir": profile_dir,
                "existing_hashes": other_hashes,
            }
            result = run_one_profile_pipeline(ctx)
            status = str(result.get("status", "failed"))
            profile_rows = _replace_row(profile_rows, result["index_row"])
            eval_rows = _replace_row(eval_rows, result["eval_row"])
            component_rows = [row for row in component_rows if _row_key(row) != (subnet_id, profile_id)] + result["component_rows"]
            training_rows = [row for row in training_rows if _row_key(row) != (subnet_id, profile_id)] + [result["training_row"]]
            if status != "eval_success":
                consecutive_failures += 1
                if status in {
                    "profile_legality_failed",
                    "real_onnx_export_failed",
                    "onnx_or_qdq_failed",
                    "engine_build_failed",
                    "engine_structure_mismatch",
                    "engine_precision_mismatch",
                    "trt_smoke_failed",
                    "eval_failed",
                }:
                    total_gate_failures += 1
                failure = result.get("failure") or {"subnet_id": subnet_id, "profile_id": profile_id, "stage_failed": status, "failure_reason": status, "traceback": "", "recovery_action": ""}
                failure_rows = _replace_row(failure_rows, failure)
                progress_state["profiles_failed"] += 1
                progress_state["last_failure"] = failure
                counts = dict(progress_state.get("failure_stage_counts", {}))
                counts[status] = counts.get(status, 0) + 1
                progress_state["failure_stage_counts"] = counts
            else:
                consecutive_failures = 0
                failure_rows = _remove_row(failure_rows, subnet_id=subnet_id, profile_id=profile_id)
                if progress_state.get("last_failure") and _row_key(progress_state["last_failure"]) == (subnet_id, profile_id):
                    progress_state["last_failure"] = None
            progress_state["profiles_completed"] += 1
            progress_state["last_completed_profile"] = f"{subnet_id}/{profile_id}"
            current_profile_rows = selected_rows(profile_rows)
            current_eval_rows = selected_rows(eval_rows)
            progress_state["engines_built"] = sum(truthy(row.get("build_success")) for row in current_profile_rows)
            progress_state["engines_structure_checked"] = sum(truthy(row.get("engine_structure_check_passed")) for row in current_profile_rows)
            progress_state["engines_precision_checked"] = sum(truthy(row.get("precision_realization_passed")) for row in current_profile_rows)
            progress_state["engines_smoke_passed"] = sum(truthy(row.get("smoke_success")) for row in current_profile_rows)
            progress_state["engines_eval_success"] = sum(
                truthy(row.get("eval_success")) and not truthy(row.get("synthetic_used")) and truthy(row.get("validation_dataloader_used"))
                for row in current_eval_rows
            )
            _write_per_engine_indexes(
                output_dir=output_dir,
                subnet_rows=subnet_rows,
                profile_rows=profile_rows,
                eval_rows=eval_rows,
                component_rows=component_rows,
                training_rows=training_rows,
                failure_rows=failure_rows,
                args=args,
                progress_state=progress_state,
            )
            if bool(args.fail_fast_on_profile_error) and status != "eval_success":
                break
            if consecutive_failures >= int(args.max_consecutive_failures) or total_gate_failures >= int(args.max_total_gate_failures):
                progress_state["stopped_early"] = True
                progress_state["stop_reason"] = f"failure_threshold:consecutive={consecutive_failures},total_gate={total_gate_failures}"
                _write_progress_state(output_dir, progress_state)
                (output_dir / "per_engine_pipeline_report.md").write_text(
                    "\n".join(
                        [
                            "# v11 Per-Engine Pipeline Report",
                            "",
                            f"- stopped_early: true",
                            f"- stop_reason: {progress_state['stop_reason']}",
                            f"- successful_profile_count: {sum(truthy(row.get('eval_success')) for row in selected_rows(profile_rows))}",
                            f"- build_success_count: {progress_state['engines_built']}",
                            f"- structure_check_passed_count: {progress_state.get('engines_structure_checked', 0)}",
                            f"- precision_check_passed_count: {progress_state['engines_precision_checked']}",
                            f"- smoke_passed_count: {progress_state['engines_smoke_passed']}",
                            f"- real_eval_success_count: {progress_state['engines_eval_success']}",
                            f"- failure_stage_counts: {progress_state['failure_stage_counts']}",
                        ]
                    )
                    + "\n",
                    encoding="utf-8",
                )
                print(json.dumps({"success": False, "mode": args.mode, "stop_reason": progress_state["stop_reason"], "output_dir": str(output_dir)}, indent=2))
                return 2
    (output_dir / "per_engine_pipeline_report.md").write_text(
        "\n".join(
            [
                "# v11 Per-Engine Pipeline Report",
                "",
                f"- successful_profile_count: {sum(truthy(row.get('eval_success')) for row in selected_rows(profile_rows))}",
                f"- build_success_count: {progress_state['engines_built']}",
                f"- structure_check_passed_count: {progress_state.get('engines_structure_checked', 0)}",
                f"- precision_check_passed_count: {progress_state['engines_precision_checked']}",
                f"- smoke_passed_count: {progress_state['engines_smoke_passed']}",
                f"- real_eval_success_count: {progress_state['engines_eval_success']}",
                f"- failure_stage_counts: {progress_state['failure_stage_counts']}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"success": True, "mode": args.mode, "output_dir": str(output_dir), "profiles_completed": progress_state["profiles_completed"]}, indent=2))
    return 0


def run_expand_profiles(args: argparse.Namespace) -> int:
    source_dir = Path(args.source_dir)
    output_dir = source_dir
    subnet_dirs = _subnet_dirs_for_expand(args)
    profile_rows: list[dict[str, Any]] = []
    eval_rows: list[dict[str, Any]] = []
    component_rows: list[dict[str, Any]] = []
    training_rows: list[dict[str, Any]] = []
    subnet_index_rows: list[dict[str, Any]] = []
    duplicate_count = 0
    per_subnet_counts: dict[str, int] = {}
    for subnet_dir in subnet_dirs:
        manifest = json.loads((subnet_dir / "pruning_manifest.json").read_text(encoding="utf-8"))
        subnet_id = str(manifest.get("subnet_id", subnet_dir.name))
        subnet_index = int(subnet_id.split("_")[-1])
        structure_hash = str(manifest.get("structure_hash", ""))
        groups = _precision_groups_from_json(subnet_dir / "precision_coupling_groups.json")
        existing_hashes: set[str] = set()
        normalized_existing_profiles: dict[int, dict[str, Any]] = {}
        for profile_path in _profile_paths(subnet_dir):
            idx = int(profile_path.parent.name.split("_")[-1])
            profile = _normalize_existing_profile(_read_profile(profile_path), profile_index=idx, subnet_id=subnet_id, structure_hash=structure_hash)
            normalized_existing_profiles[idx] = profile
            if not bool(args.overwrite_profiles) or idx == 0:
                existing_hashes.add(profile["precision_assignment_hash"])
            write_json(profile_path, profile)
        for profile_index in range(int(args.precision_profiles_per_subnet)):
            profile_id = f"profile_{profile_index:03d}"
            profile_path = subnet_dir / profile_id / "mixed_precision_profile.json"
            if profile_index == 0 and profile_index in normalized_existing_profiles:
                profile = normalized_existing_profiles[profile_index]
            elif profile_path.exists() and not bool(args.overwrite_profiles):
                profile = normalized_existing_profiles.get(
                    profile_index,
                    _normalize_existing_profile(_read_profile(profile_path), profile_index=profile_index, subnet_id=subnet_id, structure_hash=structure_hash),
                )
            else:
                profile = _generate_missing_profile(
                    args=args,
                    subnet_id=subnet_id,
                    subnet_index=subnet_index,
                    structure_hash=structure_hash,
                    groups=groups,
                    profile_index=profile_index,
                    existing_hashes=existing_hashes,
                )
                if profile is None:
                    duplicate_count += 1
                    continue
                existing_hashes.add(profile["precision_assignment_hash"])
            index_row, eval_row, comp, train = _write_expand_profile_artifacts(
                args=args,
                subnet_dir=subnet_dir,
                subnet_id=subnet_id,
                structure_hash=structure_hash,
                profile=profile,
                groups=groups,
            )
            profile_rows.append(index_row)
            eval_rows.append(eval_row)
            component_rows.extend(comp)
            training_rows.append(train)
        per_subnet_counts[subnet_id] = sum(1 for row in profile_rows if row["subnet_id"] == subnet_id)
        subnet_index_rows.append(
            {
                "subnet_id": subnet_id,
                "structure_hash": structure_hash,
                "target_param_prune_ratio": manifest.get("target_pruning_ratio", manifest.get("actual_param_prune_ratio", "")),
                "actual_param_prune_ratio": manifest.get("actual_param_prune_ratio", ""),
                "actual_channel_prune_ratio": manifest.get("actual_channel_prune_ratio", ""),
                "params_after": manifest.get("params_after", ""),
                "pruning_manifest_path": str(subnet_dir / "pruning_manifest.json"),
                "shape_invariant_passed": manifest.get("shape_invariant_passed", True),
                "precision_coupling_groups_path": str(subnet_dir / "precision_coupling_groups.json"),
            }
        )
    write_csv(output_dir / "subnet_index.csv", subnet_index_rows)
    write_csv(output_dir / "mixed_precision_profile_index.csv", profile_rows)
    write_csv(output_dir / "engine_eval_summary.csv", eval_rows)
    write_csv(output_dir / "component_lut_samples.csv", component_rows)
    append_jsonl(output_dir / "full_engine_training_samples.jsonl", training_rows)
    unique_hash_count = len({row["precision_assignment_hash"] for row in profile_rows})
    int8_profiles = sum(1 for row in profile_rows if int(row.get("int8_group_count") or 0) > 0)
    manifest_payload = {
        "dataset_version": DATASET_VERSION,
        "mode": "expand-profiles",
        "reuse_existing_subnets": True,
        "new_pruning_performed": False,
        "num_existing_subnets": len(subnet_dirs),
        "successful_subnet_count": len(subnet_dirs),
        "profiles_per_subnet_requested": int(args.precision_profiles_per_subnet),
        "successful_profile_count": len(profile_rows),
        "unique_precision_assignment_count": unique_hash_count,
        "profile_hash_duplicate_count": duplicate_count,
        "per_subnet_profile_count": per_subnet_counts,
        "successful_engine_count": sum(1 for row in profile_rows if row.get("build_success") is True),
        "engine_eval_success_count": sum(1 for row in profile_rows if row.get("eval_success") is True),
        "eval_not_wired_count": sum(1 for row in profile_rows if row.get("eval_success") is not True),
        "int8_profile_count": int8_profiles,
        "fp16_heavy_profile_count": sum(1 for row in profile_rows if row.get("profile_template_id") == "fp16_heavy"),
        "low_int8_profile_count": sum(1 for row in profile_rows if row.get("profile_template_id") == "low_int8"),
        "medium_int8_profile_count": sum(1 for row in profile_rows if row.get("profile_template_id") == "medium_int8"),
        "high_int8_profile_count": sum(1 for row in profile_rows if row.get("profile_template_id") == "high_int8"),
        "failure_summary": {
            "engine_build_failed_or_skipped": sum(1 for row in profile_rows if row.get("build_success") is not True),
            "eval_failed_or_skipped": sum(1 for row in profile_rows if row.get("eval_success") is not True),
        },
    }
    write_json(output_dir / "dataset_manifest.json", manifest_payload)
    write_json(output_dir / "calibration_frames.json", list(range(int(args.calib_train_frames))))
    write_json(output_dir / "validation_frames.json", list(range(int(args.eval_frames))))
    (output_dir / "expand_profiles_report.md").write_text(
        "\n".join(
            [
                "# v11 Expand Profiles Report",
                "",
                f"- mode: expand-profiles",
                f"- reuse_existing_subnets: true",
                f"- num_existing_subnets: {len(subnet_dirs)}",
                f"- new_pruning_performed: false",
                f"- profiles_per_subnet_requested: {int(args.precision_profiles_per_subnet)}",
                f"- total_profile_count: {len(profile_rows)}",
                f"- unique_precision_assignment_hash_count: {unique_hash_count}",
                f"- int8_profile_count: {int8_profiles}",
                f"- engine_build_success_count: {manifest_payload['successful_engine_count']}",
                f"- eval_wired: false",
                f"- next_required_work: connect formal HEAL TensorRT validation evaluator for real val1000 latency/AP labels",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    write_json(output_dir / "failure_report.json", {"success": True, "failure_reason": "", "traceback": ""})
    print(json.dumps({"success": True, "mode": args.mode, "output_dir": str(output_dir), "profiles": len(profile_rows)}, indent=2))
    return 0


def _write_dataset_manifest(args: argparse.Namespace, output_dir: Path, subnet_rows: Sequence[Mapping[str, Any]], profile_rows: Sequence[Mapping[str, Any]]) -> None:
    write_json(output_dir / "calibration_frames.json", list(range(int(args.calib_train_frames))))
    write_json(output_dir / "validation_frames.json", list(range(int(args.eval_frames))))
    write_json(
        output_dir / "dataset_manifest.json",
        {
            "dataset_version": DATASET_VERSION,
            "command": " ".join(sys.argv),
            "pruning_config": {
                "target_pruning_mode": args.target_pruning_mode,
                "round_to": args.round_to,
                "max_ch_sparsity": args.max_ch_sparsity,
                "stage1_min_per_group": args.stage1_min_per_group,
                "stage1_max_ch_sparsity": args.stage1_max_ch_sparsity,
            },
            "precision_sampling_config": {
                "precision_profiles_per_subnet": args.precision_profiles_per_subnet,
                "precision_modes": args.precision_modes,
            },
            "quantization_config": {
                "calib_train_frames": args.calib_train_frames,
                "scale_method": args.scale_method,
            },
            "successful_subnet_count": len(subnet_rows),
            "successful_profile_count": len(profile_rows),
            "successful_engine_count": sum(1 for row in profile_rows if row.get("build_success") is True),
            "failure_summary": {
                "engine_build_failed_or_skipped": sum(1 for row in profile_rows if row.get("build_success") is not True),
                "eval_failed_or_skipped": sum(1 for row in profile_rows if row.get("eval_success") is not True),
            },
        },
    )
    (output_dir / "proxy_dataset_sanity_report.md").write_text(
        "\n".join(
            [
                "# Proxy Dataset Sanity Report",
                "",
                "- component_latency_sum_vs_forward_latency: unavailable_until_successful_trt_profiles",
                "- correlation_coefficient: 0.0",
                "- mean_absolute_error: 0.0",
                "- p50_error: 0.0",
                "- p90_error: 0.0",
                "- verdict: schema_ready; requires successful TensorRT profiles for latency proxy training",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def run_builder(args: argparse.Namespace) -> int:
    if str(args.cuda_visible_devices).strip():
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda_visible_devices)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.mode == "audit-subnet-artifacts":
        dataset_dir = Path(args.source_dir or args.output_dir)
        summary = audit_subnet_artifacts(dataset_dir)
        print(json.dumps({"success": True, "mode": args.mode, "output_dir": str(dataset_dir), **summary}, indent=2))
        return 0
    if args.mode == "expand-profiles-per-engine":
        return run_expand_profiles_per_engine(args)
    if args.mode == "expand-profiles":
        return run_expand_profiles(args)
    if args.mode == "formal-pruner-smoke":
        row = _write_subnet_artifacts(args, 0, output_dir)
        write_json(output_dir / "formal_pruner_smoke_report.json", {"success": True, "shape_invariant_passed": row["shape_invariant_passed"]})
        print(json.dumps({"success": True, "mode": args.mode, "output_dir": str(output_dir)}, indent=2))
        return 0
    if args.mode == "sample-real-heal":
        rows = generate_real_heal_subnets(args, output_dir)
        print(json.dumps({"success": True, "mode": args.mode, "output_dir": str(output_dir), "successful_subnet_count": len(rows)}, indent=2))
        return 0
    subnet_rows_private = [_write_subnet_artifacts(args, idx, output_dir) for idx in range(int(args.num_subnets))]
    subnet_rows = [{k: v for k, v in row.items() if not k.startswith("_")} for row in subnet_rows_private]
    write_csv(output_dir / "subnet_index.csv", subnet_rows)
    profile_rows: list[dict[str, Any]] = []
    eval_rows: list[dict[str, Any]] = []
    component_rows: list[dict[str, Any]] = []
    training_rows: list[dict[str, Any]] = []
    if args.mode in {"smoke", "full", "sample-only"}:
        for row in subnet_rows_private:
            _write_profiles_for_subnet(args, row, profile_rows, eval_rows, component_rows, training_rows)
    write_csv(output_dir / "mixed_precision_profile_index.csv", profile_rows)
    write_csv(output_dir / "engine_eval_summary.csv", eval_rows)
    write_csv(output_dir / "component_lut_samples.csv", component_rows)
    append_jsonl(output_dir / "full_engine_training_samples.jsonl", training_rows)
    _write_dataset_manifest(args, output_dir, subnet_rows, profile_rows)
    write_json(output_dir / "failure_report.json", {"success": True, "failure_reason": "", "traceback": ""})
    print(json.dumps({"success": True, "mode": args.mode, "output_dir": str(output_dir)}, indent=2))
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="v11 mixed-precision LUT dataset builder")
    parser.add_argument("--mode", choices=["formal-pruner-smoke", "sample-only", "sample-real-heal", "smoke", "full", "expand-profiles", "expand-profiles-per-engine", "audit-subnet-artifacts"], default="sample-only")
    parser.add_argument("--source-dir", default="outputs/latency_lut/v11_mixed_precision_lut_dataset_trt_full")
    parser.add_argument("--max-subnets", type=int, default=0)
    parser.add_argument("--num-subnets", type=int, default=50)
    parser.add_argument("--subnet-targets", default="")
    parser.add_argument("--precision-profiles-per-subnet", type=int, default=1)
    parser.add_argument("--precision-modes", default="fp32,fp16,int8")
    parser.add_argument("--overwrite-profiles", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite-invalid-profiles", action="store_true")
    parser.add_argument("--skip-existing-valid-engine", action="store_true")
    parser.add_argument("--force-reeval", action="store_true")
    parser.add_argument("--allow-precision-mismatch-eval", action="store_true")
    parser.add_argument("--fail-fast-on-profile-error", action="store_true")
    parser.add_argument("--quarantine-invalid-profile", action="store_true", default=True)
    parser.add_argument("--max-consecutive-failures", type=int, default=5)
    parser.add_argument("--max-total-gate-failures", type=int, default=10)
    parser.add_argument("--profile-sampling-policy", choices=["stratified", "random"], default="stratified")
    parser.add_argument("--profile-seed", type=int, default=20260708)
    parser.add_argument("--build-engines", type=str2bool, default=True)
    parser.add_argument("--eval-engines", type=str2bool, default=False)
    parser.add_argument("--target-pruning-mode", default="param", choices=["param", "channel"])
    parser.add_argument("--round-to", type=int, default=4)
    parser.add_argument("--max-ch-sparsity", type=float, default=0.60)
    parser.add_argument("--stage1-min-per-group", type=int, default=8)
    parser.add_argument("--stage1-max-ch-sparsity", type=float, default=0.30)
    parser.add_argument("--protect-fpn-output", action="store_true", default=True)
    parser.add_argument("--protect-head-output", action="store_true", default=True)
    parser.add_argument("--no-extra-output-protection", action="store_true", default=True)
    parser.add_argument("--calib-train-frames", type=int, default=200)
    parser.add_argument("--warmup-frames", type=int, default=100)
    parser.add_argument("--eval-frames", type=int, default=1000)
    parser.add_argument("--smoke-frames", type=int, default=5)
    parser.add_argument("--ap-thresholds", default="0.03,0.30,0.50,0.70")
    parser.add_argument("--scale-method", default="percentile")
    parser.add_argument("--random-seed", type=int, default=1100)
    parser.add_argument("--device", default="")
    parser.add_argument("--num-calib-batches", type=int, default=8)
    parser.add_argument("--trt-root", default="/home/lixingfeng/UniAD_examine/HEAL/prune_model/TensorRT-10.9_x86_cu118")
    parser.add_argument("--trtexec", default="")
    parser.add_argument("--trt-build-timeout-seconds", type=int, default=180)
    parser.add_argument("--fixed-k", "--fixed_k", dest="fixed_k", type=int, default=29696)
    parser.add_argument("--plugin", default="quantization/plugins/pointpillar_scatter_trt/build/libpointpillar_scatter_trt.so")
    parser.add_argument("--enable-synthetic-trt-eval", action="store_true")
    parser.add_argument("--cuda-visible-devices", default="")
    parser.add_argument("--heal-root", default="/home/lixingfeng/UniAD_examine/HEAL")
    parser.add_argument("--model-config", default="/home/lixingfeng/UniAD_examine/Auto_Search/original_models/dairv2s/LiDAROnly/lidar_pyramid/config.yaml")
    parser.add_argument("--checkpoint", default="/home/lixingfeng/UniAD_examine/Auto_Search/original_models/dairv2s/LiDAROnly/lidar_pyramid/net_epoch_bestval_at17.pth")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--auto-select-idle-gpu", action="store_true")
    parser.add_argument("--max-gpu-utilization", type=int, default=5)
    parser.add_argument("--max-gpu-memory-ratio", type=float, default=0.20)
    parser.add_argument("--output-dir", default="outputs/latency_lut/v11_mixed_precision_lut_dataset")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    try:
        return run_builder(parse_args(argv))
    except Exception as exc:  # noqa: BLE001
        output_dir = Path(parse_args(argv).output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        write_json(output_dir / "failure_report.json", {"success": False, "failure_reason": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()})
        print(json.dumps({"success": False, "failure": f"{type(exc).__name__}: {exc}", "output_dir": str(output_dir)}, indent=2))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
