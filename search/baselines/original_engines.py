"""Strict original-model TensorRT baseline helpers.

The heavy export/build/evaluation orchestration lives in the lidar_pyramid
runner.  This module keeps the precision contracts pure and testable so the
search layer can drive TensorRT without modifying quantization internals.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

from quantization.config import TensorRTBuildConfig
from quantization.tensorrt.layer_info import is_weighted_compute_layer, load_layer_info, precision_name
from quantization.types import OnnxOriginMapResult, PrecisionAssignment, PrecisionProfileResult, stable_json_hash

from ..quantization_space.types import QuantizationSearchGroup


TRUSTED_EXPLICIT_QDQ_INT8_V1_MODULES = (
    "backbone_m1.resnet.layer0.0.conv1",
    "backbone_m1.resnet.layer0.0.conv2",
    "pyramid_backbone.resnet.layer0.0.conv2",
    "pyramid_backbone.resnet.layer0.1.conv2",
    "pyramid_backbone.resnet.layer0.2.conv2",
    "pyramid_backbone.resnet.layer1.0.conv2",
    "pyramid_backbone.resnet.layer1.0.conv3",
    "pyramid_backbone.resnet.layer1.1.conv2",
    "pyramid_backbone.resnet.layer1.2.conv2",
    "pyramid_backbone.resnet.layer1.3.conv2",
    "pyramid_backbone.resnet.layer1.4.conv2",
    "pyramid_backbone.resnet.layer2.0.conv2",
    "pyramid_backbone.resnet.layer2.0.conv3",
    "pyramid_backbone.resnet.layer2.1.conv2",
    "pyramid_backbone.resnet.layer2.2.conv2",
    "pyramid_backbone.resnet.layer2.3.conv2",
    "pyramid_backbone.resnet.layer2.4.conv2",
    "pyramid_backbone.resnet.layer2.5.conv2",
    "pyramid_backbone.resnet.layer2.6.conv2",
    "pyramid_backbone.resnet.layer2.7.conv2",
    "pyramid_backbone.single_head_0",
    "pyramid_backbone.single_head_1",
    "shrink_conv.layers.0.double_conv.0",
    "shrink_conv.layers.0.double_conv.2",
    "cls_head",
    "reg_head",
    "dir_head",
)

BASELINE_PRECISIONS = {
    "strict_fp32",
    "strict_fp16",
    "maximal_legal_int8",
    "matched_legacy_int8",
    "pure_strict_int8",
    "trusted_explicit_qdq_int8",
}


def _normalize_baseline(kind: str) -> str:
    value = str(kind).lower()
    if value not in BASELINE_PRECISIONS:
        raise ValueError(f"unsupported baseline precision: {kind}")
    return value


def make_baseline_trt_build_config(
    baseline: str,
    *,
    trtexec_path: str | Path | None,
    plugin_path: str | Path | None,
    plugin_boundary_dtype: str,
    shape_profiles: Mapping[str, Mapping[str, Sequence[int]]],
    workspace_mib: int = 512,
    timeout_seconds: int = 1800,
) -> TensorRTBuildConfig:
    """Return strict builder flags for one original-model baseline."""

    kind = _normalize_baseline(baseline)
    return TensorRTBuildConfig(
        trtexec_path=Path(trtexec_path) if trtexec_path is not None else None,
        plugin_path=Path(plugin_path) if plugin_path is not None else None,
        workspace_mib=int(workspace_mib),
        shape_profiles={
            str(name): {str(bound): tuple(int(dim) for dim in dims) for bound, dims in profile.items()}
            for name, profile in shape_profiles.items()
        },
        timeout_seconds=int(timeout_seconds),
        precision_constraints="none",
        enable_fp16=False,
        enable_int8=False,
        no_tf32=True,
        skip_inference=True,
        export_layer_info=True,
        strongly_typed=True,
        production_mode=True,
        plugin_boundary_dtype=str(plugin_boundary_dtype).lower(),
        policy_version=f"strict-original-{kind}-strongly-typed-explicit-qdq-v1",
    )


def _module_to_group(groups: Sequence[QuantizationSearchGroup]) -> dict[str, QuantizationSearchGroup]:
    result: dict[str, QuantizationSearchGroup] = {}
    for group in groups:
        if str(group.group_id).startswith(("search::", "pg::")):
            raise RuntimeError(f"synthetic_precision_group_forbidden:{group.group_id}")
        for module_path in group.module_paths:
            if module_path in result:
                raise RuntimeError(f"duplicate_precision_group_member:{module_path}")
            result[str(module_path)] = group
    return result


def build_baseline_precision_profile(
    baseline: str,
    *,
    origin_map: OnnxOriginMapResult,
    groups: Sequence[QuantizationSearchGroup],
) -> PrecisionProfileResult:
    """Build assignments for strict FP32/FP16 or maximal legal INT8 baselines."""

    kind = _normalize_baseline(baseline)
    module_groups = _module_to_group(groups)
    trusted_modules = set(TRUSTED_EXPLICIT_QDQ_INT8_V1_MODULES)
    if kind == "trusted_explicit_qdq_int8":
        missing_trusted = sorted(trusted_modules - {str(row.module_path) for row in origin_map.entries})
        if missing_trusted:
            raise RuntimeError(f"trusted_explicit_qdq_profile_modules_missing:{missing_trusted}")
    assignments: list[PrecisionAssignment] = []
    missing: list[str] = []
    for ordering, origin in enumerate(sorted(origin_map.entries, key=lambda row: (row.call_index, row.graph_index))):
        group = module_groups.get(str(origin.module_path))
        if group is None:
            missing.append(str(origin.module_path))
            continue
        if kind == "strict_fp32":
            requested = "fp32"
        elif kind == "strict_fp16":
            requested = "fp16"
        elif kind == "trusted_explicit_qdq_int8":
            requested = "int8" if str(origin.module_path) in trusted_modules else "fp16"
            if requested == "int8" and ("INT8" not in group.allowed_precisions or group.protected):
                raise RuntimeError(f"trusted_explicit_qdq_profile_module_not_legal:{origin.module_path}")
        else:
            requested = "int8" if "INT8" in group.allowed_precisions and not group.protected else "fp16"
        assignments.append(
            PrecisionAssignment(
                module_path=str(origin.module_path),
                precision_group=group.group_id,
                requested_precision=requested,
                ordering=ordering,
                protected_precision="fp16" if requested == "fp16" and kind in {"maximal_legal_int8", "matched_legacy_int8", "pure_strict_int8"} else "",
                fallback_reason=(
                    group.protection_reason
                    if requested == "fp16" and group.protected
                    else "trusted_profile_not_selected"
                    if requested == "fp16" and kind == "trusted_explicit_qdq_int8"
                    else ""
                ),
            )
        )
    if missing:
        raise RuntimeError(f"missing_quantization_group_member_mapping:{missing}")
    requested_int8 = sum(row.requested_precision == "int8" for row in assignments)
    return PrecisionProfileResult(
        profile_id=kind,
        assignments=assignments,
        requested_int8_count=requested_int8,
        requested_int8_ratio=requested_int8 / max(len(assignments), 1),
        policy_version=f"strict-original-{kind}-profile-v1",
        profile_hash=stable_json_hash(
            {
                "baseline": kind,
                "assignments": [row.to_dict() for row in sorted(assignments, key=lambda item: item.module_path)],
            }
        ),
    )


def summarize_layer_precisions(layer_info: str | Path | Sequence[Mapping[str, Any]] | Mapping[str, Any]) -> dict[str, int]:
    """Count realized compute precisions for weighted and control layers."""

    summary = {
        "weighted_layer_count": 0,
        "weighted_fp32_count": 0,
        "weighted_fp16_count": 0,
        "weighted_int8_count": 0,
        "weighted_unknown_count": 0,
        "int32_control_count": 0,
        "plugin_layer_count": 0,
    }
    for row in load_layer_info(layer_info):
        metadata = " ".join(str(row.get(key, "")) for key in ("Name", "LayerType", "Metadata", "type", "name")).lower()
        if "plugin" in metadata:
            summary["plugin_layer_count"] += 1
        if not is_weighted_compute_layer(row):
            if "int32" in str(row).lower() or "shape" in metadata:
                summary["int32_control_count"] += 1
            continue
        summary["weighted_layer_count"] += 1
        precision = precision_name(row)
        if precision == "fp32":
            summary["weighted_fp32_count"] += 1
        elif precision == "fp16":
            summary["weighted_fp16_count"] += 1
        elif precision == "int8":
            summary["weighted_int8_count"] += 1
        else:
            summary["weighted_unknown_count"] += 1
    return summary


def validate_baseline_layer_precisions(
    baseline: str,
    layer_info: str | Path | Sequence[Mapping[str, Any]] | Mapping[str, Any],
    *,
    canonical_precision_realization: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Fail-closed precision realization check for strict baseline reports."""

    kind = _normalize_baseline(baseline)
    summary = summarize_layer_precisions(layer_info)
    raw_engine_summary = dict(summary)
    coverage_counting_basis = "raw_engine_weighted_layers"
    if kind in {"matched_legacy_int8", "trusted_explicit_qdq_int8"} and canonical_precision_realization is not None:
        canonical = dict(canonical_precision_realization)
        summary.update(
            {
                "weighted_layer_count": int(canonical.get("realized_int8_count", 0))
                + int(canonical.get("realized_fp16_count", 0)),
                "weighted_fp32_count": 0,
                "weighted_fp16_count": int(canonical.get("realized_fp16_count", 0)),
                "weighted_int8_count": int(canonical.get("realized_int8_count", 0)),
                "weighted_unknown_count": int(canonical.get("unresolved_layer_count", 0)),
            }
        )
        coverage_counting_basis = "canonical_precision_realization"
    issues: list[str] = []
    status = kind
    if kind == "strict_fp32":
        if summary["weighted_fp16_count"]:
            issues.append("ordinary_weighted_layer_realized_fp16")
        if summary["weighted_int8_count"]:
            issues.append("ordinary_weighted_layer_realized_int8")
        if issues:
            status = "strict_fp32_failed"
    elif kind == "strict_fp16":
        if summary["weighted_int8_count"]:
            issues.append("ordinary_weighted_layer_realized_int8")
        if summary["weighted_fp32_count"]:
            issues.append("weighted_layer_fallback_fp32")
        if issues:
            status = "strict_fp16_failed"
    elif kind == "pure_strict_int8":
        if summary["weighted_fp32_count"] or summary["weighted_fp16_count"] or not summary["weighted_int8_count"]:
            issues.append("not_all_weighted_layers_realized_int8")
            status = "pure_strict_int8_failed"
    elif kind == "trusted_explicit_qdq_int8":
        if canonical_precision_realization is not None and not bool(
            canonical_precision_realization.get("passed", False)
        ):
            issues.append("trusted_explicit_qdq_canonical_precision_realization_failed")
            status = "trusted_explicit_qdq_int8_failed"
        if summary["weighted_int8_count"] != len(TRUSTED_EXPLICIT_QDQ_INT8_V1_MODULES):
            issues.append(
                f"trusted_explicit_qdq_int8_count_mismatch:{summary['weighted_int8_count']}!={len(TRUSTED_EXPLICIT_QDQ_INT8_V1_MODULES)}"
            )
            status = "trusted_explicit_qdq_int8_failed"
        if summary["weighted_fp32_count"]:
            issues.append("trusted_explicit_qdq_unexpected_fp32_fallback")
            status = "trusted_explicit_qdq_int8_failed"
    elif kind == "matched_legacy_int8":
        if canonical_precision_realization is not None and not bool(
            canonical_precision_realization.get("passed", False)
        ):
            issues.append("matched_legacy_canonical_precision_realization_failed")
            status = "matched_legacy_int8_failed"
        if summary["weighted_int8_count"] != 67 or summary["weighted_fp16_count"] != 3:
            issues.append(
                "matched_legacy_coverage_mismatch:"
                f"int8={summary['weighted_int8_count']} fp16={summary['weighted_fp16_count']}"
            )
            status = "matched_legacy_int8_failed"
        if summary["weighted_fp32_count"] or summary["weighted_unknown_count"]:
            issues.append("matched_legacy_unexpected_fp32_or_unknown")
            status = "matched_legacy_int8_failed"
    else:
        if not summary["weighted_int8_count"]:
            issues.append("no_weighted_layer_realized_int8")
            status = "maximal_legal_int8_failed"
    return {
        **summary,
        "coverage_counting_basis": coverage_counting_basis,
        "raw_engine_weighted_summary": raw_engine_summary,
        "baseline": kind,
        "status": status,
        "passed": not issues,
        "issues": issues,
    }
