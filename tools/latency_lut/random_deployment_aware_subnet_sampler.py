#!/usr/bin/env python3
"""Deployment-aware random subnet sampler dry-run artifacts.

This sampler intentionally does not read Taylor importance scores, does not
physically prune a model, and does not build TensorRT engines. It produces
auditable dry-run subnet manifests for validating random structure diversity
and deployment channel constraints before re-running expensive LUT generation.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import statistics
from pathlib import Path
from typing import Any, Mapping, Sequence


RANDOM_SAMPLING_METHOD = "deployment_aware_random_without_taylor_ranking"
DEFAULT_OLD_V11_SUBNET_ROOT = Path("outputs/latency_lut/v11_mixed_precision_lut_dataset_trt_full/subnets")
DEBLOCK_OUTPUT_PROTECTION_REASON = "protected_convtranspose_deblock_or_fpn_output_contract"


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            key = str(key)
            if key not in seen:
                seen.add(key)
                fields.append(key)
    if not fields:
        fields = ["empty"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            ready: dict[str, Any] = {}
            for key, value in row.items():
                ready[str(key)] = json.dumps(value, sort_keys=True) if isinstance(value, (dict, list, tuple)) else value
            writer.writerow(ready)


def _stable_hash(payload: Any) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def _as_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _parse_safe_set(raw: str | Sequence[int] | set[int]) -> set[int]:
    if isinstance(raw, str):
        return {int(item.strip()) for item in raw.split(",") if item.strip()}
    return {int(item) for item in raw}


def parse_global_target_prune_bins(raw: str | Sequence[tuple[float, float]] | None) -> list[tuple[float, float]]:
    if raw is None or raw == "":
        return [(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8)]
    if isinstance(raw, str):
        bins: list[tuple[float, float]] = []
        for item in raw.split(","):
            if not item.strip():
                continue
            left, right = item.split(":", 1)
            bins.append((float(left), float(right)))
    else:
        bins = [(float(left), float(right)) for left, right in raw]
    for left, right in bins:
        if not 0.0 <= left <= right <= 1.0:
            raise ValueError(f"invalid global target prune bin: {left}:{right}")
    if not bins:
        raise ValueError("global_target_prune_bins must not be empty")
    return bins


def _bin_label(bin_range: tuple[float, float]) -> str:
    return f"{bin_range[0]:.2f}:{bin_range[1]:.2f}"


def _subnet_bin_schedule(num_subnets: int, bins: Sequence[tuple[float, float]], subnets_per_bin: int | None = None) -> list[tuple[float, float]]:
    if subnets_per_bin is not None and int(subnets_per_bin) > 0:
        schedule = [bin_range for bin_range in bins for _ in range(int(subnets_per_bin))]
        return schedule[: int(num_subnets)]
    base = int(num_subnets) // len(bins)
    remainder = int(num_subnets) % len(bins)
    schedule: list[tuple[float, float]] = []
    for idx, bin_range in enumerate(bins):
        count = base + (1 if idx < remainder else 0)
        schedule.extend([bin_range] * count)
    return schedule[: int(num_subnets)]


def _round_down_to_multiple(value: int, multiple: int) -> int:
    multiple = max(int(multiple), 1)
    return max(multiple, (int(value) // multiple) * multiple)


def _round_up_to_multiple(value: int, multiple: int) -> int:
    multiple = max(int(multiple), 1)
    return ((int(value) + multiple - 1) // multiple) * multiple


def _target_after_channels(
    before_channels: int,
    *,
    target_prune_ratio: float,
    min_channel_keep_ratio: float,
    round_to: int,
) -> tuple[int, float]:
    keep_floor = float(min_channel_keep_ratio)
    target_keep_ratio = max(keep_floor, min(1.0, 1.0 - float(target_prune_ratio)))
    raw = max(1, int(round(float(before_channels) * target_keep_ratio)))
    min_channels = _round_up_to_multiple(max(1, int(round(float(before_channels) * keep_floor))), round_to)
    min_channels = min(int(before_channels), max(round_to, min_channels))
    after = _round_down_to_multiple(raw, round_to)
    after = min(int(before_channels), max(min_channels, after))
    return after, target_keep_ratio


def _nearest_safe_per_group(
    target: int,
    original: int,
    safe_per_group: set[int],
    *,
    max_channel_prune_ratio: float,
    min_channel_keep_ratio: float,
) -> tuple[int, str]:
    keep_floor = max(float(min_channel_keep_ratio), 1.0 - float(max_channel_prune_ratio))
    min_allowed = int(round(float(original) * keep_floor))
    candidates = [value for value in sorted(safe_per_group) if min_allowed <= value <= original]
    if not candidates:
        return original, "skipped_no_safe_per_group_intersection"
    # Tie-break toward the larger keep count to avoid over-pruning near tactic boundaries.
    selected = sorted(candidates, key=lambda value: (abs(value - target), -value))[0]
    return selected, "snapped_to_safe_per_group" if selected != target else "already_safe_per_group"


def _normalize_layer_spec(row: Mapping[str, Any]) -> dict[str, Any] | None:
    before = row.get("before") if isinstance(row.get("before"), Mapping) else {}
    after = row.get("after") if isinstance(row.get("after"), Mapping) else {}
    attrs = row.get("attrs") if isinstance(row.get("attrs"), Mapping) else {}
    before_attrs = before.get("attrs") if isinstance(before.get("attrs"), Mapping) else before
    after_attrs = after.get("attrs") if isinstance(after.get("attrs"), Mapping) else after
    module_name = str(row.get("module_name") or row.get("name") or "")
    if not module_name:
        return None
    module_type = str(row.get("module_type") or row.get("type") or before.get("module_type") or after.get("module_type") or "Conv2d")
    groups = _as_int(row.get("groups", after_attrs.get("groups", before_attrs.get("groups", attrs.get("groups", 1)))), 1)
    before_in = _as_int(
        row.get("in_channels", row.get("before_in", before_attrs.get("in_channels", before_attrs.get("out_channels", 0)))),
        0,
    )
    before_out = _as_int(
        row.get("out_channels", row.get("before_out", before_attrs.get("out_channels", after_attrs.get("out_channels", 0)))),
        0,
    )
    if before_in <= 0 or before_out <= 0:
        return None
    source_before = {
        "in_channels": _as_int(before_attrs.get("in_channels", before_in), before_in),
        "out_channels": _as_int(before_attrs.get("out_channels", before_out), before_out),
        "groups": _as_int(before_attrs.get("groups", groups), groups),
    }
    source_after = {
        "in_channels": _as_int(after_attrs.get("in_channels", before_in), before_in),
        "out_channels": _as_int(after_attrs.get("out_channels", before_out), before_out),
        "groups": _as_int(after_attrs.get("groups", groups), groups),
    }
    before_params = before.get("params") if isinstance(before.get("params"), Mapping) else {}
    weight_shape = before_params.get("weight") if isinstance(before_params, Mapping) else None
    spec = {
        "module_name": module_name,
        "module_type": module_type,
        "in_channels": before_in,
        "out_channels": before_out,
        "groups": max(groups, 1),
        "source_before": source_before,
        "source_after": source_after,
    }
    if isinstance(weight_shape, Sequence) and not isinstance(weight_shape, (str, bytes)):
        spec["weight_shape"] = [int(value) for value in weight_shape]
    return spec


def _is_deblock_convtranspose_module(module_name: str, module_type: str) -> bool:
    low_name = str(module_name).lower()
    low_type = str(module_type).lower()
    return "convtranspose" in low_type and "pyramid_backbone.deblocks" in low_name and low_name.endswith(".0")


def _extract_layer_specs_from_payload(payload: Any) -> list[dict[str, Any]]:
    rows: list[Any] = []
    if isinstance(payload, list):
        rows = payload
    elif isinstance(payload, Mapping):
        for key in ("module_channel_before_after", "before_after_shapes", "module_shapes"):
            value = payload.get(key)
            if isinstance(value, list):
                rows.extend(value)
            elif isinstance(value, Mapping):
                rows.extend(value.values())
    specs_by_name: dict[str, dict[str, Any]] = {}
    seen: set[str] = set()
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        spec = _normalize_layer_spec(row)
        if not spec:
            continue
        if "conv" not in str(spec["module_type"]).lower():
            continue
        key = str(spec["module_name"])
        if key in seen:
            existing = specs_by_name[key]
            for field in ("weight_shape", "source_before", "source_after"):
                if field in spec and field not in existing:
                    existing[field] = spec[field]
            continue
        seen.add(key)
        specs_by_name[key] = spec
    return list(specs_by_name.values())


def _load_layer_specs_from_path(path: Path) -> list[dict[str, Any]]:
    candidates: list[Path]
    if path.is_dir():
        candidates = [path / "pruning_manifest.json", path / "module_channel_before_after.json"]
    else:
        candidates = [path]
    best: list[dict[str, Any]] = []
    for candidate in candidates:
        if not candidate.is_file():
            continue
        try:
            payload = json.loads(candidate.read_text(encoding="utf-8"))
        except Exception:
            continue
        specs = _extract_layer_specs_from_payload(payload)
        if len(specs) > len(best):
            best = specs
    return best


def _find_best_source_subnet(subnet_root: Path) -> Path | None:
    if not subnet_root.is_dir():
        return None
    best_path: Path | None = None
    best_count = -1
    for manifest in sorted(subnet_root.glob("subnet_*/pruning_manifest.json")):
        specs = _load_layer_specs_from_path(manifest.parent)
        if len(specs) > best_count:
            best_count = len(specs)
            best_path = manifest.parent
    return best_path


def load_layer_specs_for_cli(source_subnet_dir: Path | None, source_subnet_root: Path) -> tuple[list[dict[str, Any]], str]:
    if source_subnet_dir is not None:
        specs = _load_layer_specs_from_path(source_subnet_dir)
        return specs, str(source_subnet_dir)
    best = _find_best_source_subnet(source_subnet_root)
    if best is not None:
        specs = _load_layer_specs_from_path(best)
        if specs:
            return specs, str(best)
    raise FileNotFoundError(f"no readable source subnet manifest under {source_subnet_root}")


def write_source_layer_spec_audit(output_dir: str | Path, source_path: str | Path, layer_specs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    uses_old_after = False
    before_matches = 0
    comparable = 0
    for spec in layer_specs:
        before = spec.get("source_before") if isinstance(spec.get("source_before"), Mapping) else {}
        after = spec.get("source_after") if isinstance(spec.get("source_after"), Mapping) else {}
        loaded = {
            "in_channels": _as_int(spec.get("in_channels"), 0),
            "out_channels": _as_int(spec.get("out_channels"), 0),
            "groups": _as_int(spec.get("groups"), 1),
        }
        before_loaded = {
            "in_channels": _as_int(before.get("in_channels"), loaded["in_channels"]),
            "out_channels": _as_int(before.get("out_channels"), loaded["out_channels"]),
            "groups": _as_int(before.get("groups"), loaded["groups"]),
        }
        after_loaded = {
            "in_channels": _as_int(after.get("in_channels"), loaded["in_channels"]),
            "out_channels": _as_int(after.get("out_channels"), loaded["out_channels"]),
            "groups": _as_int(after.get("groups"), loaded["groups"]),
        }
        if before:
            comparable += 1
            if loaded == before_loaded:
                before_matches += 1
        if after and after_loaded != before_loaded and loaded == after_loaded:
            uses_old_after = True
        rows.append(
            {
                "module_name": str(spec.get("module_name", "")),
                "loaded_baseline": loaded,
                "source_before": before_loaded,
                "source_after": after_loaded,
                "loaded_matches_source_before": loaded == before_loaded,
                "loaded_matches_source_after": loaded == after_loaded,
            }
        )
    uses_original_before = comparable > 0 and before_matches == comparable
    audit = {
        "source_path": str(source_path),
        "uses_original_before_channels": bool(uses_original_before),
        "uses_old_subnet_after_channels": bool(uses_old_after),
        "num_layers": len(layer_specs),
        "first_20_layers": rows[:20],
        "verdict": "pass" if uses_original_before and not uses_old_after else "fail",
    }
    _write_json(Path(output_dir) / "source_layer_spec_audit.json", audit)
    return audit


def _sample_layer_prune_ratio(
    sampled_target_global_prune_ratio: float,
    target_bin: tuple[float, float],
    rng: random.Random,
    *,
    max_channel_prune_ratio: float,
) -> float:
    left, right = target_bin
    width = max(right - left, 0.02)
    jitter = rng.uniform(-0.25 * width, 0.25 * width)
    value = sampled_target_global_prune_ratio + jitter
    # Keep local variation inside the same broad bin so low-prune subnets do not
    # acquire an 80%-pruned outlier layer.
    return max(0.0, min(float(max_channel_prune_ratio), min(right, max(left, value))))


def _sample_layer(
    spec: Mapping[str, Any],
    rng: random.Random,
    *,
    sampled_target_global_prune_ratio: float,
    target_global_prune_bin: tuple[float, float],
    max_channel_prune_ratio: float,
    min_channel_keep_ratio: float,
    ordinary_conv_round_to: int,
    grouped_conv_safe_per_group: set[int],
    allow_deblock_output_pruning: bool = False,
    protect_deblock_output: bool = True,
) -> tuple[dict[str, Any], dict[str, Any], float]:
    module_name = str(spec["module_name"])
    module_type = str(spec.get("module_type", "Conv2d"))
    groups = max(_as_int(spec.get("groups"), 1), 1)
    before_in = _as_int(spec.get("in_channels"), 0)
    before_out = _as_int(spec.get("out_channels"), 0)
    snapped_from_target: dict[str, Any] | None = None
    skipped_reason = ""
    layer_target_prune = _sample_layer_prune_ratio(
        sampled_target_global_prune_ratio,
        target_global_prune_bin,
        rng,
        max_channel_prune_ratio=max_channel_prune_ratio,
    )
    if groups <= 1:
        after_in, target_keep_in = _target_after_channels(
            before_in,
            target_prune_ratio=layer_target_prune,
            min_channel_keep_ratio=min_channel_keep_ratio,
            round_to=ordinary_conv_round_to,
        )
        after_out, target_keep_out = _target_after_channels(
            before_out,
            target_prune_ratio=layer_target_prune,
            min_channel_keep_ratio=min_channel_keep_ratio,
            round_to=ordinary_conv_round_to,
        )
        reason = "ordinary_conv_round_to_4"
    else:
        before_in_per_group = before_in // groups if before_in % groups == 0 else 0
        before_out_per_group = before_out // groups if before_out % groups == 0 else 0
        if before_in_per_group == 4 and before_out_per_group == 4:
            after_in = before_in
            after_out = before_out
            target_keep_in = 1.0
            target_keep_out = 1.0
            reason = "original_per_group_4_preserved"
        elif before_in_per_group <= 0 or before_out_per_group <= 0:
            after_in = before_in
            after_out = before_out
            target_keep_in = 1.0
            target_keep_out = 1.0
            reason = "skipped_group_channels_not_divisible"
            skipped_reason = reason
        else:
            target_in = max(1, int(round(before_in_per_group * max(min_channel_keep_ratio, 1.0 - layer_target_prune))))
            target_out = max(1, int(round(before_out_per_group * max(min_channel_keep_ratio, 1.0 - layer_target_prune))))
            safe_in, reason_in = _nearest_safe_per_group(
                target_in,
                before_in_per_group,
                grouped_conv_safe_per_group,
                max_channel_prune_ratio=max_channel_prune_ratio,
                min_channel_keep_ratio=min_channel_keep_ratio,
            )
            safe_out, reason_out = _nearest_safe_per_group(
                target_out,
                before_out_per_group,
                grouped_conv_safe_per_group,
                max_channel_prune_ratio=max_channel_prune_ratio,
                min_channel_keep_ratio=min_channel_keep_ratio,
            )
            after_in = groups * safe_in
            after_out = groups * safe_out
            target_keep_in = target_in / max(before_in_per_group, 1)
            target_keep_out = target_out / max(before_out_per_group, 1)
            reason = "grouped_conv_safe_per_group"
            if reason_in.startswith("skipped") or reason_out.startswith("skipped"):
                after_in = before_in
                after_out = before_out
                skipped_reason = reason_in if reason_in.startswith("skipped") else reason_out
                reason = skipped_reason
            if safe_in != target_in or safe_out != target_out:
                snapped_from_target = {
                    "target_cin_per_group": target_in,
                    "target_cout_per_group": target_out,
                    "snapped_cin_per_group": safe_in,
                    "snapped_cout_per_group": safe_out,
                    "policy": "nearest_tie_keeps_more_channels",
                }
    protected_axes: list[str] = []
    protection_reason = ""
    if (
        protect_deblock_output
        and not allow_deblock_output_pruning
        and _is_deblock_convtranspose_module(module_name, module_type)
    ):
        after_out = before_out
        target_keep_out = 1.0
        protected_axes.append("out")
        protection_reason = DEBLOCK_OUTPUT_PROTECTION_REASON
    prune_ratio_in = (before_in - after_in) / max(before_in, 1)
    prune_ratio_out = (before_out - after_out) / max(before_out, 1)
    max_prune_ratio = max(prune_ratio_in, prune_ratio_out)
    manifest_row = {
        "module_name": module_name,
        "module_type": module_type,
        "before": {"in_channels": before_in, "out_channels": before_out, "groups": groups},
        "after": {"in_channels": after_in, "out_channels": after_out, "groups": groups},
        "sampled_layer_target_prune_ratio": layer_target_prune,
        "target_keep_ratio": {"in_channels": target_keep_in, "out_channels": target_keep_out},
        "channel_prune_ratio": {"in_channels": prune_ratio_in, "out_channels": prune_ratio_out, "max": max_prune_ratio},
        "skipped_reason": skipped_reason,
    }
    if protected_axes:
        manifest_row["protected_axes"] = protected_axes
        manifest_row["protection_reason"] = protection_reason
    if "weight_shape" in spec:
        manifest_row["weight_shape"] = list(spec["weight_shape"])
    if snapped_from_target:
        manifest_row["snapped_from_target"] = snapped_from_target
    cin_per_group = after_in // groups if groups > 0 and after_in % groups == 0 else None
    cout_per_group = after_out // groups if groups > 0 and after_out % groups == 0 else None
    before_cin_per_group = before_in // groups if groups > 0 and before_in % groups == 0 else None
    before_cout_per_group = before_out // groups if groups > 0 and before_out % groups == 0 else None
    int8_supported = groups == 1 or (cin_per_group in grouped_conv_safe_per_group and cout_per_group in grouped_conv_safe_per_group)
    eligibility = {
        "module_name": module_name,
        "groups": groups,
        "before_in": before_in,
        "before_out": before_out,
        "after_in": after_in,
        "after_out": after_out,
        "before_cin_per_group": before_cin_per_group,
        "before_cout_per_group": before_cout_per_group,
        "after_cin_per_group": cin_per_group,
        "after_cout_per_group": cout_per_group,
        "cin_per_group": cin_per_group,
        "cout_per_group": cout_per_group,
        "safe_per_group_set": sorted(grouped_conv_safe_per_group),
        "int8_shape_supported": bool(int8_supported),
        "reason": reason if int8_supported else "grouped_conv_int8_per_group_shape_not_supported",
        "snapping_policy": "nearest_tie_keeps_more_channels",
        "unsupported_reason": "" if int8_supported else "grouped_conv_int8_per_group_shape_not_supported",
        "whether_8_to_4": bool(before_cin_per_group == 8 and cin_per_group == 4 and before_cout_per_group == 8 and cout_per_group == 4),
        "whether_16_to_8_or_4": bool(
            before_cin_per_group == 16
            and cin_per_group in {4, 8}
            and before_cout_per_group == 16
            and cout_per_group in {4, 8}
        ),
    }
    if snapped_from_target:
        eligibility["snapped_from_target"] = snapped_from_target
    if skipped_reason:
        eligibility["skipped_reason"] = skipped_reason
    return manifest_row, eligibility, max_prune_ratio


def _before_after_shapes(module_channel_before_after: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    shapes: list[dict[str, Any]] = []
    for row in module_channel_before_after:
        before = dict(row["before"])
        after = dict(row["after"])
        shapes.append(
            {
                "module_name": row["module_name"],
                "before": {"module_type": row.get("module_type", "Conv2d"), "attrs": before},
                "after": {"module_type": row.get("module_type", "Conv2d"), "attrs": after},
            }
        )
    return shapes


def _grouped_per_group_distribution(eligibility_rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in eligibility_rows:
        if _as_int(row.get("groups"), 1) <= 1:
            continue
        key = f"{row.get('cin_per_group')}x{row.get('cout_per_group')}"
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def _conv_param_proxy(row: Mapping[str, Any], *, after: bool) -> int:
    before = row.get("before") if isinstance(row.get("before"), Mapping) else {}
    after_row = row.get("after") if isinstance(row.get("after"), Mapping) else {}
    selected = after_row if after else before
    in_channels = max(_as_int(selected.get("in_channels"), 0), 0)
    out_channels = max(_as_int(selected.get("out_channels"), 0), 0)
    groups = max(_as_int(selected.get("groups"), 1), 1)
    kernel_h = 1
    kernel_w = 1
    weight_shape = row.get("weight_shape")
    if isinstance(weight_shape, Sequence) and not isinstance(weight_shape, (str, bytes)) and len(weight_shape) >= 4:
        kernel_h = _as_int(weight_shape[2], 1)
        kernel_w = _as_int(weight_shape[3], 1)
    return int(out_channels * max(in_channels // groups, 1) * kernel_h * kernel_w)


def _global_prune_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    before_channels = 0
    after_channels = 0
    before_params = 0
    after_params = 0
    layer_ratios: list[float] = []
    for row in rows:
        before = row.get("before") if isinstance(row.get("before"), Mapping) else {}
        after = row.get("after") if isinstance(row.get("after"), Mapping) else {}
        before_channels += _as_int(before.get("in_channels"), 0) + _as_int(before.get("out_channels"), 0)
        after_channels += _as_int(after.get("in_channels"), 0) + _as_int(after.get("out_channels"), 0)
        before_params += _conv_param_proxy(row, after=False)
        after_params += _conv_param_proxy(row, after=True)
        ratio = row.get("channel_prune_ratio", {})
        if isinstance(ratio, Mapping):
            layer_ratios.append(float(ratio.get("max", 0.0) or 0.0))
    channel_prune = 1.0 - after_channels / max(before_channels, 1)
    param_prune = 1.0 - after_params / max(before_params, 1)
    return {
        "achieved_global_channel_prune_ratio": max(0.0, min(1.0, channel_prune)),
        "achieved_global_param_prune_ratio": max(0.0, min(1.0, param_prune)),
        "achieved_global_bops_prune_ratio": max(0.0, min(1.0, param_prune)),
        "mean_layer_prune_ratio": float(statistics.fmean(layer_ratios)) if layer_ratios else 0.0,
        "median_layer_prune_ratio": float(statistics.median(layer_ratios)) if layer_ratios else 0.0,
    }


def _write_markdown_report(output_dir: Path, result: Mapping[str, Any]) -> None:
    default_audit = result.get("default_config_audit", {})
    lines = [
        "# Random Deployment-Aware Subnet Sampling Dry Run",
        "",
        f"uses_taylor_ranking: {str(bool(result.get('uses_taylor_ranking'))).lower()}",
        f"random_sampling_method: {result.get('random_sampling_method')}",
        f"max_channel_prune_ratio: {result.get('max_channel_prune_ratio')}",
        f"min_channel_keep_ratio: {result.get('min_channel_keep_ratio')}",
        f"allow_deblock_output_pruning: {str(bool(result.get('allow_deblock_output_pruning'))).lower()}",
        f"protect_deblock_output: {str(bool(result.get('protect_deblock_output'))).lower()}",
        f"source_layer_spec: {result.get('source_layer_spec', '')}",
        "",
        "## Subnets",
        "",
        "| subnet_id | target_bin | target_global | structure_hash | shape_hash | max_single_layer_prune | global_channel | global_param | global_bops | unsupported_grouped_shapes |",
        "|---|---|---:|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in result.get("subnets", []):
        lines.append(
            f"| {row['subnet_id']} | {row.get('target_global_prune_bin', '')} | {float(row.get('sampled_target_global_prune_ratio', 0.0)):.4f} | "
            f"{row['structure_hash']} | {row['shape_hash']} | {float(row['max_single_layer_channel_prune_ratio']):.4f} | "
            f"{float(row.get('achieved_global_channel_prune_ratio', 0.0)):.4f} | {float(row.get('achieved_global_param_prune_ratio', 0.0)):.4f} | "
            f"{float(row.get('achieved_global_bops_prune_ratio', 0.0)):.4f} | {row['unsupported_grouped_conv_int8_shape_count']} |"
        )
    lines.extend(
        [
            "",
            "## Default Pruning Config Audit",
            "",
            f"old_taylor_pruner_max_ch_sparsity: {default_audit.get('old_taylor_pruner_max_ch_sparsity', 'unknown')}",
            f"old_yaml_min_keep_ratio: {default_audit.get('old_yaml_min_keep_ratio', 'unknown')}",
            f"new_random_sampler_max_channel_prune_ratio: {result.get('max_channel_prune_ratio')}",
            f"new_random_sampler_min_channel_keep_ratio: {result.get('min_channel_keep_ratio')}",
            f"default_decision: {default_audit.get('default_decision', '')}",
        ]
    )
    diversity = result.get("diversity_report", {})
    lines.extend(
        [
            "",
            "## Diversity",
            "",
            f"unique_structure_hash_count: {diversity.get('unique_structure_hash_count', 0)}",
            f"unique_shape_hash_count: {diversity.get('unique_shape_hash_count', 0)}",
            f"duplicate_reject_count: {diversity.get('duplicate_reject_count', 0)}",
            f"duplicate_structure_exists: {str(bool(diversity.get('duplicate_structure_exists'))).lower()}",
            f"old_subnet_after_shape_pollution: {str(bool((result.get('source_layer_spec_audit') or {}).get('uses_old_subnet_after_channels'))).lower()}",
            "",
            "## Grouped Conv Per-Group Distribution",
            "",
        ]
    )
    distribution = result.get("grouped_conv_per_group_distribution", {})
    if distribution:
        for key, count in distribution.items():
            lines.append(f"- {key}: {count}")
    else:
        lines.append("- none")
    unsupported = int(result.get("unsupported_grouped_conv_int8_shape_count", 0) or 0)
    lines.extend(
        [
            "",
            "## Recommendation",
            "",
            (
                "Proceed to ONNX + mixed profile dry-run gates before TensorRT build."
                if unsupported == 0 and not bool(diversity.get("duplicate_structure_exists"))
                else "Do not proceed to TensorRT build until unsupported grouped shapes or duplicates are resolved."
            ),
        ]
    )
    (output_dir / "random_subnet_sampling_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def generate_random_deployment_aware_subnets(
    *,
    layer_specs: Sequence[Mapping[str, Any]],
    output_dir: str | Path,
    num_subnets: int,
    seed: int,
    max_channel_prune_ratio: float = 0.80,
    min_channel_keep_ratio: float = 0.20,
    ordinary_conv_round_to: int = 4,
    grouped_conv_safe_per_group: set[int] | Sequence[int] = (4, 8, 16, 32),
    disable_taylor_ranking: bool = True,
    diversity_reject_duplicates: bool = True,
    retry_multiplier: int = 5,
    source_layer_spec: str = "",
    global_target_prune_bins: str | Sequence[tuple[float, float]] | None = None,
    subnets_per_bin: int | None = None,
    allow_deblock_output_pruning: bool = False,
    protect_deblock_output: bool = True,
) -> dict[str, Any]:
    if not disable_taylor_ranking:
        raise ValueError("deployment-aware random sampler requires disable_taylor_ranking=True")
    specs = [_normalize_layer_spec(row) for row in layer_specs]
    specs = [row for row in specs if row is not None]
    if not specs:
        raise ValueError("layer_specs must contain at least one Conv-like layer")
    safe_set = _parse_safe_set(grouped_conv_safe_per_group)
    bins = parse_global_target_prune_bins(global_target_prune_bins)
    bin_schedule = _subnet_bin_schedule(int(num_subnets), bins, subnets_per_bin=subnets_per_bin)
    output_dir = Path(output_dir)
    subnets_dir = output_dir / "subnets"
    subnets_dir.mkdir(parents=True, exist_ok=True)
    source_audit = write_source_layer_spec_audit(output_dir, source_layer_spec, specs) if source_layer_spec else {}
    accepted: list[dict[str, Any]] = []
    all_eligibility_rows: list[dict[str, Any]] = []
    seen_structure_hashes: set[str] = set()
    seen_shape_hashes: set[str] = set()
    duplicate_reject_count = 0
    max_attempts = max(int(num_subnets), int(num_subnets) * max(int(retry_multiplier), 1))
    attempt = 0
    while len(accepted) < int(num_subnets) and attempt < max_attempts:
        subnet_index = len(accepted)
        subnet_seed = int(seed) + attempt * 1009 + subnet_index * 9176
        rng = random.Random(subnet_seed)
        target_bin = bin_schedule[subnet_index % len(bin_schedule)]
        sampled_target = rng.uniform(target_bin[0], target_bin[1])
        rows: list[dict[str, Any]] = []
        eligibility_rows: list[dict[str, Any]] = []
        max_layer_prune = 0.0
        for spec in specs:
            manifest_row, eligibility, layer_max = _sample_layer(
                spec,
                rng,
                sampled_target_global_prune_ratio=sampled_target,
                target_global_prune_bin=target_bin,
                max_channel_prune_ratio=float(max_channel_prune_ratio),
                min_channel_keep_ratio=float(min_channel_keep_ratio),
                ordinary_conv_round_to=int(ordinary_conv_round_to),
                grouped_conv_safe_per_group=safe_set,
                allow_deblock_output_pruning=bool(allow_deblock_output_pruning),
                protect_deblock_output=bool(protect_deblock_output),
            )
            rows.append(manifest_row)
            eligibility_rows.append(eligibility)
            max_layer_prune = max(max_layer_prune, float(layer_max))
        hash_rows = [
            {
                "module_name": row["module_name"],
                "module_type": row["module_type"],
                "before": row["before"],
                "after": row["after"],
            }
            for row in rows
        ]
        structure_hash = _stable_hash({"method": RANDOM_SAMPLING_METHOD, "rows": hash_rows})
        shape_hash = _stable_hash({"rows": [{"module_name": row["module_name"], "after": row["after"]} for row in rows]})
        if diversity_reject_duplicates and (structure_hash in seen_structure_hashes or shape_hash in seen_shape_hashes):
            duplicate_reject_count += 1
            attempt += 1
            continue
        seen_structure_hashes.add(structure_hash)
        seen_shape_hashes.add(shape_hash)
        subnet_id = f"subnet_{len(accepted):03d}"
        subnet_dir = subnets_dir / subnet_id
        unsupported_count = sum(1 for row in eligibility_rows if not bool(row.get("int8_shape_supported")))
        global_metrics = _global_prune_metrics(rows)
        manifest = {
            "subnet_id": subnet_id,
            "dry_run": True,
            "structure_hash": structure_hash,
            "shape_hash": shape_hash,
            "random_seed": subnet_seed,
            "target_global_prune_bin": _bin_label(target_bin),
            "sampled_target_global_prune_ratio": sampled_target,
            "random_sampling_method": RANDOM_SAMPLING_METHOD,
            "uses_taylor_ranking": False,
            "disable_taylor_ranking": True,
            "max_channel_prune_ratio": float(max_channel_prune_ratio),
            "min_channel_keep_ratio": float(min_channel_keep_ratio),
            "ordinary_conv_round_to": int(ordinary_conv_round_to),
            "round_to": int(ordinary_conv_round_to),
            "grouped_conv_safe_per_group": sorted(safe_set),
            "allow_deblock_output_pruning": bool(allow_deblock_output_pruning),
            "protect_deblock_output": bool(protect_deblock_output),
            "source_layer_spec": source_layer_spec,
            "module_channel_before_after": rows,
            "before_after_shapes": _before_after_shapes(rows),
            "max_single_layer_channel_prune_ratio": max_layer_prune,
            **global_metrics,
            "unsupported_grouped_conv_int8_shape_count": unsupported_count,
            "pruned_model_object": None,
            "note": "dry_run_manifest_only_no_physical_prune_no_engine_build_no_eval",
        }
        _write_json(subnet_dir / "pruning_manifest.json", manifest)
        _write_json(subnet_dir / "module_channel_before_after.json", rows)
        _write_json(subnet_dir / "grouped_conv_int8_eligibility_report.json", eligibility_rows)
        summary_row = {
            "subnet_id": subnet_id,
            "subnet_dir": str(subnet_dir),
            "manifest_path": str(subnet_dir / "pruning_manifest.json"),
            "structure_hash": structure_hash,
            "shape_hash": shape_hash,
            "random_seed": subnet_seed,
            "target_global_prune_bin": _bin_label(target_bin),
            "sampled_target_global_prune_ratio": sampled_target,
            "random_sampling_method": RANDOM_SAMPLING_METHOD,
            "max_single_layer_channel_prune_ratio": max_layer_prune,
            **global_metrics,
            "unsupported_grouped_conv_int8_shape_count": unsupported_count,
        }
        accepted.append(summary_row)
        all_eligibility_rows.extend(eligibility_rows)
        attempt += 1
    diversity_report = {
        "requested_subnet_count": int(num_subnets),
        "accepted_subnet_count": len(accepted),
        "attempt_count": attempt,
        "duplicate_reject_count": duplicate_reject_count,
        "unique_structure_hash_count": len({row["structure_hash"] for row in accepted}),
        "unique_shape_hash_count": len({row["shape_hash"] for row in accepted}),
        "duplicate_structure_exists": len({row["structure_hash"] for row in accepted}) != len(accepted),
        "duplicate_shape_exists": len({row["shape_hash"] for row in accepted}) != len(accepted),
        "diversity_reject_duplicates": bool(diversity_reject_duplicates),
    }
    result = {
        "uses_taylor_ranking": False,
        "random_sampling_method": RANDOM_SAMPLING_METHOD,
        "max_channel_prune_ratio": float(max_channel_prune_ratio),
        "min_channel_keep_ratio": float(min_channel_keep_ratio),
        "ordinary_conv_round_to": int(ordinary_conv_round_to),
        "grouped_conv_safe_per_group": sorted(safe_set),
        "allow_deblock_output_pruning": bool(allow_deblock_output_pruning),
        "protect_deblock_output": bool(protect_deblock_output),
        "global_target_prune_bins": [_bin_label(bin_range) for bin_range in bins],
        "source_layer_spec": source_layer_spec,
        "source_layer_spec_audit": source_audit,
        "subnets": accepted,
        "diversity_report": diversity_report,
        "default_config_audit": {
            "old_taylor_pruner_max_ch_sparsity": "0.60 from pruning/config.py PruningConfig.max_ch_sparsity",
            "old_yaml_min_keep_ratio": "0.5 from pruning/configs/lidar_pyramid_pruning_default.yaml",
            "default_decision": "new random sampler defaults added as max_channel_prune_ratio=0.80 and min_channel_keep_ratio=0.20; old Taylor/greedy defaults are not changed",
        },
        "grouped_conv_per_group_distribution": _grouped_per_group_distribution(all_eligibility_rows),
        "unsupported_grouped_conv_int8_shape_count": sum(1 for row in all_eligibility_rows if not bool(row.get("int8_shape_supported"))),
    }
    _write_json(output_dir / "diversity_report.json", diversity_report)
    _write_json(output_dir / "results.json", result)
    _write_csv(output_dir / "results.csv", accepted)
    _write_markdown_report(output_dir, result)
    return result


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-subnets", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-subnet-dir", type=Path, default=None)
    parser.add_argument("--source-subnet-root", type=Path, default=DEFAULT_OLD_V11_SUBNET_ROOT)
    parser.add_argument("--max-channel-prune-ratio", type=float, default=0.80)
    parser.add_argument("--min-channel-keep-ratio", type=float, default=0.20)
    parser.add_argument("--ordinary-conv-round-to", type=int, default=4)
    parser.add_argument("--grouped-conv-safe-per-group", type=str, default="4,8,16,32")
    parser.add_argument("--global-target-prune-bins", type=str, default="0.0:0.2,0.2:0.4,0.4:0.6,0.6:0.8")
    parser.add_argument("--subnets-per-bin", type=int, default=0)
    parser.add_argument("--allow-deblock-output-pruning", type=str, default="false")
    parser.add_argument("--protect-deblock-output", type=str, default="true")
    parser.add_argument("--disable-taylor-ranking", type=str, default="true")
    parser.add_argument("--diversity-reject-duplicates", type=str, default="true")
    return parser


def _str2bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    layer_specs, source = load_layer_specs_for_cli(args.source_subnet_dir, args.source_subnet_root)
    result = generate_random_deployment_aware_subnets(
        layer_specs=layer_specs,
        output_dir=args.output_dir,
        num_subnets=args.num_subnets,
        seed=args.seed,
        max_channel_prune_ratio=args.max_channel_prune_ratio,
        min_channel_keep_ratio=args.min_channel_keep_ratio,
        ordinary_conv_round_to=args.ordinary_conv_round_to,
        grouped_conv_safe_per_group=_parse_safe_set(args.grouped_conv_safe_per_group),
        disable_taylor_ranking=_str2bool(args.disable_taylor_ranking),
        diversity_reject_duplicates=_str2bool(args.diversity_reject_duplicates),
        source_layer_spec=source,
        global_target_prune_bins=parse_global_target_prune_bins(args.global_target_prune_bins),
        subnets_per_bin=int(args.subnets_per_bin) if int(args.subnets_per_bin) > 0 else None,
        allow_deblock_output_pruning=_str2bool(args.allow_deblock_output_pruning),
        protect_deblock_output=_str2bool(args.protect_deblock_output),
    )
    print(f"wrote {len(result['subnets'])} random deployment-aware dry-run subnets to {args.output_dir}")
    print(f"uses_taylor_ranking={str(result['uses_taylor_ranking']).lower()}")
    print(f"unique_structure_hash_count={result['diversity_report']['unique_structure_hash_count']}")
    print(f"unsupported_grouped_conv_int8_shape_count={result['unsupported_grouped_conv_int8_shape_count']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
