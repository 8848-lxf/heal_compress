"""Compact evidence reporter for the RTX 4090 Transformer alignment run."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


_COMPACT_FIELDS = (
    "model",
    "family",
    "candidate_id",
    "structure_kind",
    "d_h",
    "heads",
    "projection_width",
    "profile",
    "structure_hash",
    "onnx_sha256",
    "engine_sha256",
    "scale_hash",
    "requested_realized_conflict_count",
    "evaluated",
    "skipped",
    "AP30",
    "AP50",
    "AP70",
    "mAP",
    "delta_structure",
    "delta_precision_base",
    "delta_precision_candidate",
    "interaction",
    "p50_ms",
    "p90_ms",
    "p95_ms",
    "p99_ms",
    "structure_speedup",
    "precision_speedup",
    "total_speedup",
    "neighbor_control_speedup",
    "tactic_signature",
    "fusion_signature",
    "kernel_count",
    "cast_count",
    "reformat_count",
    "accuracy_class",
    "latency_class",
    "build_repeat_stable",
    "search_space_candidate",
    "diagnostic_only",
)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({str(key) for row in rows for key in row})
    if not fields:
        fields = ["status"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(value, sort_keys=True)
                    if isinstance(value, (dict, list, tuple))
                    else value
                    for key, value in row.items()
                }
            )


def accuracy_class(
    delta_map: float,
    *,
    evaluated: int,
    skipped: int,
    finite: bool,
) -> str:
    if int(evaluated) != 500 or int(skipped) != 0 or not bool(finite):
        return "INVALID_EVALUATION"
    delta = float(delta_map)
    if delta < -0.010:
        return "UNSAFE"
    if delta < -0.003:
        return "BORDERLINE"
    return "SAFE"


def precision_interaction(
    *,
    baseline_p32_map: float,
    baseline_profile_map: float,
    candidate_p32_map: float,
    candidate_profile_map: float,
) -> dict[str, float]:
    baseline_p32 = float(baseline_p32_map)
    baseline_profile = float(baseline_profile_map)
    candidate_p32 = float(candidate_p32_map)
    candidate_profile = float(candidate_profile_map)
    delta_structure = candidate_p32 - baseline_p32
    delta_precision_base = baseline_profile - baseline_p32
    delta_precision_candidate = candidate_profile - candidate_p32
    return {
        "delta_structure": delta_structure,
        "delta_precision_base": delta_precision_base,
        "delta_precision_candidate": delta_precision_candidate,
        "delta_total": candidate_profile - baseline_p32,
        "interaction": delta_precision_candidate - delta_precision_base,
    }


def build_repeat_stability(
    reduction_ratios: Sequence[float], *, required_reduction: float
) -> dict[str, Any]:
    values = tuple(float(value) for value in reduction_ratios)
    if len(values) != 3:
        raise ValueError("three_independent_builds_required")
    threshold = float(required_reduction)
    passing = sum(value > threshold for value in values)
    no_slowdown = min(values) >= 0.0
    stable = passing == 3 or (passing >= 2 and no_slowdown)
    return {
        "build_repeat_stable": stable,
        "passing_builds": passing,
        "required_reduction_ratio": threshold,
        "minimum_reduction_ratio": min(values),
        "maximum_reduction_ratio": max(values),
        "reductions": list(values),
    }


def compact_evidence_record(row: Mapping[str, Any]) -> dict[str, Any]:
    required = (
        "candidate_id",
        "structure_hash",
        "onnx_sha256",
        "engine_sha256",
        "requested_realized_conflict_count",
        "evaluated",
        "skipped",
    )
    missing = [key for key in required if row.get(key) in (None, "")]
    if missing:
        raise RuntimeError(f"compact_evidence_missing:{','.join(missing)}")
    if int(row["requested_realized_conflict_count"]) != 0:
        raise RuntimeError("compact_evidence_precision_conflict")
    return {key: row[key] for key in _COMPACT_FIELDS if key in row}


def _interaction_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    index = {
        (str(row.get("model")), str(row.get("structure_hash")), str(row.get("profile"))): row
        for row in rows
        if row.get("mAP") is not None
    }
    baselines = {
        (str(row.get("model")), str(row.get("profile"))): row
        for row in rows
        if row.get("structure_kind") == "baseline" and row.get("mAP") is not None
    }
    result: list[dict[str, Any]] = []
    for row in rows:
        profile = str(row.get("profile"))
        if profile == "P32" or row.get("mAP") is None:
            continue
        model = str(row.get("model"))
        structure_hash = str(row.get("structure_hash"))
        candidate_p32 = index.get((model, structure_hash, "P32"))
        baseline_p32 = baselines.get((model, "P32"))
        baseline_profile = baselines.get((model, profile))
        if not candidate_p32 or not baseline_p32 or not baseline_profile:
            continue
        metrics = precision_interaction(
            baseline_p32_map=float(baseline_p32["mAP"]),
            baseline_profile_map=float(baseline_profile["mAP"]),
            candidate_p32_map=float(candidate_p32["mAP"]),
            candidate_profile_map=float(row["mAP"]),
        )
        result.append(
            {
                "model": model,
                "family": row.get("family"),
                "candidate_id": row.get("candidate_id"),
                "structure_hash": structure_hash,
                "profile": profile,
                **metrics,
            }
        )
    return result


def _contract(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {
        "schema_version": "transformer-dh-power-alignment-4090-contract-v1",
        "execution_platform": "RTX4090_SM89",
        "models": {},
        "ga_executed": False,
        "greedy_executed": False,
        "full1789_executed": False,
    }
    for model in sorted({str(row.get("model")) for row in rows if row.get("model")}):
        selected = [row for row in rows if str(row.get("model")) == model]
        profiles: dict[str, dict[str, list[Any]]] = {}
        for profile in ("P32", "P16", "P8"):
            profile_rows = [row for row in selected if row.get("profile") == profile]
            profiles[profile] = {
                "allowed": [row.get("candidate_id") for row in profile_rows if row.get("search_space_candidate") is True],
                "experimental": [
                    row.get("candidate_id")
                    for row in profile_rows
                    if row.get("search_space_candidate") is not True
                    and row.get("accuracy_class") in {"SAFE", "BORDERLINE"}
                ],
                "rejected": [
                    row.get("candidate_id")
                    for row in profile_rows
                    if row.get("accuracy_class") in {"UNSAFE", "INVALID_EVALUATION"}
                ],
            }
        result["models"][model] = {"profiles": profiles}
    return result


def write_power_alignment_reports(
    output_root: Path,
    *,
    candidate_rows: Sequence[Mapping[str, Any]],
    single_family_rows: Sequence[Mapping[str, Any]],
    joint_rows: Sequence[Mapping[str, Any]],
    build_repeat_rows: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    root = Path(output_root).resolve()
    single = [dict(row) for row in single_family_rows]
    joint = [dict(row) for row in joint_rows]
    all_rows = [*single, *joint]
    fixed_single = [row for row in single if int(row.get("evaluated", 0)) == 500]
    fixed_joint = [row for row in joint if int(row.get("evaluated", 0)) == 500]
    latency_single = [row for row in single if row.get("p50_ms") is not None]
    latency_joint = [row for row in joint if row.get("p50_ms") is not None]
    interactions = _interaction_rows(all_rows)

    outputs = {
        "power_alignment_candidate_manifest.csv": candidate_rows,
        "power_alignment_single_family_fixed500.csv": fixed_single,
        "power_alignment_single_family_latency.csv": latency_single,
        "power_alignment_joint_fixed500.csv": fixed_joint,
        "power_alignment_joint_latency.csv": latency_joint,
        "power_alignment_same_profile_speedup.csv": [row for row in all_rows if row.get("structure_speedup") is not None],
        "power_alignment_fp32_reference_speedup.csv": [row for row in all_rows if row.get("total_speedup") is not None],
        "power_alignment_neighbor_controls.csv": [row for row in all_rows if row.get("neighbor_control_speedup") is not None],
        "power_alignment_build_repeat_stability.csv": build_repeat_rows,
        "power_alignment_tactic_transitions.csv": [row for row in all_rows if row.get("tactic_signature")],
        "power_alignment_precision_interaction.csv": interactions,
    }
    for name, rows in outputs.items():
        _write_csv(root / name, list(rows))

    accuracy_boundary = {
        status: [row.get("candidate_id") for row in all_rows if row.get("accuracy_class") == status]
        for status in ("SAFE", "BORDERLINE", "UNSAFE", "INVALID_EVALUATION")
    }
    latency_boundary = {
        status: [row.get("candidate_id") for row in all_rows if row.get("latency_class") == status]
        for status in ("HIGH_ALIGNMENT_BENEFICIAL", "ALIGNMENT_ADVANTAGE", "TACTIC_UNSTABLE", "NO_BENEFIT")
    }
    contract = _contract(all_rows)
    _write_json(root / "power_alignment_accuracy_boundary.json", accuracy_boundary)
    _write_json(root / "power_alignment_latency_boundary.json", latency_boundary)
    _write_json(root / "power_alignment_contract.json", contract)
    summary = {
        "candidate_rows": len(candidate_rows),
        "single_family_rows": len(single),
        "joint_rows": len(joint),
        "single_family_fixed500_rows": len(fixed_single),
        "joint_fixed500_rows": len(fixed_joint),
        "formal_latency_rows": len(latency_single) + len(latency_joint),
        "search_space_candidates": sum(row.get("search_space_candidate") is True for row in all_rows),
    }
    _write_json(root / "reports" / "report_summary.json", summary)
    (root / "root_conclusion_power_alignment.md").write_text(
        "# RTX 4090 Transformer d_h power-alignment conclusion\n\n"
        f"- candidate manifest rows: `{summary['candidate_rows']}`\n"
        f"- single-family result rows: `{summary['single_family_rows']}`\n"
        f"- joint result rows: `{summary['joint_rows']}`\n"
        f"- fixed500 rows: `{summary['single_family_fixed500_rows'] + summary['joint_fixed500_rows']}`\n"
        f"- formal latency rows: `{summary['formal_latency_rows']}`\n"
        f"- SEARCH_SPACE_CANDIDATE rows: `{summary['search_space_candidates']}`\n"
        "- execution platform: `RTX4090_SM89`\n"
        "- P8 means SmoothQuant INT8 Q/K projection, not FP8.\n"
        "- GA/Greedy/full1789 were not executed by this experiment.\n"
        "- Detailed conclusions are evidence-derived in `power_alignment_contract.json`; "
        "missing evidence remains rejected or experimental.\n",
        encoding="utf-8",
    )
    return {"summary": summary, "contract": contract}


__all__ = [
    "accuracy_class",
    "build_repeat_stability",
    "compact_evidence_record",
    "precision_interaction",
    "write_power_alignment_reports",
]
